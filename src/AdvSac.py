from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type, TypeVar, Union, NamedTuple

import sys
import os
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import copy

import torch as th

from gymnasium import spaces
from gymnasium.spaces import Box

from stable_baselines3 import SAC
from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.common.type_aliases import GymEnv, TrainFreq, RolloutReturn
from stable_baselines3.common.vec_env import VecEnv, VecNormalize
from stable_baselines3.common.buffers import ReplayBuffer, BaseBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.utils import should_collect_more_steps


class AdvSAC(SAC):
    def __init__(
        self,
        policy: Union[str, Type[SACPolicy]],
        env: Union[GymEnv, str],
        adv_agent = None, # needs a .predict() method
        evaluation_interval: int = 10,
        clip_reward_factor: bool = False,
        numerical_baseline: Optional[Union[List, float]] = None, #sum at end of ep or list of values
        *args,
        **kwargs,
    ):
        super().__init__(
            policy,
            env,
            *args,
            **kwargs,
        )
        assert (num is None) != (adv is None), 'Choose between an adversarial agent or a numerical baseline'

        self.evaluation_interval = evaluation_interval
        self.adv_agent = adv_agent
        self.numerical_baseline = numerical_baseline
        self.clip_reward_factor = clip_reward_factor
        self.policy_b_rewards = RewardBuffer(
            env.time_steps-1,
            env.observation_space,
            env.action_space
        )

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        train_freq: TrainFreq,
        replay_buffer: ReplayBuffer,
        action_noise: Optional[ActionNoise] = None,
        learning_starts: int = 0,
        log_interval: Optional[int] = None
    ) -> RolloutReturn:
        """
            Collect experiences and store them into a ``ReplayBuffer``.

            :param env: The training environment
            :param callback: Callback that will be called at each step
                (and at the beginning and end of the rollout)
            :param train_freq: How much experience to collect
                by doing rollouts of current policy.
                Either ``TrainFreq(<n>, TrainFrequencyUnit.STEP)``
                or ``TrainFreq(<n>, TrainFrequencyUnit.EPISODE)``
                with ``<n>`` being an integer greater than 0.
            :param action_noise: Action noise that will be used for exploration
                Required for deterministic policy (e.g. TD3). This can also be used
                in addition to the stochastic policy for SAC.
            :param learning_starts: Number of steps before learning for the warm-up phase.
            :param replay_buffer:
            :param log_interval: Log data every ``log_interval`` episodes
            :param evaluation_interval: Frequency of evaluations to adjust rewards
            :return:
        """
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)
        # print('ts', self.num_timesteps)
        # print('env_ts',env.envs[0].time_step)

        num_collected_steps, num_collected_episodes = 0, 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert train_freq.frequency > 0, "Should at least collect one step or episode."

        if env.num_envs > 1:
            assert train_freq.unit == TrainFrequencyUnit.STEP, "You must use only one env when doing episodic training."

        if self.use_sde:
            self.actor.reset_noise(env.num_envs)

        callback.on_rollout_start()
        continue_training = True
        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
            if self.use_sde and self.sde_sample_freq > 0 and num_collected_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.actor.reset_noise(env.num_envs)

            # Select action randomly or according to policy
            actions, buffer_actions = self._sample_action(learning_starts, action_noise, env.num_envs)

            # Rescale and perform action
            new_obs, rewards, dones, infos = env.step(actions)

            self.num_timesteps += env.num_envs
            num_collected_steps += 1

            # Give access to local variables
            callback.update_locals(locals())
            # Only stop training if return value is False, not when it is None.
            if not callback.on_step():
                return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training=False)

            # Retrieve reward and episode length if using Monitor wrapper
            self._update_info_buffer(infos, dones)

            self.policy_b_rewards.add(rewards)
            
            self._store_transition(replay_buffer, buffer_actions, new_obs, rewards, dones, infos)
            self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

            # For DQN, check if the target network should be updated
            # and update the exploration schedule
            # For SAC/TD3, the update is done at the same time as the gradient update
            # see https://github.com/hill-a/stable-baselines/issues/900
            self._on_step()

            for idx, done in enumerate(dones):
                if done:
                    # Update stats
                    num_collected_episodes += 1
                    self._episode_num += 1

                    if action_noise is not None:
                        kwargs = dict(indices=[idx]) if env.num_envs > 1 else {}
                        action_noise.reset(**kwargs)

                    # Log training infos
                    if log_interval is not None and self._episode_num % log_interval == 0:
                        self._dump_logs()

                    # Perform periodic evaluation
                    if self._episode_num % self.evaluation_interval == 0:
                        print('eval')
                        # print(f'evaluation {self._episode_num}, {num_collected_steps}')
                        # Evaluate Policy A
                        if self.adv_agent:
                            env_copy = copy.deepcopy(env)
                            policy_a_rewards = []

                            obs = env_copy.reset()
                            done_a = False
                            while not done_a:
                                action_a = self.adv_agent.predict(obs, deterministic=True)
                                obs, reward_a, done_a, _ = env_copy.step(action_a)
                                policy_a_rewards.append(reward_a)

                        elif self.numerical_baseline:
                            policy_a_rewards.append(reward_a)
                        
                        else:
                            raise NotImplementedError

                        # Calculate cumulative rewards
                        cumulative_reward_a = np.sum(policy_a_rewards).item()
                        cumulative_reward_b = np.sum(self.policy_b_rewards.rewards).item()
                        print(f'cumulative_reward_b {cumulative_reward_b}')
                        print(f'cumulative_reward_a {cumulative_reward_a}')

                        # Calculate the reward adjustment factor
                        reward_factor = cumulative_reward_b / cumulative_reward_a if cumulative_reward_a != 0 else 1

                        self.logger.record("Cumulative Reward/Policy B", cumulative_reward_b)
                        self.logger.record("Cumulative Reward/Policy A", cumulative_reward_a)
                        self.logger.record('Cumulative Reward/True Reward Factor', reward_factor) #want it to be < 0

                        if self.clip_reward_factor:
                            reward_factor = max(0.5, min(reward_factor, 1.5))
                            self.logger.record("Cumulative Reward/Cliped Reward Factor", reward_factor)
                        print(f'reward_factor {reward_factor}')

                        start = self.replay_buffer.pos - len(self.policy_b_rewards.rewards)
                        end = self.replay_buffer.pos
                        # Adjust the reward proportionally
                        self.replay_buffer.rewards[start:end] *= reward_factor

                        # Clear episode-specific buffer for the next episode
                        self.policy_b_rewards.reset()
                    self.policy_b_rewards.reset()  # Reset cumulative reward for Policy B
        callback.on_rollout_end()

        return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training)
