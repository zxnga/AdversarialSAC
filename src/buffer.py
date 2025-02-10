from typing import Union, Optional

import torch as th
import numpy as np
from gymnasium import spaces

from stable_baselines3.common.vec_env import VecNormalize

class RewardBufferSamples(NamedTuple):
    rewards: th.Tensor

class RewardBuffer(BaseBuffer):
    rewards: np.ndarray

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)
        self.reset()

    def reset(self) -> None:
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        super().reset()

    def add(self, reward: np.ndarray) -> None:
        # print('pos', self.pos)
        self.rewards[self.pos] = np.array(reward)
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> RewardBufferSamples:
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))
        data = (
            self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env),
        )
        return RewardBufferSamples(*tuple(map(self.to_torch, data)))