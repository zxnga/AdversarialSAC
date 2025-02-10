import gym
import copy

class PolicyComparisonWrapper(gym.Wrapper):
    def __init__(self, env, policy_a, evaluation_interval=10, reward_weight=0.1):
        super(PolicyComparisonWrapper, self).__init__(env)
        self.policy_a = policy_a
        self.evaluation_interval = evaluation_interval
        self.reward_weight = reward_weight
        self.episode_count = 0
        self.evaluating = False

    def step(self, action_b):
        obs_b, reward_b, done_b, info_b = self.env.step(action_b)
        
        if self.evaluating:
            action_a = self.policy_a.select_action(self.env_copy)
            obs_a, reward_a, done_a, info_a = self.env_copy.step(action_a)
            
            self.cumulative_reward_b += reward_b
            self.cumulative_reward_a += reward_a

            if done_b:
                if self.cumulative_reward_b > self.cumulative_reward_a:
                    adjusted_reward_b = self.cumulative_reward_b + self.reward_weight
                else:
                    adjusted_reward_b = self.cumulative_reward_b - self.reward_weight
                
                return obs_b, adjusted_reward_b, done_b, info_b

        return obs_b, reward_b, done_b, info_b

    def reset(self, **kwargs):
        self.episode_count += 1
        self.evaluating = (self.episode_count % self.evaluation_interval == 0)
        
        if self.evaluating:
            self.env_copy = copy.deepcopy(self.env)
            self.cumulative_reward_b = 0
            self.cumulative_reward_a = 0

        return self.env.reset(**kwargs)