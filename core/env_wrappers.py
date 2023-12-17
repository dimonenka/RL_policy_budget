from gymnasium import Wrapper
from gymnasium.spaces import Box
import numpy as np


class ResourceN(Wrapper):
    def __init__(self, env, n_agents=None, low=0, high=1, n_policies=1, base_reward=False, seed=None):
        super().__init__(env)
        self.n_policies = n_policies
        self.low, self.high = low, high
        self.reward_length = (self.high - self.low) / self.n_policies / 2
        self.reward_coef = 1
        self.base_reward = base_reward
        self.init_agents()
        self.observation_space = Box(low=0.0, high=5.0, shape=(4 + self.n_agents,), dtype=np.int32)
        self.reset()

    def init_agents(self):
        x, y = self.map.shape
        self.targets = np.array(np.meshgrid(np.arange(x), np.arange(y))).T.reshape(-1, 2)
        self.n_agents = self.targets.shape[0]

    def step(self, action):
        self.timestep += 1
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        info['loc'] = 0
        reward = self.get_modified_reward(done) if not self.base_reward else self.get_base_reward(reward)
        return self.get_modified_obs(obs), reward, done, info

    def get_modified_obs(self, obs):
        return np.concatenate((obs, self.visited.reshape(-1,)))

    def get_modified_reward(self, done):
        self.visited[self.current_pos[0], self.current_pos[1]] = 1
        reward = np.zeros(self.n_agents)
        if done and np.array_equal(self.current_pos, self.initial_pos):
            r = 1 - self.timestep / 100
            for i, (x, y) in enumerate(self.targets):
                reward[i] = r * self.visited[x, y]
            return reward
        return reward

    def get_base_reward(self, reward):
        return np.full((self.n_agents,), reward.sum().item())

    def reset(self):
        self.timestep = 0
        self.visited = np.zeros(self.map.shape, dtype=np.int32)
        return self.get_modified_obs(self.env.reset()[0])


class CartPoleNLocWrapper(Wrapper):
    def __init__(self, env, n_agents, low=-0.5, high=0.5, n_policies=1, base_reward=False, seed=42):
        super().__init__(env)
        self.n_agents = n_agents
        self.low, self.high = low, high
        self.n_policies = n_policies
        self.reward_coef = 0.1
        self.base_reward = base_reward
        self.init_agents(seed)

    @property
    def reward_length(self):
        return (self.high - self.low) / 20

    def init_agents(self, seed=42):
        np.random.seed(seed)
        self.targets = np.random.uniform(self.low, self.high, self.n_agents)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        loc = self.get_loc(obs, info)
        if not self.base_reward:
            reward = self.get_modified_reward(loc, reward)
        else:
            reward = self.get_base_reward(reward)
        return obs, reward, done, info

    def get_loc(self, obs, info):
        loc = obs[0]
        info['loc'] = loc
        return loc

    def get_modified_reward(self, loc, reward):
        return (np.abs(loc - self.targets) < self.reward_length).astype(int) * self.reward_coef

    def get_base_reward(self, reward):
        return np.full((self.n_agents,), reward)

    def reset(self):
        return self.env.reset()[0]


class MujocoNSpeedWrapper(CartPoleNLocWrapper):
    def get_loc(self, obs, info):
        loc = info['x_velocity']
        info['loc'] = loc
        return loc

    def get_modified_reward(self, loc, reward):
        reward_velocity = np.clip(1 - np.abs(loc - self.targets) / self.reward_length, 0, 1)
        reward = reward_velocity * self.reward_coef
        return reward
