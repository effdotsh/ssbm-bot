import gym
from gym import spaces

import numpy as np
class TestEnv(gym.Env):
    def __init__(self):
        self.state = np.random.rand(3)
        self.observation_space = spaces.Box(shape=self.state.shape, low=-1, high = 1)
        self.action_space = spaces.Discrete(2)
        self.rewards = []
    def step(self, action: int):
        r=0
        if((action == 0 and self.state[0] > self.state[1]) or (action == 1 and self.state[1] > self.state[0])):
            r=1

        self.state=np.random.rand(3)
        self.rewards.append(r)
        return [self.state, r, True, {}]
    def reset(self):
        self.state=np.random.rand(3)
        return self.state
    def render(self):
        pass