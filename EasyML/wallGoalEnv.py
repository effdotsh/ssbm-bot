import math
import random

import gym
from gym import spaces

import numpy as np


class NavEnv(gym.Env):
    def __init__(self):
        self.agent_x = 0.
        self.agent_y = 0.
        self.target_x = 0.
        self.target_y = 0.
        self.agent_speed = 1.

        self.state = self.reset()
        self.rewards = []

        self.frame = 0

    def step(self, action: int):
        if (action == 0 or action == 2) and not (self.agent_y > 0.3 and self.agent_x+action-1 != 4):
            self.agent_x += action-1

        if action == 1 or action == 3:
            self.agent_y += action-2

        self.state = np.array([self.agent_x, self.agent_y, self.target_x, self.target_y])

        r = -math.dist((self.agent_x, self.agent_y), (self.target_x, self.target_y))/14.15
        self.rewards.append(r)

        done = False
        self.frame += 1
        self.frame %=20
        if self.frame == 0:
            done = True

        return [self.state, r, done, {}]

    def reset(self):
        self.agent_x = float(random.randint(0, 9))/10
        self.agent_y = float(random.randint(0, 9))/10
        self.target_x = float(random.randint(0, 9))/10
        self.target_y = float(random.randint(0, 9))/10
        return np.array([self.agent_x, self.agent_y, self.target_x, self.target_y], dtype=float)

    def render(self):
        pass
