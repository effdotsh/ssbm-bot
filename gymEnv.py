import gym
import random
from gym import spaces
import math
import numpy as np

frames_per_step = 100

def rand_value():
    return random.random()-0.5 * 1000
def getObs():
    return np.array([rand_value(), rand_value(), rand_value()])
class CharacterEnv(gym.Env):
    def __init__(self):
        super(CharacterEnv, self).__init__()

        self.obs = getObs()
        self.r = 0
        self.frame = 0

        self.observation_space = spaces.Box(shape=np.array([3]), dtype=np.float, low=-500, high=500)
        self.action_space = spaces.Box(shape=np.array([1]), dtype=np.float, low = -1, high=1)

        self.rewards = []
    def step(self, action: float):
        action *= 1500
        self.frame += 1
        self.frame %= frames_per_step

        if(self.frame == 0):
            print(self.r)
            self.rewards.append(self.r)
            self.r = 0

        actual = self.obs[0] + self.obs[1] + self.obs[2]
        r = -math.fabs(action - actual)
        self.r += r
        self.obs = getObs()
        return [self.obs, r if self.frame == frames_per_step-1 else 0, self.frame == 0, {}]



    def reset(self):
        return self.obs

