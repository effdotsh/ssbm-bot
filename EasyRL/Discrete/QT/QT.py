import random
from collections import deque

import numpy
import numpy as np

from .table import Table


class QT:
    def __init__(self, obs_dim: int, action_dim: int, discount_factor: float = 0.995, epsilon=1, epsilon_decay=0.9995,
                 obs_divider=5, learning_rate=3e-4, min_tests = 2):
        self.min_tests = min_tests

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.obs_divider = obs_divider
        states_shape = [self.obs_divider] * obs_dim
        self.action_dim = action_dim
        self.table = Table(states_shape=states_shape, num_actions=action_dim, learning_rate=learning_rate,
                           discount_factor=discount_factor)

        self.losses = deque(maxlen=1000)

    def get_inp(self, obs):
        return [int(i // (1 / self.obs_divider)) for i in obs]

    def predict(self, obs):
        obs = self.get_inp(obs)
        row, q_val = self.table.get_action(obs)

        if random.random() < self.epsilon or row[-1] < self.min_tests:
            return random.randint(0, self.action_dim - 1)
        else:
            return np.argmax(row[:-1])

    def learn_experience(self, obs, action, reward, new_obs, done):
        obs = self.get_inp(obs)
        new_obs = self.get_inp(new_obs)

        row, q_val = self.table.get_action(obs)
        self.losses.append(abs(q_val-reward))

        self.epsilon *= self.epsilon_decay
        self.table.learn_experience(obs, action, reward, new_obs, done)

    def train(self):
        return

    def get_log(self):
        obj = {
            "epsilon":self.epsilon,
            "learning rate": self.table.learning_rate,
            "table size": len(self.table.table),
            "losses":np.mean(self.losses)
        }
        return obj