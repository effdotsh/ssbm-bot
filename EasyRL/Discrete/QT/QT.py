import random

import numpy
from table import Table


class QT:
    def __init__(self, obs_dim: int, action_dim: int, discount_factor: float = 0.995, epsilon=1, epsilon_decay=0.995,
                 obs_divider=200, learning_rate=3e-4):
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.obs_divider = obs_divider
        states_shape = [self.obs_divider] * obs_dim
        self.action_dim = action_dim
        self.table = Table(states_shape=states_shape, num_actions=action_dim, learning_rate=learning_rate,
                           discount_factor=discount_factor)

    def predict(self, obs):
        inp = [int(i // (1 / self.obs_divider)) for i in obs]
        action, _q_val = self.table.get_action(inp)

        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            return action

    def learn_experience(self, obs, action, reward, new_obs, done):
        self.epsilon *= self.epsilon_decay
        self.table.learn_experience(obs, action, reward, new_obs, done)

    def train(self):
        return