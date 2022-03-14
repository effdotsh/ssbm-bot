import numpy as np
import copy
import random


class Table:
    def __init__(self, states_shape, num_actions, learning_rate=3e-4, epsilon=0.99, discount_factor=0.95):
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.num_actions = num_actions
        self.states_shape = states_shape

        table_shape = np.array([self.states_shape, self.num_actions]).flatten()
        self.table = np.full(table_shape, 0., dtype=np.float)

    def get_action(self, state):
        row = self.table[state]

        action = np.argmax(row)
        return action, row[action]

    def learn_experience(self, obs, action, reward, new_obs, done):
        self.table[obs][action] = (1 - self.learning_rate) * self.table[obs][action] + self.learning_rate * reward

        if not done:
            self.table[obs][action] += self.learning_rate * self.discount_factor * self.get_action(new_obs)[1]
