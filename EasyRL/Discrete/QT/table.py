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

        table_shape = list(copy.deepcopy(self.states_shape))
        table_shape.append(self.num_actions)
        table_shape=tuple(table_shape)


        print(table_shape)
        # self.table = np.full(table_shape, 0., dtype=object)

        self.table: dict = {}

    def get_row(self, state):
        s = str(state)
        if s not in self.table:
            self.table[s] = np.zeros(self.num_actions+1)
            print(s)
        return self.table[s]

    def get_action(self, state):

        row = self.get_row(state)

        action = np.argmax(row[:-1])
        return row, row[action]

    def learn_experience(self, obs, action, reward, new_obs, done):

        row = self.get_row(obs)
        row[action] = (1 - self.learning_rate) * row[action] + self.learning_rate * reward
        row[-1] += 1
        if not done:
            row[action] += self.learning_rate * self.discount_factor * self.get_action(new_obs)[1]
