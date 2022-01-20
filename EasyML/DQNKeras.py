import random

import keras.callbacks
import tensorflow as tf
import tensorflow.python.keras.callbacks
from tensorflow.keras.layers import Dense, Dropout, Conv2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import time

from collections import deque

import numpy as np

import datetime


class DQNAgent:
    def __init__(self, num_inputs, num_outputs, learning_rate=0.001, min_replay_size=10_000, max_replay_size=50_000,
                 minibatch_size=16, discount_factor=0.9, update_target_every=5, epsilon=1, min_epsilon=0.01,
                 epsilon_decay=0.99):
        # Gets Trained
        self.model = self.create_model(num_inputs=num_inputs, num_outputs=num_outputs, learning_rate=learning_rate)

        # Gets predicted from
        self.target_model = self.create_model(num_inputs=num_inputs, num_outputs=num_outputs,
                                              learning_rate=learning_rate)
        self.target_model.set_weights(self.model.get_weights())

        self.min_replay_size = min_replay_size
        self.replay_memory = deque(maxlen=max_replay_size)

        self.target_update_counter = 0
        self.num_updates = 0

        self.minibatch_size = minibatch_size
        self.discount_factor = discount_factor
        self.update_target_every = update_target_every
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.epsilon = epsilon

        self.first_train = False

    def create_model(self, num_inputs, num_outputs, learning_rate):
        model = Sequential()
        model.add(Dense(num_inputs, input_shape=[num_inputs]))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(128, activation='relu'))
        # model.add(Dense(128, activation='tanh'))
        # model.add(Dense(64, activation='relu'))
        model.add(Dense(num_outputs, activation='linear'))

        model.compile(loss="mse", optimizer=Adam(learning_rate=learning_rate), metrics=['accuracy'])
        # model.build(input_shape=[1 for i in range(num_inputs)])

        return model

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape))[0]

    def predict(self, state, out_eps=False):
        randVal = np.random.random()
        if randVal < self.epsilon:
            # Random action
            action = np.random.randint(0, self.num_outputs)
            if out_eps:
                real_prediction = np.argmax(self.get_qs(state))
                print(f'{real_prediction} -> {action}')

        else:
            # q-table action
            action = np.argmax(self.get_qs(state))


        return action

    def train(self, terminal_state):
        if len(self.replay_memory) < self.min_replay_size:
            return

        self.first_train = True
        self.epsilon *= self.epsilon_decay ** self.minibatch_size
        self.epsilon = max(self.min_epsilon, self.epsilon)
        self.num_updates += self.minibatch_size

        minibatch = random.sample(self.replay_memory, self.minibatch_size)

        current_states = np.array([transition[0] for transition in minibatch])

        current_qs_list = self.model.predict(current_states)

        new_current_states = np.array([transition[3] for transition in minibatch])

        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        y = []

        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + self.discount_factor * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state)
            y.append(current_qs)

        self.model.fit(np.array(X), np.array(y), batch_size=self.minibatch_size, verbose=0, shuffle=False,
                       callbacks=None, use_multiprocessing=True)

        if terminal_state:
            self.target_update_counter += 1

        if self.target_update_counter > self.update_target_every:
            print('REEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE')
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0



