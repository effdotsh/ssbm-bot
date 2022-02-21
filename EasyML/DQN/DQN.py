import copy
import random

import torch
from torch import nn

from torchsample.modules import ModuleTrainer
import time

from collections import deque

import numpy as np

import datetime

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'
print(device)


class DQNetwork(nn.Module):
    def __init__(self, num_inputs: int, num_outputs: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(num_inputs, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, num_outputs)
        )

    def forward(self, inputs):
        return self.layers(inputs)


class DQN:
    def __init__(self, num_inputs, num_actions, learning_rate=0.001, min_replay_size=10_000, max_replay_size=50_000,
                 batch_size=16, discount_factor=0.9, update_target_every=5, epsilon=1, min_epsilon=0.01,
                 epsilon_decay=0.99):
        # Gets Trained
        self.model = self.create_model(num_inputs=num_inputs, num_actions=num_actions)
        # Gets predicted from
        # TODO: use the pytorch equivilent of keras' get_weights and set_weights. This implementation is kinda sloppy
        self.target_model = copy.deepcopy(self.model)

        self.trainer = ModuleTrainer(self.model)

        optim = torch.optim.Adam(self.model.parameters(),
                                 lr=learning_rate)
        self.trainer.compile(loss='mse_loss',
                             optimizer=optim)
        self.min_replay_size = min_replay_size
        self.replay_memory = deque(maxlen=max_replay_size)

        self.target_update_counter = 0
        self.num_updates = 0

        self.minibatch_size = batch_size
        self.discount_factor = discount_factor
        self.update_target_every = update_target_every
        self.num_inputs = num_inputs
        self.num_outputs = num_actions

        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.epsilon = epsilon

        self.first_train = False

    def create_model(self, num_inputs: int, num_actions: int):
        model = DQNetwork(num_inputs, num_actions)

        return model.to(device)

    def learn_expirience(self, state, action, reward, next_state, done):
        self.replay_memory.append((state, action, reward, next_state, done))

    def get_qs(self, state):
        state_tensor = torch.Tensor(state).to(device)
        return self.model.forward(state_tensor)

    def predict(self, state):
        randVal = random.random()
        if randVal < self.epsilon:
            # Random action
            action = np.random.randint(0, self.num_outputs)

        else:
            # q-table action
            qs = self.get_qs(state).detach().cpu().numpy()
            # print(qs)
            action = np.argmax(qs)


        return action

    def train(self):
        terminal_state = True
        if len(self.replay_memory) < self.min_replay_size:
            return

        self.first_train = True
        self.epsilon *= self.epsilon_decay ** self.minibatch_size
        self.epsilon = max(self.min_epsilon, self.epsilon)
        self.num_updates += self.minibatch_size

        minibatch = random.sample(self.replay_memory, self.minibatch_size)

        # for transition in minibatch:
        #     print(torch.Tensor(transition[0]))

        current_states = torch.Tensor(np.array([np.array(transition[0]) for transition in minibatch])).to(device)

        # print(current_states)
        current_qs_list = self.model.forward(current_states)

        new_current_states = np.array([transition[3] for transition in minibatch])

        future_qs_list = self.target_model.forward(torch.Tensor(new_current_states).to(device))

        X = []
        y = []

        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index].detach().cpu().numpy())
                new_q = reward + self.discount_factor * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state)
            y.append(current_qs.detach().cpu().numpy())

        X = np.array(X)
        y = np.array(y)
        # self.model.fit(np.array(X), np.array(y), batch_size=self.minibatch_size, verbose=0, shuffle=False,
        #                callbacks=None, use_multiprocessing=True)
        # print(torch.Tensor(y))
        self.trainer.fit(torch.Tensor(X).to(device), torch.Tensor(y).to(device), batch_size=self.minibatch_size,
                         verbose=0)
        # print(self.trainer.history.batch_metrics)
        if terminal_state:
            self.target_update_counter += 1

        if self.target_update_counter > self.update_target_every:
            # TODO: Same as above
            self.target_model = copy.deepcopy(self.model)
            self.target_update_counter = 0

    def get_log(self):
        return {}