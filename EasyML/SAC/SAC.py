import gym
import numpy as np
from collections import deque
import torch
import argparse
from .buffer import ReplayBuffer
import glob
from .utils import save, collect_random
import random
from .agent import SACAgent

default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SAC:
    def __init__(self, num_inputs, num_actions, batch_size=256, device=default_device, max_replay_size=10_000,
                 min_replay_size=512, discount_factor=0.99, tau=1e-2, learning_rate=5e-4):
        self.discount_factor = discount_factor
        assert min_replay_size >= batch_size

        self.num_inputs = num_inputs
        self.num_actions = num_actions

        self.agent = SACAgent(state_size=num_inputs,
                              action_size=num_actions,
                              device=device, discount_factor=discount_factor, tau=tau, learning_rate=learning_rate)

        self.buffer = ReplayBuffer(buffer_size=max_replay_size, batch_size=batch_size, device=device)

        self.min_replay_size = min_replay_size
        self.steps = 0

        self.stats = []

    def predict(self, state):
        action = self.agent.get_action(state)
        if len(self.buffer.memory) < self.min_replay_size:
            action = random.randint(0, self.num_actions - 1)
            print('reeee')
        return action

    def update_replay_memory(self, state, action, reward, next_state, done):
        self.steps += 1
        self.buffer.add(state, action, reward, next_state, done)

    def train(self):
        if len(self.buffer.memory) < self.min_replay_size:
            return False
        # policy_loss, alpha_loss, bellmann_error1, bellmann_error2, current_alpha\
        self.stats = self.agent.learn(self.steps, self.buffer.sample())
        return True

    def save(self, name="model", wandb=None, ep=0):
        save(name, model=self.agent.actor_local, wandb=wandb, ep=ep)