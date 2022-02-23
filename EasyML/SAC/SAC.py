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
    def __init__(self, obs_dim, action_dim, batch_size=256, device=default_device, max_replay_size=10_000,
                 min_replay_size=512, discount_factor=0.99, tau=1e-2, learning_rate=5e-4):
        self.discount_factor = discount_factor
        assert min_replay_size >= batch_size

        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.agent = SACAgent(state_size=obs_dim,
                              action_size=action_dim,
                              device=device, discount_factor=discount_factor, tau=tau, learning_rate=learning_rate)

        self.buffer = ReplayBuffer(buffer_size=max_replay_size, batch_size=batch_size, device=device)

        self.min_replay_size = min_replay_size
        self.steps = 0

        self.stats = []

    def predict(self, obs):
        action = self.agent.get_action(obs)
        if len(self.buffer.memory) < self.min_replay_size:
            action = random.randint(0, self.action_dim - 1)
        return action

    def learn_expirience(self, obs, action, reward, new_obs, done):
        self.steps += 1
        self.buffer.add(obs, action, reward, new_obs, done)

    def train(self):
        if len(self.buffer.memory) < self.min_replay_size:
            return False
        # policy_loss, alpha_loss, bellmann_error1, bellmann_error2, current_alpha
        self.stats = self.agent.learn(self.steps, self.buffer.sample())
        return True

    def get_log(self):
        obj = {
            "buffer_size": len(self.buffer.memory),
            "policy_loss": None,
            "alpha_loss": None,
            "bellmann_error1": None,
            "bellmann_error2": None,
            "current_alpha": None
        }
        if len(self.stats) != 0:
            obj['policy_loss'] = self.stats[0]
            obj['alpha_loss'] = self.stats[1]
            obj['bellmann_error1'] = self.stats[2]
            obj['bellmann_error2'] = self.stats[3]
            obj['current_alpha'] = self.stats[4]

        return obj

    def save(self, name="model", wandb=None, ep=0):
        save(name, model=self.agent.actor_local, wandb=wandb, ep=ep)
