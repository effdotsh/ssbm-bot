# Implementation based off of AI4Finance-Foundation ElegantRL

import os
import time
import torch
import numpy as np
import multiprocessing as mp

from ReplayBuffer import ReplayBuffer, ReplayBufferList

from .SACAgent import AgentSAC
default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SAC:
    def __init__(self, obs_dim: int, action_dim: int):
        self.agent = init_agent(args, gpu_id, env)
        self.buffer = init_buffer(args, gpu_id)
        self.evaluator = init_evaluator(args, gpu_id)
    def predict(self, obs):
        pass

    def learn_experience(self, obs, action, reward, new_obs, done):
        pass

    def train(self, verified=False):
        pass

    def get_log(self):
        pass