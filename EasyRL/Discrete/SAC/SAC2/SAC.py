# Implementation based off of AI4Finance-Foundation ElegantRL

import os
import time
import torch
import numpy as np
import multiprocessing as mp

from .ReplayBuffer import ReplayBuffer, ReplayBufferList

from .SACAgent import AgentSAC

default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SAC:
    def __init__(self, obs_dim: int, action_dim: int, net_width: int = 256, discount_factor=0.995, learning_rate=1e-4,
                 max_buffer_size=1_000_000, min_buffer_size=1000, batch_size=256, soft_update_tau=0.005, repeat_times=1):
        self.min_buffer_size = min_buffer_size
        self.repeat_times = repeat_times
        self.soft_update_tau = soft_update_tau
        self.batch_size = batch_size
        self.agent = AgentSAC()
        self.agent.init(state_dim=obs_dim, action_dim=action_dim, net_dim=net_width, gamma=discount_factor,
                        learning_rate=learning_rate)
        self.buffer = ReplayBuffer(gpu_id=0,
                                   max_len=max_buffer_size,
                                   state_dim=obs_dim,
                                   action_dim=1)

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        torch.set_grad_enabled(False)

        self.traj = []
        self.final_obs = None
        self.replay_len = 0


    def predict(self, obs):
        ten_state = torch.as_tensor(obs, dtype=torch.float32)
        ten_action = self.agent.select_actions(ten_state.unsqueeze(0))[0]
        # print(ten_action.numpy())
        action = np.argmax(ten_action.numpy())

        return action

    def learn_experience(self, obs, action, reward, new_obs, done):
        self.buffer.extend_buffer(obs, (reward, action, done))
        self.replay_len += 1
        if self.replay_len > self.min_buffer_size and (self.replay_len - self.min_buffer_size) % self.batch_size == 0:
            self.train(True)

    def train(self, verified=False):
        if not verified:
            return
        torch.set_grad_enabled(True)
        logging_tuple = self.agent.update_net(self.buffer, batch_size=self.batch_size, repeat_times=self.repeat_times,soft_update_tau=self.soft_update_tau)
        torch.set_grad_enabled(False)

    def get_log(self):
        return {}
