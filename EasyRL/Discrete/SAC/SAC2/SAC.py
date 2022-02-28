# Implementation based off of AI4Finance-Foundation ElegantRL

import os
import time
import torch
import numpy as np
import multiprocessing as mp

from ReplayBuffer import ReplayBuffer, ReplayBufferList

from .SACAgent import AgentSAC_H

default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SAC:
    def __init__(self, obs_dim: int, action_dim: int, net_width: int = 256, discount_factor=0.995, learning_rate=1e-4,
                 max_buffer_len=100_000):
        self.agent = AgentSAC_H()
        self.agent.init(state_dim=obs_dim, action_dim=action_dim, net_dim=net_width, gamma=discount_factor,
                        learning_rate=learning_rate)
        self.buffer = ReplayBuffer(gpu_id=0,
                                   max_len=max_buffer_len,
                                   state_dim=obs_dim,
                                   action_dim=1)

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        torch.set_grad_enabled(False)

        self.traj = []
        self.final_obs

    def predict(self, obs):
        ten_state = torch.as_tensor(obs, dtype=torch.float32)
        ten_action = self.agent.select_actions(ten_state.unsqueeze(0))[0]
        action = ten_action.numpy()
        return action

    def learn_experience(self, obs, action, reward, new_obs, done):
        ten_other = torch.empty(2 + self.action_dim)
        ten_other[0] = reward
        ten_other[1] = done
        ten_other[2:] = torch.Tensor([action])

        ten_state = torch.as_tensor(obs, dtype=torch.float32)
        self.traj.append((ten_state, ten_other))
        self.final_obs = new_obs

    def train(self, verified=False):
        self.agent.states[0] = self.final_obs
        traj_state = torch.stack([item[0] for item in self.traj])
        traj_other = torch.stack([item[1] for item in self.traj])
        traj_list = [
            (traj_state, traj_other),
        ]

        trajectory = self.agent.convert_trajectory(traj_list)  # [traj_env_0, ]


        pass

    def get_log(self):
        pass
