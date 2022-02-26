# Implementation based on XinJingHao/PPO-Discrete-Pytorch

import torch
import numpy as np


class PPO:
    def __init__(self, obs_dim: int, action_dim: int, discount_factor: float = 0.995, lambd=0.95,
                 clip_rate: float = 0.2, K_epochs: int = 10, net_width: int = 128, learning_rate: float = 1e-4,
                 l2_reg: float = 0, batch_size: int = 64, entropy_coef: float = 0, entropy_coef_decay: float = 0.99,
                 adv_normalization: bool = False):
        self.adv_normalization = adv_normalization
        self.entropy_coef_decay = entropy_coef_decay
        self.entropy_coef = entropy_coef
        self.batch_size = batch_size
        self.l2_reg = l2_reg
        self.learning_rate = learning_rate
        self.net_width = net_width
        self.K_epochs = K_epochs
        self.clip_rate = clip_rate
        self.lambd = lambd
        self.discount_factor = discount_factor
        self.action_dim = action_dim
        self.obs_dim = obs_dim

