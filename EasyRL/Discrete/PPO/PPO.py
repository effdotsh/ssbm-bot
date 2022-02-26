# Implementation based on XinJingHao/PPO-Discrete-Pytorch
import colorama
import torch
import numpy as np

from .agent import PPO_Agent

default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PPO:
    def __init__(self, obs_dim: int, action_dim: int, discount_factor: float = 0.995, lambd=0.95,
                 clip_rate: float = 0.2, K_epochs: int = 10, net_width: int = 128, learning_rate: float = 1e-4,
                 l2_reg: float = 0, batch_size: int = 64, entropy_coef: float = 0, entropy_coef_decay: float = 0.99,
                 adv_normalization: bool = False, device: str = default_device, T_horizon:int=2048):
        self.T_horizon = T_horizon
        self.device = device
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

        self.model = PPO_Agent(state_dim=obs_dim, action_dim=action_dim, gamma=discount_factor, lambd=lambd,
                               net_width=net_width, lr=learning_rate, clip_rate=clip_rate, K_epochs=K_epochs,
                               batch_size=batch_size, l2_reg=l2_reg, entropy_coef=entropy_coef,
                               adv_normalization=adv_normalization, entropy_coef_decay=entropy_coef_decay,
                               device=default_device)
        self.traj_length = 0

        self.log_obg = {
            "a_loss":None,
            "c_loss":None,
            "entropy":None
        }
    def predict(self, obs, deterministic=False):
        if deterministic:
            action, pi_a = self.model.evaluate(torch.from_numpy(obs).float().to(self.device))
        else:
            action, pi_a = self.model.select_action(torch.from_numpy(obs).float().to(self.device))
        return action, pi_a

    def learn_experience(self, obs, action, reward, new_obs, done, dead:bool=None, pi_a = 1.0):
        self.traj_length += 1.0
        if dead is None:
            dead=done
        # self.model.put_data((obs, action, reward, new_obs, 1.0, done, False))
        self.model.put_data((obs, action, reward, new_obs, pi_a, done, dead))

        if self.traj_length >= self.T_horizon:
            print(f'{colorama.Fore.BLUE}Updating Model...{colorama.Fore.RESET}')
            self.train(True)

    def train(self, verified=False):
        if not verified:
            return
        self.traj_length =0
        a_loss, c_loss, entropy = self.model.train()
        self.log_obg = {
            "a_loss":a_loss,
            "c_loss":c_loss,
            "entropy":entropy
        }

    def get_log(self):
        return self.log_obg