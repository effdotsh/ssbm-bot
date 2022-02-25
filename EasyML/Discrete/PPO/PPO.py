# Implementation based on vwxyzjn/cleanrl
import argparse
import os
import random
import time
from distutils.util import strtobool

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int):
        super(Agent, self).__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, action_dim), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()

        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PPO:
    def __init__(self, obs_dim: int, action_dim: int, device=default_device, learning_rate: float = 2.5e-4,
                 num_steps: int = 128, use_gae: bool = True, discount_factor: float = 0.995, gae_lambda: float = 0.95,
                 update_epochs: int = 4, num_minibatches: int = 4, clip_coef: float = 0.2, norm_adv: bool = True,
                 clip_vloss: bool = True, ent_coef: float = 0.01, vf_coef: float = 0.5, max_grad_norm: float = 0.5,
                 target_kl=None):
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.clip_vloss = clip_vloss
        self.norm_adv = norm_adv
        self.clip_coef = clip_coef
        self.num_minibatches = num_minibatches
        self.update_epochs = update_epochs

        self.gae_lambda = gae_lambda
        self.discount_factor = discount_factor
        self.use_gae = use_gae
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_steps = num_steps
        self.learning_rate = learning_rate
        self.agent = Agent(obs_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=self.learning_rate, eps=1e-5)
        self.device = device

        # ALGO Logic: Storage setup
        self.obs = torch.zeros((self.num_steps, self.obs_dim)).to(self.device)
        self.actions = torch.zeros((self.num_steps, self.action_dim)).to(device)
        self.logprobs = torch.zeros((self.num_steps, 1)).to(device)
        self.rewards = torch.zeros((self.num_steps, 1)).to(device)
        self.dones = torch.zeros((self.num_steps, 1)).to(device)
        self.values = torch.zeros((self.num_steps, 1)).to(device)

        # TRY NOT TO MODIFY: start the game
        self.global_step = 0
        self.start_time = time.time()
        self.next_obs = torch.Tensor(self.obs_dim)
        self.next_done = torch.zeros(1).to(device)
        # self.num_updates = args.total_timesteps // args.batch_size
        self.global_step = 0
        self.step = 0

        self.batch_size = num_steps
        self.minibatch_size = self.num_steps // self.num_minibatches

        self.log_obj = {"charts/learning_rate": self.optimizer.param_groups[0]["lr"],
                        "losses/value_loss": None,
                        "losses/policy_loss": None,
                        "losses/entropy": None,
                        "losses/old_approx_kl": None,
                        "losses/approx_kl": None,
                        "losses/clipfrac": None,
                        "losses/explained_variance": None
                        }

    def predict(self, obs):
        obs = torch.Tensor(obs)
        action, logprob, _, value = self.agent.get_action_and_value(obs)
        return int(action)

    def learn_experience(self, obs, action, reward, new_obs, done):
        obs = torch.Tensor(obs)
        action = None  # Action is recalculated to get the logprob
        self.obs[self.step] = obs

        with torch.no_grad():
            action, logprob, _, value = self.agent.get_action_and_value(obs)
            self.values[self.step] = value.flatten()
        self.logprobs[self.step] = logprob

        self.actions[self.step] = action
        self.rewards[self.step] = reward
        self.dones[self.step] = torch.Tensor([1 if done else 0])
        self.step += 1
        self.global_step += 1
        if self.step >= self.num_steps:
            self.train()

    def train(self):
        if self.step < self.num_steps:
            return
        self.step = 0
        v_loss = None
        pg_loss = None
        entropy_loss = None
        # bootstrap value if not done
        with torch.no_grad():
            next_value = self.agent.get_value(self.obs[-1]).reshape(1, -1)
            if self.use_gae:
                advantages = torch.zeros_like(self.rewards).to(self.device)
                lastgaelam = 0
                for t in reversed(range(self.num_steps)):
                    if t == self.num_steps - 1:
                        nextnonterminal = 1.0 - self.dones[-1]
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - self.dones[t + 1]
                        nextvalues = self.values[t + 1]
                    delta = self.rewards[t] + self.discount_factor * nextvalues * nextnonterminal - self.values[t]
                    advantages[
                        t] = lastgaelam = delta + self.discount_factor * self.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + self.values
            else:
                returns = torch.zeros_like(self.rewards).to(self.device)
                for t in reversed(range(self.num_steps)):
                    if t == self.num_steps - 1:
                        nextnonterminal = 1.0 - self.dones[-1]
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - self.dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = self.rewards[t] + self.discount_factor * nextnonterminal * next_return
                advantages = returns - self.values

        # flatten the batch
        b_obs = self.obs.reshape((-1,) + (self.obs_dim,))
        b_logprobs = self.logprobs.reshape(-1)
        b_actions = self.actions.reshape((-1,) + ())
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = self.values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(self.batch_size)
        clipfracs = []
        for epoch in range(self.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, self.batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(b_obs[mb_inds],
                                                                                   b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if self.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if self.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -self.clip_coef,
                        self.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                self.optimizer.step()

            if self.target_kl is not None:
                if approx_kl > self.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        self.log_obj = {"charts/learning_rate": self.optimizer.param_groups[0]["lr"],
                        "losses/value_loss": v_loss.item(),
                        "losses/policy_loss": pg_loss.item(),
                        "losses/entropy": entropy_loss.item(),
                        "losses/old_approx_kl": old_approx_kl.item(),
                        "losses/approx_kl": approx_kl.item(),
                        "losses/clipfrac": np.mean(clipfracs),
                        "losses/explained_variance": explained_var
                        }

    def get_log(self):
        return self.log_obj
