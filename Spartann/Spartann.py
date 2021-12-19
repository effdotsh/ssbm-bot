# Spartnn - State Predictor And Reward Tree Neural Network

import random

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from typing import List


def generate_input_tensor(num_choices, chosen_action: int, inputs):
    input_tensor = np.zeros(num_choices)
    input_tensor[chosen_action] = 1.0
    input_tensor = np.append(input_tensor, inputs)

    input_tensor = torch.tensor(input_tensor)
    input_tensor = input_tensor.float()
    return input_tensor


class RewardPredictor(nn.Module):
    def __init__(self, num_inputs, num_choices):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(num_inputs + num_choices, 10),
            # nn.ReLU(),
            # nn.Linear(10, 10),
            nn.Sigmoid(),
            nn.Linear(10, 10),
            nn.Sigmoid(),
            nn.Linear(10, 1),
        )

    def forward(self, inputs):
        return self.layers(inputs)


class StatePredictor(nn.Module):
    def __init__(self, num_inputs, num_choices):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(num_inputs + num_choices, 10),
            # nn.ReLU(),
            # nn.Linear(10, 10),
            nn.Sigmoid(),
            nn.Linear(10, 10),
            nn.Sigmoid(),
            nn.Linear(10, num_inputs),
        )

    def forward(self, inputs):
        return self.layers(inputs)


class Node:
    def __init__(self, inputs, state_predictor, reward_predictor, discount_factor, num_choices, max_depth, depth=0):
        self.num_choices = num_choices
        self.max_depth = max_depth
        self.inputs: np.ndarray = inputs
        self.children: List[Node] = []

        self.state_predictor: StatePredictor = state_predictor
        self.reward_predictor: RewardPredictor = reward_predictor

        self.depth: int = depth
        self.discount_factor: float = discount_factor

        # Generate children
        if depth < max_depth:
            for i in range(num_choices):
                generated_inputs = generate_input_tensor(num_choices=self.num_choices, chosen_action=i,
                                                         inputs=self.inputs)
                predicted_state = self.state_predictor.forward(generated_inputs)
                child: Node = Node(inputs=predicted_state, state_predictor=self.state_predictor,
                                   reward_predictor=self.reward_predictor, discount_factor=self.discount_factor,
                                   num_choices=self.num_choices, max_depth=self.max_depth, depth=self.depth + 1)
                self.children.append(child)

    def compute_value(self) -> int:
        if self.maxdepth == self.depth:
            return self.reward_predictor.forward(inputs=self.inputs)

        max_val = -1 * 10e10
        for c in self.children:
            cval = c.compute_value()
            if cval > max_val:
                max_val = cval

        return self.reward_predictor.forward(inputs=self.inputs) + max_val * self.discount_factor

    def get_action(self) -> int:
        action = 0
        max_reward = -1 * 10e10
        for i, child in enumerate(self.children):
            cval = child.compute_value()
            if cval > max_reward:
                action = i
                max_reward = cval
        return action


class Overseer:
    # TODO: also have a network to predict the future state from the state + action, and max reward from that
    # The goal of this network is to guess the reward returned by an acrtion in ad=vance, and use backprop to update from an observed reward. An action can be chosen by testing each input through the reward predictor and choosing the one with the highest reward
    def __init__(self, num_inputs, num_choices, epsilon_greedy_chance=1, epsilon_greedy_decrease=0.0001,
                 learning_rate=0.0003, num_steps_ahead=0):
        self.num_inputs: int = num_inputs
        self.num_choices: int = num_choices

        self.network = RewardPredictor(num_inputs=num_inputs, num_choices=num_choices)

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        self.loss = []

        self.epsilon_greedy_chance = epsilon_greedy_chance
        self.epsilon_greedy_decrease = epsilon_greedy_decrease

    def predict(self, inputs):
        output = 0
        max_reward = -1 * 10e10

        for i in range(self.num_choices):
            network_in: torch.Tensor = generate_input_tensor(num_choices=self.num_choices, chosen_action=i,
                                                             inputs=inputs)

            predicted_reward_tensor: torch.Tensor = self.network.forward(network_in)
            predicted_reward: float = predicted_reward_tensor.item()
            if predicted_reward > max_reward:
                max_reward = predicted_reward
                output = i

        # Implement epsilon greedy policy
        if random.random() < self.epsilon_greedy_chance:
            output = random.randint(0, self.num_choices - 1)
        if self.epsilon_greedy_chance > 0:
            self.epsilon_greedy_chance -= self.epsilon_greedy_decrease
            if self.epsilon_greedy_chance < 0:
                self.epsilon_greedy_chance = 0

        return output

    def learn(self, chosen_action, inputs, observed_reward):
        network_in = generate_input_tensor(num_choices=self.num_choices, chosen_action=chosen_action, inputs=inputs)
        predicted_reward_tensor = self.network.forward(network_in)

        loss = self.criterion(predicted_reward_tensor, torch.tensor([observed_reward]))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.loss.append(loss.item())
