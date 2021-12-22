# Spartnn - State Predictor And Reward Tree Neural Network
# TODO: Nodes currentky dont know which action they've taken or will take
import random

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from typing import List

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def generate_input_tensor(num_choices, chosen_action: int, inputs) -> torch.Tensor:
    if isinstance(inputs, torch.Tensor):
        inputs = inputs.detach().numpy()

    input_tensor = np.zeros(num_choices)
    input_tensor[chosen_action] = 1.0
    input_tensor = np.append(input_tensor, inputs)

    input_tensor = torch.tensor(input_tensor)
    input_tensor = input_tensor.float()
    return input_tensor


class RewardPredictor(nn.Module):
    def __init__(self, num_inputs, num_choices, layers=None):
        super().__init__()
        if layers is None:
            self.layers = nn.Sequential(
                nn.Linear(num_inputs + num_choices, 10),
                # nn.ReLU(),
                # nn.Linear(10, 10),
                # nn.Linear(10, 10),
                nn.Sigmoid(),
                nn.Linear(10, 1),
            )
        else:
            self.layers = layers

    def forward(self, inputs) -> torch.Tensor:
        return self.layers(inputs)


class StatePredictor(nn.Module):
    def __init__(self, num_inputs, num_choices, layers=None):
        super().__init__()

        if (layers is None):
            self.layers = nn.Sequential(
                nn.Linear(num_inputs + num_choices, 10),
                # nn.ReLU(),
                # nn.Linear(10, 10),
                nn.Linear(10, 10),
                nn.Sigmoid(),
                nn.Linear(10, 10),
                nn.Sigmoid(),
                nn.Linear(10, 10),
                nn.Sigmoid(),
                nn.Linear(10, num_inputs),
            )
        else:
            self.layers = layers

    def forward(self, inputs) -> torch.Tensor:
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
        max_val = -1 * 10e10

        if self.max_depth == self.depth:
            for i in range(self.num_choices):
                generated_inputs = generate_input_tensor(num_choices=self.num_choices, chosen_action=i,
                                                         inputs=self.inputs)
                cval = self.reward_predictor.forward(generated_inputs).item()
                if cval > max_val:
                    max_val = cval
            return max_val

        for i, c in enumerate(self.children):
            generated_inputs = generate_input_tensor(num_choices=self.num_choices, chosen_action=i, inputs=self.inputs)

            cval = c.compute_value() * self.discount_factor + self.reward_predictor.forward(
                inputs=generated_inputs).item()
            if cval > max_val:
                max_val = cval

        return max_val

    def get_action(self) -> int:
        action = 0
        max_reward = -1 * 10e10

        if self.depth == self.max_depth:
            for i in range(self.num_choices):
                generated_inputs = generate_input_tensor(num_choices=self.num_choices, chosen_action=i,
                                                         inputs=self.inputs)
                cval = self.reward_predictor.forward(generated_inputs).item()
                if cval > max_reward:
                    max_reward = cval
                    action = i

            return action
        for i, child in enumerate(self.children):
            generated_inputs = generate_input_tensor(num_choices=self.num_choices, chosen_action=i, inputs=self.inputs)

            cval = child.compute_value() * self.discount_factor + self.reward_predictor(generated_inputs)
            if cval > max_reward:
                action = i
                max_reward = cval

        return action


class Overseer:
    # The goal of this network is to guess the reward returned by an acrtion in ad=vance, and use backprop to update from an observed reward. An action can be chosen by testing each input through the reward predictor and choosing the one with the highest reward
    def __init__(self, num_inputs, num_choices, epsilon_greedy_chance=1, epsilon_greedy_decrease=0.0001,
                 reward_network_learning_rate=0.0003, state_network_learning_rate=0.00003, search_depth=2,
                 discount_factor=0, reward_network_layers=None, state_network_layers=None):
        self.search_depth = search_depth
        self.discount_factor = discount_factor
        self.num_inputs: int = num_inputs
        self.num_choices: int = num_choices

        self.reward_network = RewardPredictor(num_inputs=num_inputs, num_choices=num_choices,
                                              layers=reward_network_layers)
        self.reward_network_criterion = nn.MSELoss()
        self.reward_network_optimizer = torch.optim.Adam(self.reward_network.parameters(),
                                                         lr=reward_network_learning_rate)
        self.reward_network_loss = []

        self.state_predictor = StatePredictor(num_inputs=num_inputs, num_choices=num_choices,
                                              layers=state_network_layers)
        self.state_network_criterion = nn.MSELoss()
        self.state_network_optimizer = torch.optim.Adam(self.reward_network.parameters(),
                                                        lr=state_network_learning_rate)
        self.state_network_loss = []

        self.epsilon_greedy_chance = epsilon_greedy_chance
        self.epsilon_greedy_decrease = epsilon_greedy_decrease

        self.rewards = []
        self.frame = 0

    def predict(self, inputs):
        # output = 0
        # max_reward = -1 * 10e10
        #
        # for i in range(self.num_choices):
        #     network_in: torch.Tensor = generate_input_tensor(num_choices=self.num_choices, chosen_action=i,
        #                                                      inputs=inputs)
        #
        #     predicted_reward_tensor: torch.Tensor = self.reward_network.forward(network_in)
        #     predicted_reward: float = predicted_reward_tensor.item()
        #     if predicted_reward > max_reward:
        #         max_reward = predicted_reward
        #         output = i

        base_node = Node(inputs=inputs, state_predictor=self.state_predictor,
                         reward_predictor=self.reward_network, discount_factor=self.discount_factor,
                         num_choices=self.num_choices, max_depth=self.search_depth)
        output = base_node.get_action()

        # Implement epsilon greedy policy
        if random.random() < self.epsilon_greedy_chance:
            output = random.randint(0, self.num_choices - 1)
        if self.epsilon_greedy_chance > 0:
            self.epsilon_greedy_chance -= self.epsilon_greedy_decrease
            if self.epsilon_greedy_chance < 0:
                self.epsilon_greedy_chance = 0

        return output

    def learn_reward(self, chosen_action, inputs, observed_reward, ):
        network_in = generate_input_tensor(num_choices=self.num_choices, chosen_action=chosen_action, inputs=inputs)
        predicted_reward_tensor = self.reward_network.forward(network_in)

        loss = self.reward_network_criterion(predicted_reward_tensor, torch.tensor([observed_reward]))
        self.reward_network_optimizer.zero_grad()
        loss.backward()
        self.reward_network_optimizer.step()
        self.reward_network_loss.append(loss.item())
        self.frame += 1
        self.rewards.append(observed_reward)

    def learn_state(self, chosen_action, old_state: np.ndarray, new_state: np.ndarray):
        # print(type(new_state))
        # if not isinstance(old_state, np.ndarray):
        #     print('ewfoibweo')
        #     old_state = np.array(old_state)
        # if not isinstance(new_state, np.ndarray):
        #     new_state = np.array(new_state)

        old_state = old_state.astype(np.float32)
        new_state = new_state.astype(np.float32)

        network_in = generate_input_tensor(num_choices=self.num_choices, chosen_action=chosen_action, inputs=old_state)
        predicted_state_tensor = self.state_predictor.forward(network_in)

        loss = self.state_network_criterion(predicted_state_tensor, torch.tensor(new_state))
        self.state_network_optimizer.zero_grad()
        loss.backward()
        self.state_network_optimizer.step()
        self.state_network_loss.append(loss.item())

    def log(self, history: int):
        state_network_avg_loss = np.mean(self.state_network_loss[-history:])
        reward_network_avg_loss = np.mean(self.reward_network_loss[-history:])
        avg_reward = np.mean(self.rewards[-history:])

        print('#######################')
        print(f'Frame: {self.frame}')
        print(f'State Network Loss: {state_network_avg_loss}')
        print(f'Reward Network Loss: {reward_network_avg_loss}')
        print(f'Average Reward: {avg_reward}')
        print(f'Epsilon Greedy Chance: {self.epsilon_greedy_chance}')
        print('#######################')
