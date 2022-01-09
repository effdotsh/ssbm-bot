import torch
import numpy as np
from torch import nn

from typing import List


def generate_input_tensor(num_choices, chosen_action: int, inputs) -> torch.Tensor:
    if isinstance(inputs, torch.Tensor):
        inputs = inputs.detach().numpy()

    input_tensor = np.zeros(num_choices).astype(float)
    input_tensor[chosen_action] = 1.0
    input_tensor = np.append(input_tensor, inputs).astype(float)

    input_tensor = torch.tensor(input_tensor)
    input_tensor = input_tensor.float()
    return input_tensor


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


class RewardPredictor(nn.Module):
    def __init__(self, num_inputs, num_choices, layers=None):
        super().__init__()
        if layers is None:
            self.layers = nn.Sequential(
                nn.Linear(num_inputs + num_choices, 256),
                nn.Tanh(),
                nn.Linear(256, 256),
                nn.Tanh(),
                nn.Linear(256, 256),
                nn.Tanh(),
                nn.Linear(256, 1),
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
                nn.Linear(num_inputs + num_choices, 256),
                nn.Linear(256, 256),
                nn.Tanh(),
                nn.Linear(256, 256),
                nn.Tanh(),
                nn.Linear(256, num_inputs),
            )
        else:
            self.layers = layers

    def forward(self, inputs) -> torch.Tensor:
        return self.layers(inputs)
