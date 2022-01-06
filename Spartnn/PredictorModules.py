import torch
import numpy as np
from torch import nn
def generate_input_tensor(num_choices, chosen_action: int, inputs) -> torch.Tensor:
    if isinstance(inputs, torch.Tensor):
        inputs = inputs.detach().numpy()

    input_tensor = np.zeros(num_choices).astype(float)
    input_tensor[chosen_action] = 1.0
    input_tensor = np.append(input_tensor, inputs).astype(float)

    input_tensor = torch.tensor(input_tensor)
    input_tensor = input_tensor.float()
    return input_tensor



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