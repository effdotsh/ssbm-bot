import numpy as np
import torch
from torch import nn


class Network(nn.Module):
    def __init__(self, num_inputs, num_choices):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(num_inputs+num_choices, 10),
            # nn.ReLU(),
            nn.Linear(10, 1),
        )
    def forward(self, inputs):
        return self.layers(inputs)

class DiscreteNetwork:
    def __init__(self, num_inputs, num_choices):
        self.num_inputs: int = num_inputs
        self.num_choices: int = num_choices

        self.network = Network(num_inputs=num_inputs, num_choices=num_choices)
    def predict(self, inputs):
        output = 0
        max_reward = -1 * 10e10

        for i in range(self.num_choices):
            network_in = np.zeros(self.num_choices)
            network_in[i] = 1
            network_in = np.append(network_in, inputs)

            network_in = torch.tensor(network_in)
            network_in = network_in.float()
            print(network_in)

            predicted_reward: torch.Tensor = self.network.forward(network_in)
            if(predicted_reward > max_reward):
                max_reward = predicted_reward
                output = i
        return output

if __name__ =='__main__':
    nn = DiscreteNetwork(num_inputs=2, num_choices=3)
    data = np.array([2.0, 4.0])
    print(nn.predict(data))