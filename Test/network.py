import numpy as np
import torch
from torch import nn
from tqdm import tqdm

class Network(nn.Module):
    def __init__(self, num_inputs, num_choices):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(num_inputs+num_choices, 2),
            nn.ReLU(),
            nn.Linear(2, 1),
        )
    def forward(self, inputs):
        return self.layers(inputs)

class Overseer:
    # The goal of this network is to guess the reward returned by an acrtion in ad=vance, and use backprop to update from an observed reward. An action can be chosen by testing each input through the reward predictor and choosing the one with the highest reward
    def __init__(self, num_inputs, num_choices):
        self.num_inputs: int = num_inputs
        self.num_choices: int = num_choices

        self.network = Network(num_inputs=num_inputs, num_choices=num_choices)

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=0.0005)
    def predict(self, inputs):
        output = 0
        max_reward = -1 * 10e10

        for i in range(self.num_choices):
            network_in = self._generate_input_tensor(chosen_action=i, inputs=inputs)

            predicted_reward: torch.Tensor = self.network.forward(network_in)
            if(predicted_reward > max_reward):
                max_reward = predicted_reward
                output = i
        return output

    def learn(self, chosen_action, inputs, observed_reward):
        network_in= self._generate_input_tensor(chosen_action=chosen_action, inputs=inputs)
        predicted_reward = self.network.forward(network_in)

        loss = self.criterion(predicted_reward, torch.tensor([observed_reward]))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()



    def _generate_input_tensor(self, chosen_action: int, inputs):
        input_tensor = np.zeros(self.num_choices)
        input_tensor[chosen_action] = 1
        input_tensor = np.append(input_tensor, inputs)

        input_tensor = torch.tensor(input_tensor)
        input_tensor = input_tensor.float()
        return input_tensor
##########################################################################

from matplotlib import pyplot as plt
def graph(points):
    avg = 200
    graph = []
    for i in range(avg, len(points)):
        graph.append(np.mean(points[i - avg:i]))
    print(graph)
    plt.plot(graph)
    plt.show()
if __name__ =='__main__':
    nn = Overseer(num_inputs=2, num_choices=2)
    obs = np.random.rand(2)

    rewards = []
    for i in tqdm(range(100_000)):
        action = nn.predict(obs)
        r=0.
        if(action == 0 and obs[0] > obs[1] or action==1 and obs[0]<obs[1]):
            r = 1.
        rewards.append(r)
        nn.learn(chosen_action=action, inputs=obs, observed_reward=r)

        obs = np.random.rand(2)

    graph(rewards)