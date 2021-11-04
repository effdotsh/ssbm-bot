import os
import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np


import random
np.array([])
class CharacterAI(nn.Module):
    def __init__(self, architecture):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(4, 2),
            # nn.ReLU(),
            nn.Linear(2, 1),
        )
    def forward(self, inputs):
        return self.layers(inputs)

def rand_value():
    return random.random() - 0.5 * 100
if __name__== '__main__':
    torch.manual_seed(1)
    ai = CharacterAI([3, 1])
    x = torch.tensor([[1.0, 2.0, 1.0, 6.0], [0.4, 1.0, 2.0, 5.0]])
    print(ai.forward(x))

    # params = ai.parameters()
    # for p in params:
        # print(p)


    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(ai.parameters(), lr=0.001)

    epochs = 15000

    losses = []
    for i in range(epochs):
        num1 = rand_value()
        num2 = rand_value()
        num3 = rand_value()
        num4 = rand_value()
        input = [num1, num2, num3, num4]
        output = torch.tensor([num1 * num2 / num3 + num4*1000])
        pred = ai.forward(torch.tensor(input))
        loss = criterion(pred, output)

        print(f'epoch: {i}, loss: {loss}')
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(losses[0])
    plt.plot(range(epochs),losses)
    plt.show()