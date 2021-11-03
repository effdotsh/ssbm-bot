import os
import torch
from torch import nn
import random
import matplotlib.pyplot as plt


class CharacterAI(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.network = nn.Linear(num_inputs, num_outputs)
    def forward(self, inputs):
        return self.network(inputs)

def rand_value():
    return random.random() - 0.5 * 100
if __name__== '__main__':
    torch.manual_seed(1)
    ai = CharacterAI(3, 1)

    x = torch.tensor([[1.0, 2.0, 1.0], [1.0, 2.0, 5.0]])
    print(ai.forward(x))

    [w, b] = ai.parameters()
    w0 = w[0][0].item()
    b0 = b[0].item()
    print(w0, b0)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(ai.parameters(), lr=0.001)

    epochs = 50000

    losses = []
    for i in range(epochs):
        num1 = rand_value()
        num2 = rand_value()
        num3 = rand_value()
        input = [num1, num2, num3]
        output = torch.tensor([num1 * num2 / num3 + 1000])
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