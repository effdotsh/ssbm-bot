import random

import numpy as np
import torch
from torch import nn

from collections import deque

from Spartnn.spartnn_components import RewardPredictor, StatePredictor, Node, generate_input_tensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


class Overseer:
    # The goal of this network is to guess the reward returned by an acrtion in advance, and use backprop to update from an observed reward. An action can be chosen by testing each input through the reward predictor and choosing the one with the highest reward
    def __init__(self, num_inputs, num_outputs, epsilon_greedy_chance=1, epsilon_greedy_decay=0.9999,
                 reward_network_learning_rate=0.00003, state_network_learning_rate=0.0003, search_depth=2,
                 discount_factor=0.999, reward_network_layers=None, state_network_layers=None,
                 max_replay_size=10_000_000, min_replay_size=10_000, batch_size=256, update_every=500):
        self.update_every = update_every
        self.batch_size = batch_size
        self.min_replay_size = min_replay_size
        self.max_replay_size = max_replay_size
        self.step = 0
        self.first_run = False

        self.search_depth = search_depth
        self.discount_factor = discount_factor
        self.num_inputs: int = num_inputs
        self.num_outputs: int = num_outputs

        self.reward_network = RewardPredictor(num_inputs=num_inputs, num_choices=num_outputs,
                                              layers=reward_network_layers)
        self.reward_network_criterion = nn.MSELoss()
        self.reward_network_optimizer = torch.optim.Adam(self.reward_network.parameters(),
                                                         lr=reward_network_learning_rate)
        self.reward_network_loss = []

        self.state_predictor = StatePredictor(num_inputs=num_inputs, num_choices=num_outputs,
                                              layers=state_network_layers)
        self.state_network_criterion = nn.MSELoss()
        self.state_network_optimizer = torch.optim.Adam(self.reward_network.parameters(),
                                                        lr=state_network_learning_rate)
        self.state_network_loss = []

        self.epsilon_greedy_chance = epsilon_greedy_chance
        self.epsilon_greedy_decay = epsilon_greedy_decay

        self.rewards = []
        self.frame = 0

        self.replay_memory = deque(maxlen=max_replay_size)

    def predict(self, inputs, out_eps=False):
        base_node = Node(inputs=inputs, state_predictor=self.state_predictor,
                         reward_predictor=self.reward_network, discount_factor=self.discount_factor,
                         num_choices=self.num_outputs, max_depth=self.search_depth)
        output = base_node.get_action()

        # Implement epsilon greedy policy
        if random.random() < self.epsilon_greedy_chance:
            new_out = random.randint(0, self.num_outputs - 1)
            if out_eps:
                print(f'{output} -> {new_out}')
            output = new_out

        if self.first_run:
            self.epsilon_greedy_chance *= self.epsilon_greedy_decay

        return output

    def learn_reward(self, chosen_action, inputs: np.ndarray, observed_reward: float):
        inputs = inputs.astype(float)

        network_in = generate_input_tensor(num_choices=self.num_outputs, chosen_action=chosen_action, inputs=inputs)
        predicted_reward_tensor = self.reward_network.forward(network_in)
        loss = self.reward_network_criterion(predicted_reward_tensor.float(), torch.tensor([observed_reward]).float())
        self.reward_network_optimizer.zero_grad()

        loss.backward()
        self.reward_network_optimizer.step()
        self.reward_network_loss.append(loss.item())
        self.rewards.append(observed_reward)

    def learn_state(self, chosen_action, old_state: np.ndarray, new_state: np.ndarray):
        old_state = old_state.astype(float)
        new_state = new_state.astype(float)
        # print(type(new_state))
        # if not isinstance(old_state, np.ndarray):
        #     print('ewfoibweo')
        #     old_state = np.array(old_state)
        # if not isinstance(new_state, np.ndarray):
        #     new_state = np.array(new_state)

        old_state = old_state.astype(np.float32)
        new_state = new_state.astype(np.float32)

        network_in = generate_input_tensor(num_choices=self.num_outputs, chosen_action=chosen_action, inputs=old_state)
        predicted_state_tensor = self.state_predictor.forward(network_in)

        loss = self.state_network_criterion(predicted_state_tensor, torch.tensor(new_state))
        self.state_network_optimizer.zero_grad()
        loss.backward()
        self.state_network_optimizer.step()
        self.state_network_loss.append(loss.item())

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)
        self.step += 1
        self.frame += 1

        if self.step % self.update_every == 0:
            self.train()
            self.log()

    def train(self):
        if len(self.replay_memory) < self.min_replay_size or len(self.replay_memory) < self.batch_size:
            return

        self.first_run = True
        batch_indices = np.unique(
            [random.randint(0, len(self.replay_memory) - 1) for i in range(len(self.replay_memory))])
        batch_indices.sort()

        for e, index in enumerate(batch_indices):

            transition = self.replay_memory[index - e]
            # transition is tuple (old_state, action, reward, new_state, done)
            self.learn_reward(chosen_action=transition[1], inputs=transition[0], observed_reward=transition[2])
            if not transition[4]:
                self.learn_state(chosen_action=transition[1], old_state=transition[0], new_state=transition[3])

            # Remove transition from replay memory
            del self.replay_memory[index - e]

    def log(self, history: int = 400):
        history = min([history, len(self.rewards), len(self.state_network_loss), len(self.reward_network_loss)])
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
