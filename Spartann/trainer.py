from Spartnn import Overseer
import numpy as np
from tqdm import tqdm
from wallGoalEnv import NavEnv
from randNumEnv import TestEnv
from matplotlib import pyplot as plt

from torch import nn
def graph(points):
    avg = 200
    graph = []
    for i in range(avg, len(points)):
        graph.append(np.mean(points[i - avg:i]))
    print(graph)
    plt.plot(graph)
    plt.show()

if __name__ =='__main__':

    stateNet = nn.Sequential(
        nn.Linear(4, 10),
        # nn.ReLU(),
        # nn.Linear(10, 10),
        # nn.Linear(10, 10),
        nn.Sigmoid(),
        nn.Linear(10, 1),
    )

    # env = NavEnv()
    # nn = Overseer(reward_network_layers=stateNet, num_inputs=4, num_choices=5, epsilon_greedy_chance=1, epsilon_greedy_decrease=0.00001, discount_factor=0.95, search_depth=1)
    #
    env = TestEnv()
    nn = Overseer(reward_network_layers=stateNet, num_inputs=2, num_choices=2, epsilon_greedy_chance=1, epsilon_greedy_decrease=0.00001, search_depth=1)




    rewards = []
    state = env.reset()
    for i in tqdm(range(200_000)):
        action = nn.predict(state)
        next_state, reward, done, _callback = env.step(action)

        nn.learn_reward(chosen_action=action, inputs=state, observed_reward=reward)

        if done:
            rewards.append(reward)

            state = env.reset()
        else:
            nn.learn_state(chosen_action=action, old_state=state, new_state=next_state)

        state = next_state

    #
    # rewards = []
    # state = env.reset()
    # for i in tqdm(range(500_000)):
    #     action = nn.predict(state)
    #     next_state, reward, done, _callback = env.step(action)
    #
    #     nn.learn(chosen_action=action, inputs=state, observed_reward=reward)
    #     state = next_state
    #     rewards.append(reward)


    graph(rewards)
    graph(nn.reward_network_loss)
    graph(nn.state_network_loss)