from DQNKeras import DQNAgent #2:02


import numpy as np
from tqdm import tqdm
from randNumEnv import TestEnv
from matplotlib import pyplot as plt

from torch import nn
def graph(points):
    avg = 200
    graph = []
    for i in range(avg, len(points)):
        graph.append(np.mean(points[i - avg:i]))
    plt.plot(graph)
    plt.show()

if __name__ =='__main__':


    env = TestEnv()
    nn = DQNAgent(num_inputs=2, num_outputs=2, min_replay_size=128)
    #
    # env = TestEnv()
    # nn = Overseer(num_inputs=2, num_choices=2, epsilon_greedy_chance=1, epsilon_greedy_decrease=0.0001, search_depth=0)




    rewards = []
    state = env.reset()
    for i in tqdm(range(1_000)):
        action = nn.predict(state)
        next_state, reward, done, _callback = env.step(action)
        rewards.append(reward)
        # nn.learn_reward(chosen_action=action, inputs=state, observed_reward=reward)
        nn.update_replay_memory((state, action, reward, next_state, True))
        if i % 100 == 0:
            nn.train(True)
        else:
            nn.train(False)
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
    # graph(nn.)