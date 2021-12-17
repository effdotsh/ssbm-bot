from network import Overseer
import numpy as np
from tqdm import tqdm
from testEnv import TestEnv

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
    env = TestEnv()
    # num_inputs = int(env.observation_space.shape[0])
    # num_outputs = int(env.action_space.n)
    nn = Overseer(num_inputs=2, num_choices=3, epsilon_greedy_chance=1, epsilon_greedy_decrease=0.00001)

    rewards = []
    state = env.reset()
    for i in tqdm(range(500_000)):
        action = nn.predict(state)
        next_state, reward, done, _callback = env.step(action)

        nn.learn(chosen_action=action, inputs=state, observed_reward=reward)
        state = next_state
        rewards.append(reward)

    graph(rewards)
    graph(nn.loss)