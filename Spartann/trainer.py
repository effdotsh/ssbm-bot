from Spartann import Overseer
import numpy as np
from tqdm import tqdm
from navEnv import NavEnv

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
    env = NavEnv()
    # num_inputs = int(env.observation_space.shape[0])
    # num_outputs = int(env.action_space.n)
    nn = Overseer(num_inputs=4, num_choices=5, epsilon_greedy_chance=1, epsilon_greedy_decrease=0.00001)

    rewards = []
    state = env.reset()
    for i in tqdm(range(500_000)):
        action = nn.predict(state)
        next_state, reward, done, _callback = env.step(action)

        state = next_state

        if done:
            env.reset()
            rewards.append(reward)
            nn.learn(chosen_action=action, inputs=state, observed_reward=reward)

    graph(rewards)
    graph(nn.loss)