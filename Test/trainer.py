from network import Overseer
import numpy as np
from tqdm import tqdm

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