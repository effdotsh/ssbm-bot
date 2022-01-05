from DQNTorch import DQNAgent
import random
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot
agent = DQNAgent(num_inputs=1, num_outputs=2, min_replay_size=100, update_target_every=1, epsilon_decay=0.5)
rewards = []

for i in tqdm(range(100_000)): #epoch
        target_x = 1 if random.random() > 0.5 else -1


        action = agent.predict(np.array([target_x])) * 2 -1

        reward = 1 if action == target_x else 0
        agent.update_replay_memory(
            (np.array([target_x]), action, reward, np.array([target_x]), True)
        )


            # print(f'{current_x}, {target_x}')

        rewards.append(reward)
        if i % 100 == 0:
            agent.train(True)



def graph(points):
    avg = 200
    graph = []
    for i in range(avg, len(points)):
        graph.append(np.mean(points[i - avg:i]))
    pyplot.plot(graph)
    pyplot.show()
if __name__ == '__main__':
    graph(rewards)