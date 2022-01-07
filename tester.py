from Spartnn.Spartnn import Overseer as DQNAgent
from tqdm import tqdm
import numpy as np
''
from randNumEnv import TestEnv


env = TestEnv()

num_episodes=30_000

agent = DQNAgent(num_inputs=2, num_outputs=2, min_replay_size=128, batch_size=64, epsilon_greedy_decay=0.999, search_depth=0)


# Iterate over episodes
for episode in tqdm(range(1, num_episodes + 1)):
    # Restarting episode - reset episode reward and step number
    episode_reward = 0
    step = 1

    current_state = env.reset()

    done = False
    while not done:

        action = agent.predict(current_state)

        new_state, reward, done, _callback = env.step(action)

        episode_reward += reward


        agent.update_replay_memory((current_state, action, reward, new_state, done))
        agent.train()

        current_state = new_state
        step += 1




def graph(points):
    avg = 100
    graph = []
    epsilon=[1]
    for i in range(avg, len(points)):
        graph.append(np.mean(points[i - avg:i]))
        epsilon.append(epsilon[-1]*.999)

    e = []
    for v in epsilon:
        e.append(1-v)
    pyplot.plot(graph)
    pyplot.plot(e)
    pyplot.show()
if __name__ == '__main__':
    from  matplotlib import pyplot
    graph(env.rewards)