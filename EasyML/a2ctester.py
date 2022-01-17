from A2CKeras import A2CAgent

from tqdm import tqdm
import numpy as np
import gym

env = gym.make("CartPole-v0")  # Create the environment
env.seed(1)

num_episodes = 30_000

agent = A2CAgent(num_inputs=4, num_actions=2)
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

        agent.update_replay_buffer(reward)
        episode_reward += reward

        current_state = new_state
        step += 1
    agent.train()


def graph(points):
    avg = 100
    graph = []
    epsilon = [1]
    for i in range(avg, len(points)):
        graph.append(np.mean(points[i - avg:i]))
        epsilon.append(epsilon[-1] * .999)

    e = []
    for v in epsilon:
        e.append(1 - v)
    pyplot.plot(graph)
    pyplot.plot(e)
    pyplot.show()


if __name__ == '__main__':
    from matplotlib import pyplot

    graph(env.rewards)
