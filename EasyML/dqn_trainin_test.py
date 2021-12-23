from DQN import DQNAgent
from tqdm import tqdm
import numpy as np
''
from randNumEnv import TestEnv


env = TestEnv()

EPISODES=5_000
agent = DQNAgent(num_inputs=2, num_outputs=2, min_replay_size=128)

ep_rewards = [-1]

epsilon = 1
MIN_EPSILON = 0.01
EPSILON_DECAY = 0.9
AGGREGATE_STATS_EVERY = 5
# Iterate over episodes
for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):

    # Update tensorboard step every episode
    agent.tensorboard.step = episode

    # Restarting episode - reset episode reward and step number
    episode_reward = 0
    step = 1

    # Reset environment and get initial state
    current_state = env.reset()

    # Reset flag and start iterating until episode ends
    done = False
    while not done:

        # This part stays mostly the same, the change is to query a model for Q values
        if np.random.random() > epsilon:
            # Get action from Q table
            action = np.argmax(agent.get_qs(current_state))
        else:
            # Get random action
            action = np.random.randint(0, agent.num_outputs)

        new_state, reward, done, _callback = env.step(action)

        # Transform new continous state to new discrete state and count reward
        episode_reward += reward


        # Every step we update replay memory and train main network
        agent.update_replay_memory((current_state, action, reward, new_state, done))
        agent.train(done, step)

        current_state = new_state
        step += 1

    # Append episode reward to a list and log stats (every given number of episodes)
    ep_rewards.append(episode_reward)
    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
        min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
        max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
        # agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)


    # Decay epsilon
    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)

def graph(points):
    avg = 200
    graph = []
    for i in range(avg, len(points)):
        graph.append(np.mean(points[i - avg:i]))
    pyplot.plot(graph)
    pyplot.show()
if __name__ == '__main__':
    from  matplotlib import pyplot

    graph(env.rewards)
    graph(ep_rewards)