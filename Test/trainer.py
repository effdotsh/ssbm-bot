from stable_baselines3 import A2C as MLAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from matplotlib import pyplot as plt
import numpy as np
import testEnv

def graph(points):
    avg = 200
    graph = []
    for i in range(avg, len(points)):
        graph.append(np.mean(points[i-avg:i]))
    print(graph)
    plt.plot(graph)
    plt.show()
if __name__=='__main__':
    env = testEnv.TestEnv()
    model = MLAlgorithm("MlpPolicy", env=env)


    # model.learn(total_timesteps=100_000)
    obs = env.reset()
    for i in range(10_000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        collected = model.collect_rollouts(env, BaseCallback, model.rollout_buffer, 64)
        if(collected):
            model.train()




    graph(env.rewards)