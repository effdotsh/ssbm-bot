import random

from .DQN_Outs import DQN as DQNAgent
import numpy as np


class DQN:
    def __init__(self, obs_dim, action_dim, **kwargs):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.agent = DQNAgent(obs_dim=obs_dim+action_dim, action_dim=1, **kwargs)

    def create_input_tensor(self, obs, action):
        action_inputs = np.zeros(self.action_dim)
        if action != -1:
            action_inputs[action] = 1

        input_tensor = np.concatenate([obs, action_inputs])
        return input_tensor

    def learn_experience(self, obs, action, reward, new_obs, done):
        obs_tensor = self.create_input_tensor(obs, action)


        new_action = self.predict(new_obs)
        new_obs_tensor = self.create_input_tensor(new_obs, new_action)
        self.agent.learn_experience(obs=obs_tensor, action=0, reward=reward, new_obs=new_obs_tensor, done=done)

    def predict(self, obs):
        action = 0
        if random.random() < self.agent.epsilon:
            action = random.randint(0, self.action_dim-1)
        else:
            max_q = None
            for a in range(0, self.action_dim):
                input_tensor = self.create_input_tensor(obs, a)
                q_val = float(self.agent.get_qs(input_tensor))
                if a == 0 or q_val > max_q:
                    max_q = q_val
                    action = a
        return action

    def train(self):
        self.agent.train()

    def get_log(self):
        return self.agent.get_log()
