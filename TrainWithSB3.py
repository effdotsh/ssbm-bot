from wandb.integration.sb3 import WandbCallback
import wandb
from stable_baselines3 import A2C, DQN, PPO
from GymEnv import SmashEnv


env = SmashEnv(wandb)
wandb.init(project="SmashBot", name="SB3-PPO")

model = PPO("MlpPolicy", env, verbose=1, learning_rate=1e-4)
model.learn(total_timesteps=60 * 60 * 60 * 100)  # 100 hours
model.save("dqn_smash")
