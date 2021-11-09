#!/usr/bin/python3
import argparse
import math
import gameManager
import gymEnv
import melee
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.callbacks import CheckpointCallback

def check_port(value):
    ivalue = int(value)
    if ivalue < 1 or ivalue > 4:
        raise argparse.ArgumentTypeError("%s is an invalid controller port. \
                                         Must be 1, 2, 3, or 4." % value)
    return ivalue


parser = argparse.ArgumentParser(description='Example of libmelee in action')
parser.add_argument('--port', '-p', type=check_port,
                    help='The controller port (1-4) your AI will play on',
                    default=1)
parser.add_argument('--opponent', '-o', type=check_port,
                    help='The controller port (1-4) the opponent will play on',
                    default=2)
parser.add_argument('--debug', '-d', action='store_true',
                    help='Debug mode. Creates a CSV of all game states')
parser.add_argument('--address', '-a', default="127.0.0.1",
                    help='IP address of Slippi/Wii')
parser.add_argument('--dolphin_executable_path', '-e',
                    help='The directory where dolphin is',
                    default='C:/Users/human-w/AppData/Roaming/Slippi Launcher/netplay/')
parser.add_argument('--connect_code', '-t', default="",
                    help='Direct connect code to connect to in Slippi Online')
parser.add_argument('--iso', default='SSBM.iso', type=str,
                    help='Path to melee iso.')

args: gameManager.Args = parser.parse_args()


env = gymEnv.CharacterEnv(args=args, player_port=args.port, opponent_port=args.opponent)

model = PPO("MlpPolicy", env, learning_rate=1e-4)


checkpoint_callback = CheckpointCallback(save_freq=3600, save_path='./logs-ppo/',
                                         name_prefix='rl_model', verbose=3)

model.learn(total_timesteps=5e50, callback=checkpoint_callback)

