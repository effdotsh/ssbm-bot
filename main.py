#!/usr/bin/python3
import argparse
import math
import gameManager
import FoxEnv
import melee
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.callbacks import CheckpointCallback


import threading
from queue import Queue

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


game = gameManager.Game(args)
game.enterMatch(cpu_level=0, opponant_character=melee.Character.FOX)

# game2 = gameManager.Game(args)
# game2.connect()

p1_queue = Queue()
p2_queue = Queue()

p1_env = FoxEnv.FoxEnv(game=game, player_port=args.port, opponent_port=args.opponent, queue=p1_queue)
p2_env = FoxEnv.FoxEnv(game=game, player_port=args.opponent, opponent_port=args.port, queue=p2_queue)


checkpoint_callback = CheckpointCallback(save_freq=500, save_path='./fox-a2c/',
                                         name_prefix='rl_model', verbose=3)

def learn(model):
    model.learn(total_timesteps=5e20, callback=checkpoint_callback)





p1_model = A2C("MlpPolicy", p1_env)
# p1_model = A2C.load(path="fox-a2c/rl_model_16500_steps", env=p1_env, learning_rate=0.0007)
p2_model = A2C("MlpPolicy", p2_env)





threading.Thread(target=learn, args=(p1_model,)).start()
# threading.Thread(target=learn, args=(p2_model,)).start()

# p1_model.learn(total_timesteps=5e20, callback=checkpoint_callback)

# for i in range(int(5e20)):
#     p1_model.learn(total_timesteps=1, callback=checkpoint_callback)
#     p2_model.learn(total_timesteps=1, callback=checkpoint_callback)
