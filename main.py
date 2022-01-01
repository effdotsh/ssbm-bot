#!/usr/bin/python3
import argparse
import math
import time

import gameManager
import melee
import platform

# from EasyML.Spartnn import Overseer

from EasyML.DQNTorch import DQNAgent

from CharacterGymEnv import CharacterEnv

from  movesList import CharacterMovesets

from CharacterController import CharacterController

dolphin_path = ''
if platform.system() == "Darwin":
    dolphin_path = "/Users/human/Library/Application Support/Slippi Launcher/netplay/Slippi Dolphin.app"
elif platform.system() == "Windows":
    dolphin_path = "C:/Users/human/AppData/Roaming/Slippi Launcher/netplay/"
elif platform.system() == "Linux":
    dolphin_path = "/home/human/.config/Slippi Launcher/netplay/squashfs-root/usr/bin"

print(dolphin_path)


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
                    default=dolphin_path)
parser.add_argument('--connect_code', '-t', default="",
                    help='Direct connect code to connect to in Slippi Online')
parser.add_argument('--iso', default='SSBM.iso', type=str,
                    help='Path to melee iso.')

args: gameManager.Args = parser.parse_args()

start_time = time.time()
if __name__ == '__main__':
    game = gameManager.Game(args)
    game.enterMatch(cpu_level=6, opponant_character=melee.Character.MARTH)

    character = CharacterController(port=args.port, opponent_port=args.opponent, game=game, moveset=CharacterMovesets.FOX, min_replay_size=1500, minibatch_size=128, max_replay_size=300_000,
                     learning_rate=0.00004, update_target_every=5, discount_factor=0.999, epsilon_decay=0.9997, epsilon=1)


    while True:
        gamestate = game.console.step()
        character.run_frame(gamestate)