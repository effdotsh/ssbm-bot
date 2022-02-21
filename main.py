#!/usr/bin/python3
import argparse
import copy
import time

import torch

import GameManager
import melee
import platform

import os

from Agent import Agent, Algorithm

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
parser.add_argument('--model_path', default='model/dqn', type=str)
parser.add_argument('--load_from', default=-1, type=int)
parser.add_argument('--compete', default=False, action='store_true')
parser.add_argument('--cpu_level', default=0, type=int)
parser.add_argument('--wandb', default=False, action='store_true')

args: GameManager.Args = parser.parse_args()

start_time = time.time()
if __name__ == '__main__':
    character = melee.Character.FOX
    opponent = melee.Character.CPTFALCON if not args.compete else character

    if not os.path.isdir(f'{args.model_path}/{character.name}'):
        os.makedirs(f'{args.model_path}/{character.name}')

    game = GameManager.Game(args)
    game.enterMatch(cpu_level=args.cpu_level if not args.compete else 0, opponant_character=opponent,
                    player_character=character,
                    stage=melee.Stage.FINAL_DESTINATION)
    step = args.load_from

    agent1 = Agent(player_port=args.port, opponent_port=args.opponent, game=game, algorithm=Algorithm.SAC)

    while True:  # Training loop
        gamestate = game.get_gamestate()
        agent1.run_frame(gamestate)

        step += 1
