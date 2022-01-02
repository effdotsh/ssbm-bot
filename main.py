#!/usr/bin/python3
import argparse
import time

import torch

import gameManager
import melee
import platform

import os

from movesList import CharacterMovesets

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
parser.add_argument('--model_path', default='model/dqn', type=str)
parser.add_argument('--load_from', default=-1, type=int)

args: gameManager.Args = parser.parse_args()

start_time = time.time()
if __name__ == '__main__':
    os.makedirs(args.model_path)
    character = melee.Character.JIGGLYPUFF

    game = gameManager.Game(args)
    game.enterMatch(cpu_level=9 if args.load_from < 0 else 0, opponant_character=character, player_character=character)
    step = args.load_from

    if args.load_from < 0:  # Start training against CPU
        agent1 = CharacterController(port=args.port, opponent_port=args.opponent, game=game,
                                     moveset=CharacterMovesets[character.name], min_replay_size=2000, minibatch_size=128,
                                     max_replay_size=300_000,
                                     learning_rate=0.00004, update_target_every=2, discount_factor=0.999,
                                     epsilon_decay=0.9997, epsilon=1)

    else:  # Use TLto self-train
        agent1 = CharacterController(port=args.port, opponent_port=args.opponent, game=game,
                                     moveset=CharacterMovesets[character.name], min_replay_size=2000, minibatch_size=128,
                                     max_replay_size=300_000,
                                     learning_rate=0.00004, update_target_every=2, discount_factor=0.999,
                                     epsilon_decay=0.999, epsilon=0.3)

        agent2 = CharacterController(port=args.opponent, opponent_port=args.port, game=game,
                                     moveset=CharacterMovesets[character.name])

        agent1.model.model.load_state_dict(torch.load(f'{args.model_path}/{character.name}_{step}'))
        agent2.model = agent1.model

    while True:  # Training loop
        gamestate = game.console.step()
        if gamestate is None:
            continue
        agent1.run_frame(gamestate, log=True)
        if args.load_from >= 0:
            agent2.run_frame(gamestate, log=False)

        if (step % 1000 == 0):
            torch.save(agent1.model.model.state_dict(), f'{args.model_path}/{character.name}_{step}')

        step += 1
