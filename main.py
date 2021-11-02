#!/usr/bin/python3
import argparse
import math

import game
import melee

def check_port(value):
    ivalue = int(value)
    if ivalue < 1 or ivalue > 4:
        raise argparse.ArgumentTypeError("%s is an invalid controller port. \
                                         Must be 1, 2, 3, or 4." % value)
    return ivalue


parser = argparse.ArgumentParser(description='Example of libmelee in action')
parser.add_argument('--port', '-p', type=check_port,
                    help='The controller port (1-4) your AI will play on',
                    default=2)
parser.add_argument('--opponent', '-o', type=check_port,
                    help='The controller port (1-4) the opponent will play on',
                    default=1)
parser.add_argument('--debug', '-d', action='store_true',
                    help='Debug mode. Creates a CSV of all game states')
parser.add_argument('--address', '-a', default="127.0.0.1",
                    help='IP address of Slippi/Wii')
parser.add_argument('--dolphin_executable_path', '-e',
                    help='The directory where dolphin is',
                    default='/home/human/.config/Slippi Launcher/netplay/squashfs-root/usr/bin')
parser.add_argument('--connect_code', '-t', default="",
                    help='Direct connect code to connect to in Slippi Online')
parser.add_argument('--iso', default='SSBM.iso', type=str,
                    help='Path to melee iso.')

args: game.Args = parser.parse_args()



attacking = 0
game = game.Game(args=args)
controller = game.getController(args.port)
controller_opponent = game.getController(args.opponent)

game.enterMatch(stage=melee.Stage.FINAL_DESTINATION)

