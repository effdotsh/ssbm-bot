#!/usr/bin/python3
import argparse
import math

import game
import random
import melee

# This example program demonstrates how to use the Melee API to run a console,
#   setup controllers, and send button presses over to a console

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
controller = game.getController()
controller_opponent = game.getControllerOpponent()

game.enterMatch()
jumping = False
moveY = 0
while True:
    gamestate = game.getState()

    # Slippi Online matches assign you a random port once you're in game that's different
    #   than the one you're physically plugged into. This helper will autodiscover what
    #   port we actually are.
    discovered_port = args.port
    if args.connect_code != "":
        discovered_port = melee.gamestate.port_detector(gamestate, melee.Character.FOX, 0)
    if discovered_port > 0:
        player_state: melee.PlayerState = gamestate.players.get(discovered_port)
        oppenent_state: melee.PlayerState = gamestate.players.get(1)



        print(player_state.action)
        if(oppenent_state.y > player_state.y and not jumping):
            controller.press_button(melee.Button.BUTTON_Y)
            jumping = True
            moveY = 0

        elif(oppenent_state.y<player_state.y and player_state.on_ground and moveY == 0):
            moveY = -1

        else:
            controller.release_button(melee.Button.BUTTON_Y)
            jumping = False
            moveY = 0


        if(player_state.x < oppenent_state.x and oppenent_state.on_ground):
            controller.tilt_analog_unit(melee.Button.BUTTON_MAIN, 1, moveY)
        elif (player_state.x > oppenent_state.x and oppenent_state.on_ground):
            controller.tilt_analog_unit(melee.Button.BUTTON_MAIN, -1, moveY)
        else:
            controller.tilt_analog(melee.Button.BUTTON_MAIN, 0, 0)

        if(attacking >= 10 and moveY == 0 and math.dist((player_state.x, player_state.y), (oppenent_state.x, oppenent_state.y)) < 5):
            controller.press_button(melee.Button.BUTTON_A)
            attacking = 0
        else:
            controller.release_button(melee.Button.BUTTON_A)
            attacking += 1

        controller.flush()
