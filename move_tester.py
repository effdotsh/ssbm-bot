import time

import Args
import GameManager
import melee
import platform

import os
import DataHandler
import numpy as np


args = Args.get_args()


if __name__ == '__main__':
    character = melee.Character.FOX
    opponent = melee.Character.MARTH if not args.compete else character
    stage = melee.Stage.FINAL_DESTINATION
    print(f'{character.name} vs. {opponent.name} on {stage.name}')


    game = GameManager.Game(args)
    game.enterMatch(cpu_level=0, opponant_character=opponent,
                    player_character=character,
                    stage=stage, rules=False)

    num_buttons = len(DataHandler.buttons) + 1
    axis_size = 3
    num_c = 5
    maxes = [axis_size, axis_size, num_c, num_buttons]

    last_action = 120
    while True:
        gamestate = game.get_gamestate()
        player: melee.PlayerState = gamestate.players.get(game.controller.port)
        print(player.action, player.action_frame)
