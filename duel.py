import math
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

    tree, map = DataHandler.load_model(player_character=character, opponent_character=opponent, stage=stage)
    print('loaded')

    game = GameManager.Game(args)
    game.enterMatch(cpu_level=9, opponant_character=opponent,
                    player_character=character,
                    stage=stage, rules=False)

    num_buttons = len(DataHandler.buttons) + 1
    axis_size = 3
    num_c = 5
    maxes = [axis_size, axis_size, num_c, num_buttons]

    last_action = 120
    while True:
        gamestate = game.get_gamestate()
        # print('----------')
        # [A, B, X, Y, Z, L, R, MAIN_STICK, C_STICK, L_SHOULDER, R_SHOULDER]
        action = DataHandler.predict(tree=tree, map=map, gamestate=gamestate, player_port=game.controller.port,
                                            opponent_port=game.controller_opponent.port, maxes=maxes)

        buttons = [melee.Button.BUTTON_A, melee.Button.BUTTON_B, melee.Button.BUTTON_X, melee.Button.BUTTON_Y, melee.Button.BUTTON_Z, melee.Button.BUTTON_L, melee.Button.BUTTON_R]
        for e, active in enumerate(action[:7]):
            if active == 1:
                game.controller.press_button(buttons[e])
            else:
                game.controller.release_button(buttons[e])

        game.controller.tilt_analog(melee.Button.BUTTON_MAIN, action[7][0], action[7][1])
        game.controller.tilt_analog(melee.Button.BUTTON_C, action[8][0], action[8][1])
        game.controller.press_shoulder(melee.Button.BUTTON_L, action[9])
        game.controller.press_shoulder(melee.Button.BUTTON_R, action[10])

        game.controller.flush()
        print(action)


