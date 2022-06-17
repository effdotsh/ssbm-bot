import math
import time

import Args
import GameManager
import melee
import platform

import os
import DataHandler
import numpy as np

import MovesList

args = Args.get_args()
smash_last = False
dud = [0, 0, 0, 0, 0, 0, 0, (0.5, 0.5), (0.5, 0.5), 0.0, 0.0]
def validate_action(action, gamestate: melee.GameState, port:int):
    global smash_last
    player:melee.PlayerState = gamestate.players.get(port)
    if player.action in MovesList.smashes:
        if smash_last:
           return dud
        smash_last = True
    else:
        smash_last = False
    return action


if __name__ == '__main__':
    character = melee.Character.CPTFALCON
    opponent = melee.Character.CPTFALCON if not args.compete else character
    stage = melee.Stage.FINAL_DESTINATION
    print(f'{character.name} vs. {opponent.name} on {stage.name}')

    tree, map = DataHandler.load_model(player_character=character, opponent_character=opponent, stage=stage)
    print('loaded')

    game = GameManager.Game(args)
    game.enterMatch(cpu_level=0, opponant_character=opponent,
                    player_character=character,
                    stage=stage, rules=False)

    num_buttons = len(DataHandler.buttons) + 1
    axis_size = 3
    num_c = 5

    last_action = 120
    while True:
        gamestate = game.get_gamestate()
        # print('----------')
        # [A, B, X, Y, Z, L, R, MAIN_STICK, C_STICK, L_SHOULDER, R_SHOULDER]
        action = DataHandler.predict(tree=tree, map=map, gamestate=gamestate, player_port=game.controller.port,
                                            opponent_port=game.controller_opponent.port)

        action = validate_action(action, gamestate, game.controller.port)
        buttons = [melee.Button.BUTTON_A, melee.Button.BUTTON_B, melee.Button.BUTTON_X, melee.Button.BUTTON_Y, melee.Button.BUTTON_Z, melee.Button.BUTTON_L, melee.Button.BUTTON_R]
        # if action is None:
        #     print('action is none')
        #     continue
        # if len(action) < 8:
        #     print(action)
        for e, active in enumerate(action[:7]):
            if active == 1:
                game.controller.press_button(buttons[e])
            else:
                game.controller.release_button(buttons[e])

        game.controller.tilt_analog(melee.Button.BUTTON_MAIN, action[7][0], action[7][1])
        game.controller.tilt_analog(melee.Button.BUTTON_C, action[8][0], action[8][1])
        # game.controller.press_shoulder(melee.Button.BUTTON_L, action[9])
        # game.controller.press_shoulder(melee.Button.BUTTON_R, action[10])

        game.controller.flush()
        print(action)


