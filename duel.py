import math
import pickle
import time

import Args
import GameManager
import melee
import platform

import os
import DataHandler
import numpy as np

import MovesList

import random

args = Args.get_args()
smash_last = False



def validate_action(action, gamestate: melee.GameState, port: int):
    # global smash_last
    player: melee.PlayerState = gamestate.players.get(port)
    edge = melee.stages.EDGE_POSITION.get(gamestate.stage)
    if player.character == melee.enums.Character.MARTH:
        vel_y = player.speed_y_self + player.speed_y_attack
        x = np.sign(player.position.x)
        if player.jumps_left == 0 and player.position.y < -20 and vel_y < 0:
            if abs(player.position.x) < edge-10:
                return [[0, 0, 0], x, 0, 0, 0]
            else:
                return [[0, 1, 0], -0.5 * x, 0.85, 0, 0]
        elif player.y < 0 and player.jumps_left > 0:
            return [[1, 0, 0], x, 0, 0, 0]

    # if player.action in MovesList.smashes:
    #     if smash_last:
    #        return dud
    #     smash_last = True
    # else:
    #     smash_last = False
    return action


def load_model(path: str):
    path = f'models/{file_name}.pkl'
    if os.path.exists(path):
        with open(path, 'rb') as file:
            data = pickle.load(file)
            return data
    else:
        print("Model does not exist")
        quit()




if __name__ == '__main__':
    character = melee.Character.MARTH
    opponent = melee.Character.CPTFALCON if not args.compete else character
    stage = melee.Stage.BATTLEFIELD
    drop_every = 30
    file_name = f'{character.name}_v_{opponent.name}_on_{stage.name}'
    print(file_name)

    model = load_model(file_name)
    print('loaded')

    game = GameManager.Game(args)
    game.enterMatch(cpu_level=9, opponant_character=opponent,
                    player_character=character,
                    stage=stage, rules=False)

    last_action = 120
    fc = 0
    while True:
        gamestate = game.get_gamestate()
        player: melee.PlayerState = gamestate.players.get(game.controller.port)
        fc += 1
        # print('----------')

        inp = DataHandler.generate_input(gamestate, game.controller.port, game.controller_opponent.port)
        # print(inp)
        a = model.predict(np.array([inp]))

        action = DataHandler.decode_from_model(a, player)

        action = validate_action(action, gamestate, game.controller.port)
        b = melee.enums.Button

        button_used = False
        for i in range(len(MovesList.buttons)):
            if action[0][i] == 1:
                button_used = True
                game.controller.press_button(MovesList.buttons[i][0])
            else:
                game.controller.release_button(MovesList.buttons[i][0])

        if action[0][0] == 1:  # jump
            game.controller.tilt_analog_unit(melee.Button.BUTTON_MAIN, 0, 0)
            game.controller.tilt_analog_unit(melee.Button.BUTTON_C, 0, 0)
            gamestate = game.get_gamestate()
            gamestate = game.get_gamestate()
            game.controller.release_all()

            gamestate = game.get_gamestate()

        else:
            game.controller.tilt_analog_unit(melee.Button.BUTTON_MAIN, action[-4], action[-3])
            game.controller.tilt_analog_unit(melee.Button.BUTTON_C, action[-2], action[-1])
        # game.controller.press_shoulder(melee.Button.BUTTON_L, action[9])
        # game.controller.press_shoulder(melee.Button.BUTTON_R, action[10])

        game.controller.flush()

        if button_used:
            gamestate = game.get_gamestate()
            gamestate = game.get_gamestate()
            gamestate = game.get_gamestate()
            fc += 3
            game.controller.release_all()

        if fc >= drop_every:
            game.controller.release_all()
            gamestate = game.get_gamestate()
            gamestate = game.get_gamestate()
            fc = 0
