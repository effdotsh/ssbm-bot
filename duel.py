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
def validate_action(action, gamestate: melee.GameState, port:int):
    # global smash_last
    player: melee.PlayerState = gamestate.players.get(port)
    edge = melee.stages.EDGE_POSITION.get(gamestate.stage)
    if player.character == melee.enums.Character.MARTH:
        vel_y = player.speed_y_self + player.speed_y_attack
        if player.jumps_left == 0 and player.position.y < -20 and vel_y < 0:
            x = np.sign(player.position.x)
            if abs(player.position.x) < edge:
                return [0, 1, x, 0, 0, 0]
            else:
                return [0, 1, -0.4* x, 0.9, 0, 0]
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


def decode_from_model(action: np.ndarray, player: melee.PlayerState= None):
    action = action[0]
    if player is not None and player.y > 0:
        reduce = [175, 229, 13, 67]
        for i in reduce:
            action[i]/=1.5
        action[40]/=2

    output = [0, 0, 0, 0, 0, 0]
    # a = random.choices(list(range(len(action))), weights=action, k=1)[0]
    a = np.argmax(action)

    sticks = a % 81
    button = a // 81

    c_stick = sticks%9
    move_stick = sticks//9

    c_y = c_stick%3
    c_x = c_stick//3

    move_y = move_stick%3
    move_x = move_stick//3

    output[button] = 1

    output[2] = move_x-1
    output[3] = move_y-1

    output[4] = c_x-1
    output[5] = c_y-1

    if (output[4], output[5]) != (0, 0):
        print(time.time())
    # print(move_x, move_y)
    # action:np.ndarray = action[0]
    # b = np.argmax(action[:7])
    # if action[b] > -0.70:
    #     output[b] = 1
    #
    # print(action[:7])
    # for i in range(7, 11):
    #     if abs(action[i]) > 0.3:
    #         output[i] = 1 * np.sign(action[i])
    # output[11] = action[11]
    # output[12] = action[12]
    print(a, output, action[a])
    return output
if __name__ == '__main__':
    character = melee.Character.MARTH
    opponent = melee.Character.CPTFALCON if not args.compete else character
    stage = melee.Stage.BATTLEFIELD
    drop_every = 7
    file_name = f'{character.name}_v_{opponent.name}_on_{stage.name}'
    print(file_name)

    model = load_model(file_name)
    print('loaded')

    game = GameManager.Game(args)
    game.enterMatch(cpu_level=9, opponant_character=opponent,
                    player_character=character,
                    stage=stage, rules=False)

    num_buttons = len(DataHandler.buttons) + 1
    axis_size = 3
    num_c = 5

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

        action = decode_from_model(a, player)

        action = validate_action(action, gamestate, game.controller.port)
        b = melee.enums.Button
        # buttons = [[b.BUTTON_X, b.BUTTON_Y], [b.BUTTON_L, b.BUTTON_R], [b.BUTTON_Z], [b.BUTTON_A], [b.BUTTON_B]]
        buttons = [[b.BUTTON_X, b.BUTTON_Y], [b.BUTTON_B]]

        button_used = False
        for i in range(len(buttons)):
            if action[i] == 1:
                button_used = True
                game.controller.press_button(buttons[i][0])
            else:
                game.controller.release_button(buttons[i][0])

        if action[0] == 1:
            game.controller.tilt_analog_unit(melee.Button.BUTTON_MAIN, 0, 0)
            game.controller.tilt_analog_unit(melee.Button.BUTTON_C, 0, 0)
            gamestate = game.get_gamestate()
            gamestate = game.get_gamestate()
            gamestate = game.get_gamestate()

        else:
            game.controller.tilt_analog_unit(melee.Button.BUTTON_MAIN, action[2], action[3])
            game.controller.tilt_analog_unit(melee.Button.BUTTON_C, action[4], action[5])
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

