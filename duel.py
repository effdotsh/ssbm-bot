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

args = Args.get_args()
smash_last = False
dud = np.zeros(9)
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


def load_model(path: str):
    path = f'models/{file_name}.pkl'
    if os.path.exists(path):
        with open(path, 'rb') as file:
            data = pickle.load(file)
            return data
    else:
        print("Model does not exist")
        quit()


def decode_from_model(action: np.ndarray):
    action = action[0]
    reduce = [418, 427, 454, 481, 472, 463, 436, 409, 445, 13, 67]
    for i in reduce:
        action[i]/=3
    # action[13]/=6

    output = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    a = np.argmax(action)

    # print(a, action[a])
    sticks = a % 81
    button = a // 81

    c_stick = sticks%9
    move_stick = sticks//9

    c_y = c_stick%3
    c_x = c_stick//3

    move_y = move_stick%3
    move_x = move_stick//3

    output[button] = 1

    output[5] = move_x-1
    output[6] = move_y-1

    output[7] = c_x-1
    output[8] = c_y-1

    if (output[7], output[8]) != (0, 0):
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
    print(a, output)
    return output
if __name__ == '__main__':
    character = melee.Character.JIGGLYPUFF
    opponent = melee.Character.CPTFALCON if not args.compete else character
    stage = melee.Stage.BATTLEFIELD
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
    while True:
        gamestate = game.get_gamestate()
        # print('----------')

        inp = DataHandler.generate_input(gamestate, game.controller.port, game.controller_opponent.port)
        # print(inp)
        a = model.predict(np.array([inp]))

        action = decode_from_model(a)

        action = validate_action(action, gamestate, game.controller.port)
        b = melee.enums.Button
        buttons = [[b.BUTTON_X, b.BUTTON_Y], [b.BUTTON_L, b.BUTTON_R], [b.BUTTON_Z], [b.BUTTON_A], [b.BUTTON_B]]

        button_used = False
        for e, active in enumerate(action[:5]):
            if active == 1:
                button_used = True
                game.controller.press_button(buttons[e][0])
            else:
                game.controller.release_button(buttons[e][0])

        if action[0] == 1:
            game.controller.tilt_analog_unit(melee.Button.BUTTON_MAIN, 0, 0)
            game.controller.tilt_analog_unit(melee.Button.BUTTON_C, 0, 0)
        else:
            game.controller.tilt_analog_unit(melee.Button.BUTTON_MAIN, action[5], action[6])
            game.controller.tilt_analog_unit(melee.Button.BUTTON_C, action[7], action[8])
        # game.controller.press_shoulder(melee.Button.BUTTON_L, action[9])
        # game.controller.press_shoulder(melee.Button.BUTTON_R, action[10])

        game.controller.flush()




        if button_used:
            gamestate = game.get_gamestate()
            gamestate = game.get_gamestate()
            gamestate = game.get_gamestate()

            game.controller.release_all()



