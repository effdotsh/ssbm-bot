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
    output = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0, 0.0]
    action:np.ndarray = action[0]
    b = np.argmax(action[:7])
    if action[b] > -0.70:
        output[b] = 1

    print(action[:7])
    for i in range(7, 11):
        if abs(action[i]) > 0.3:
            output[i] = 1 * np.sign(action[i])
    output[11] = action[11]
    output[12] = action[12]

    return output
if __name__ == '__main__':
    character = melee.Character.CPTFALCON
    opponent = melee.Character.JIGGLYPUFF if not args.compete else character
    stage = melee.Stage.FINAL_DESTINATION
    file_name = f'{character.name}_v_{opponent.name}_on_{stage.name}'
    print(file_name)

    model = load_model(file_name)
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

        inp = DataHandler.generate_input(gamestate, game.controller.port, game.controller_opponent.port)
        # print(inp)
        a = model.predict(np.array([inp]))

        action = decode_from_model(a)

        action = validate_action(action, gamestate, game.controller.port)
        buttons = [melee.Button.BUTTON_A, melee.Button.BUTTON_B, melee.Button.BUTTON_X, melee.Button.BUTTON_Y, melee.Button.BUTTON_Z, melee.Button.BUTTON_L, melee.Button.BUTTON_R]
        # if action is None:
        #     print('action is none')
        #     continue
        # if len(action) < 8:
        #     print(action)

        button_used = False
        for e, active in enumerate(action[:7]):
            if active == 1:
                button_used = True
                game.controller.press_button(buttons[e])
            else:
                game.controller.release_button(buttons[e])

        game.controller.tilt_analog_unit(melee.Button.BUTTON_MAIN, action[7], action[8])
        game.controller.tilt_analog_unit(melee.Button.BUTTON_C, action[9], action[10])
        # game.controller.press_shoulder(melee.Button.BUTTON_L, action[9])
        # game.controller.press_shoulder(melee.Button.BUTTON_R, action[10])

        game.controller.flush()




        if button_used:
            gamestate = game.get_gamestate()

            game.controller.release_all()



