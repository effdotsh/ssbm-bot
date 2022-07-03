import math
import pickle
import time

import keras

import Args
import GameManager
import melee
import platform

import os
import DataHandler
import numpy as np

import MovesList

import random
from Bot import Bot
args = Args.get_args()
smash_last = False

player_character = melee.Character.MARTH
opponent_character = melee.Character.CPTFALCON
stage = melee.Stage.FINAL_DESTINATION
level=9


def load_model(path: str):
    path = f'{path}'
    if os.path.exists(path):
        with open(path, 'rb') as file:
            data = pickle.load(file)
            return data
    else:
        print("Model does not exist")
        quit()


if __name__ == '__main__':
    file_name = f'models2/{player_character.name}_v_{opponent_character.name}_on_{stage.name}.pkl'
    # file_name = 'generated_models/old/FALCO_v_FALCO_on_FINAL_DESTINATION.pkl_9.pkl'
    print(file_name)

    model: keras.Model = load_model(file_name)
    game = GameManager.Game(args)
    game.enterMatch(cpu_level=level, opponant_character=opponent_character,
                    player_character=player_character,
                    stage=stage, rules=False)

    bot1 = Bot(model=model, controller=game.controller, opponent_controller=game.opponent_controller)
    # bot2 = Bot(model=model, controller=game.opponent_controller, opponent_controller=game.controller)

    while True:
        gamestate = game.get_gamestate()
        bot1.act(gamestate)
        # bot2.act(gamestate)