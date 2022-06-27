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
from Bot import Bot
args = Args.get_args()
smash_last = False

character = melee.Character.FALCO
opponent = melee.Character.FALCO
stage = melee.Stage.FINAL_DESTINATION
level=9


def load_model(path: str):
    path = f'models2/{file_name}.pkl'
    if os.path.exists(path):
        with open(path, 'rb') as file:
            data = pickle.load(file)
            return data
    else:
        print("Model does not exist")
        quit()


if __name__ == '__main__':
    drop_every = 30
    file_name = f'{character.name}_v_{opponent.name}_on_{stage.name}'
    print(file_name)

    model = load_model(file_name)
    print('loaded')

    game = GameManager.Game(args)
    game.enterMatch(cpu_level=level, opponant_character=opponent,
                    player_character=character,
                    stage=stage, rules=False)

    last_action = 120
    fc = 0


    bot = Bot(model=model, controller=game.controller, opponent_controller=game.opponent_controller)
    while True:
        gamestate = game.get_gamestate()
        bot.act(gamestate)