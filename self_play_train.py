import copy
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

character = melee.Character.MARTH
stage = melee.Stage.FINAL_DESTINATION


def load_model(path: str):
    path = f'models2/{path}.pkl'
    if os.path.exists(path):
        with open(path, 'rb') as file:
            data = pickle.load(file)
            return data
    else:
        print("Model does not exist")
        quit()

def mutate_model(m: keras.Model):
    model = copy.deepcopy(m)

    return model

if __name__ == '__main__':
    file_name = f'{character.name}_v_{character.name}_on_{stage.name}'
    champion_model: keras.Model = load_model(file_name)

    print(file_name)

    generation_counter = 0

    while True:
        challenger_model = mutate_model(champion_model)

        game = GameManager.Game(args)
        game.enterMatch(cpu_level=0, opponant_character=character,
                        player_character=character,
                        stage=stage, rules=False)


        bot1 = Bot(model=champion_model, controller=game.controller, opponent_controller=game.opponent_controller)
        bot2 = Bot(model=challenger_model, controller=game.opponent_controller, opponent_controller=game.controller)
        gamestate = game.get_gamestate()


        challenger_wins = False
        while gamestate.menu_state not in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:

            gamestate = game.get_gamestate()
            bot1.act(gamestate)
            # bot2.act(gamestate)

            player: melee.PlayerState = gamestate.players.get(bot1.controller.port)
            if player.action in MovesList.dead_list:
                if player.stock == 1:
                    challenger_wins = True

        if challenger_wins:
            with open(f'generated_models/{file_name}_{generation_counter}', 'wb') as file:
                pickle.dump(challenger_model, file)
            champion_model = challenger_model
        else:
            print('fail')
        generation_counter += 1