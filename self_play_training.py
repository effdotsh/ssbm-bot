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
import colorama

args = Args.get_args()
smash_last = False

character = melee.Character.CPTFALCON
stage = melee.Stage.FINAL_DESTINATION


def load_model(path: str):
    path = f'{path}'
    if os.path.exists(path):
        with open(path, 'rb') as file:
            data = pickle.load(file)
            return data
    else:
        print("Model does not exist")
        quit()


def mutate_model(m: keras.Model):
    model = copy.deepcopy(m)
    weights = np.array(model.get_weights())
    noise = 0.3
    for perceptron in weights:
        for w in range(len(perceptron)):
                perceptron[w] += 2*(random.random()-0.5) * noise

    model.set_weights(weights)
    return model


if __name__ == '__main__':
    # file_name = f'generated_models/{character.name}_v_{character.name}_on_{stage.name}.pkl'
    file_name = f'generated_models/CPTFALCON_v_CPTFALCON_on_FINAL_DESTINATION_9.pkl'
    champion_model: keras.Model = load_model(file_name)

    print(file_name)

    generation_counter = 9

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
        fc = 60*20
        while gamestate.menu_state in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
            fc -= 1
            gamestate = game.get_gamestate()
            bot1.act(gamestate)
            bot2.act(gamestate)

            champ: melee.PlayerState = gamestate.players.get(bot1.controller.port)
            chall: melee.PlayerState = gamestate.players.get(bot2.controller.port)

            if champ.action in MovesList.dead_list:
                if champ.stock == 1:
                    challenger_wins = True

            if fc == 0 and champ.percent == 0 and chall.percent == 0 and champ.stock == 4 and chall.stock == 4:
                break

        print(gamestate.menu_state)
        game.console.stop()

        time.sleep(1)
        os.system('pkill dolphin-emu')
        time.sleep(1)

        if challenger_wins:
            folder = 'generated_models'
            if not os.path.isdir(folder):
                os.mkdir(f'{folder}/')
            with open(f'{folder}/{file_name}_{generation_counter}.pkl', 'wb') as file:
                pickle.dump(challenger_model, file)
            champion_model = challenger_model
            print(colorama.Fore.GREEN, time.time(), generation_counter, " - Challenger Wins")

        else:
            print(colorama.Fore.RED, time.time(), generation_counter, " - Champion Wins")
        print(colorama.Fore.RESET)
        generation_counter += 1
