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


character = melee.Character.FALCO
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
            perceptron[w] += 2 * (random.random() - 0.5) * noise

    model.set_weights(weights)
    return model


if __name__ == '__main__':
    filename = f'{character.name}_v_{character.name}_on_{stage.name}.pkl'

    path = f'models2/{filename}'
    # path = f'generated_models/CPTFALCON_v_CPTFALCON_on_FINAL_DESTINATION_9.pkl'
    champion_model: keras.Model = load_model(path)

    print(path)
    # [[player_stock, player_percent], [opponent_stock, opponent_percent]]
    best_results = [[0, 0], [4, 0]]

    generation_counter = 0

    while True:
        challenger_model = mutate_model(champion_model)

        game = GameManager.Game(args)
        game.enterMatch(cpu_level=9, opponant_character=character,
                        player_character=character,
                        stage=stage, rules=False)

        bot1 = Bot(model=champion_model, controller=game.controller, opponent_controller=game.opponent_controller)

        gamestate = game.get_gamestate()

        results = [[1, 0], [4, 0]]

        while gamestate.menu_state in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
            gamestate = game.get_gamestate()
            bot1.act(gamestate)

            player: melee.PlayerState = gamestate.players.get(bot1.controller.port)
            opponent: melee.PlayerState = gamestate.players.get(bot1.opponent_controller.port)
            if results[0][0] > 0 and results[1][0] > 0:
                results = [[player.stock, player.percent], [opponent.stock, opponent.percent]]

        if results[0][0] == 0:
            results[0][1] = 0
        if results[1][0] == 0:
            results[1][1] = 0

        # best if player ends with more stock or the same stock and less percent. also best if the opponent ends with
        # fewer stocks, or the same # of stocks but a higher percent
        best_yet = (results[0][0] > best_results[0][0] or (
                results[0][0] == best_results[0][0] and results[0][1] < best_results[0][1])) or (
                           results[1][0] < best_results[1][0] or (
                           results[1][0] == best_results[1][0] and results[1][1] > best_results[1][1]))


        print(gamestate.menu_state)
        game.console.stop()

        time.sleep(1)
        os.system('pkill dolphin-emu')
        time.sleep(1)

        if best_yet:
            folder = 'generated_models'
            if not os.path.isdir(folder):
                os.mkdir(f'{folder}/')
            while not os.path.isdir(folder):
                pass
            with open(f'{folder}/{filename}_{generation_counter}.pkl', 'wb') as file:
                pickle.dump(challenger_model, file)
            champion_model = challenger_model
            best_results = results
            print(colorama.Fore.GREEN, time.time(), generation_counter, " - Challenger Wins", results)

        else:
            print(colorama.Fore.RED, time.time(), generation_counter, " - Challenger Loses", results)
        print(colorama.Fore.RESET)
        generation_counter += 1
