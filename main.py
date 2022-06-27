#!/usr/bin/python3
import copy

import melee

import os
import json
from tqdm import tqdm
import time
import numpy as np

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
import pickle

import Args
from DataHandler import get_ports, controller_states_different, generate_input, generate_output
import MovesList

args = Args.get_args()

player_character = melee.Character.FOX
opponent_character = melee.Character.CPTFALCON
stage = melee.Stage.BATTLEFIELD


# nothing_chance = 0.05
def create_model(replay_paths, player_character: melee.Character,
                 opponent_character: melee.Character, stage: melee.Stage):
    X = []
    Y = []
    for path in tqdm(replay_paths):
        console = melee.Console(is_dolphin=False,
                                allow_old_version=True,
                                path=path)
        try:
            console.connect()
        except:
            console.stop()
            print('console failed to connect', path, time.time())
            continue

        gamestate: melee.GameState = console.step()
        player_port, opponent_port = get_ports(gamestate, player_character=player_character,
                                               opponent_character=opponent_character)
        if player_port == -1:
            print('bad port', path, gamestate.players.keys(), time.time())

            continue

        last_player: melee.PlayerState = copy.deepcopy(gamestate.players.get(player_port))
        while True:
            try:
                gamestate: melee.GameState = console.step()
            except:
                break
            if gamestate is None or gamestate.stage is None:
                break

            player: melee.PlayerState = gamestate.players.get(player_port)
            opponent: melee.PlayerState = gamestate.players.get(opponent_port)
            if player is None or opponent is None:
                break

            if player.action in MovesList.dead_list:
                continue

            if not controller_states_different(player, last_player):
                continue

            last_player = player

            inp = generate_input(gamestate=gamestate, player_port=player_port, opponent_port=opponent_port)

            action = generate_output(player)
            if action == 19 or action == -1:
                continue
            out = np.zeros(19)
            out[action] = 1

            if inp is None:
                break
            if action is None:
                break

            X.append(inp)
            Y.append(out)

    X = np.array(X)
    Y = np.array(Y)

    print(len(X), len(Y))
    print(len(X[0]), len(Y[0]))

    # train
    model = Sequential([
        Dense(64, activation='tanh', input_shape=(len(X[0]),)),
        Dense(64, activation='tanh'),
        Dense(64, activation='tanh'),
        Dense(len(Y[0]), activation='tanh'),
    ])

    opt = keras.optimizers.Adam(
        learning_rate=1e-4,
        name="Adam",
    )

    model.compile(
        optimizer=opt,
        loss='mean_squared_error',
        metrics=['accuracy'],
    )

    model.fit(
        X,  # training data
        Y,  # training targets
        shuffle=True
    )

    folder = 'models2'
    pickle_file_path = f'{folder}/{player_character.name}_v_{opponent_character.name}_on_{stage.name}.pkl'

    if not os.path.isdir(folder):
        os.mkdir(f'{folder}/')

    with open(pickle_file_path, 'wb') as file:
        pickle.dump(model, file)


if __name__ == '__main__':
    # print(f'{player_character.name} vs. {opponent_character.name} on {stage.name}')

    f = open('replays2.json', 'r')
    j = json.load(f)
    # replay_paths = j[f'{player_character.name}_{opponent_character.name}'][stage.name]
    #
    # create_model(replay_paths=replay_paths, player_character=player_character,
    #              opponent_character=opponent_character, stage=stage)

    characters = [melee.Character.FOX, melee.Character.MARTH, melee.Character.FALCO, melee.Character.CPTFALCON,
                  melee.Character.JIGGLYPUFF]

    for c1 in characters:
        for c2 in characters:
            for s in [melee.Stage.BATTLEFIELD, melee.Stage.FINAL_DESTINATION]:
                print(f'{c1.name} vs. {c2.name} on {s.name}')

                replay_paths = j[f'{c1.name}_{c2.name}'][s.name]

                create_model(replay_paths=replay_paths, player_character=c1,
                             opponent_character=c2, stage=s)
