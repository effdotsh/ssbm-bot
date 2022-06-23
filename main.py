#!/usr/bin/python3

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


# nothing_chance = 0.05
def create_model(replay_paths, player_character: melee.Character,
                 opponent_character: melee.Character, stage: melee.Stage):
    pickle_file_path = f'models/{player_character.name}_v_{opponent_character.name}_on_{stage.name}.pkl'

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

        last_input = gamestate.players.get(player_port).controller_state
        last_input_opp = gamestate.players.get(opponent_port).controller_state
        while True:
            try:
                gamestate: melee.GameState = console.step()
            except:
                break
            if gamestate is None or gamestate.stage is None:
                break

            player: melee.PlayerState = gamestate.players.get(player_port)
            opponent: melee.PlayerState = gamestate.players.get(opponent_port)

            if player.action in MovesList.dead_list or opponent.action in MovesList.dead_list:
                continue

            new_input = player.controller_state
            if not controller_states_different(new_input, last_input):
                continue

            last_input = new_input

            inp = generate_input(gamestate=gamestate, player_port=player_port, opponent_port=opponent_port)

            action = generate_output(new_input)
            if action == 14:
                continue
            out = np.zeros(14)
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
        # Dense(64, activation='tanh'),
        Dense(len(Y[0]), activation='tanh'),
    ])

    opt = keras.optimizers.Adam(
        learning_rate=3e-3,
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

    if not os.path.isdir('models'):
        os.mkdir('models/')

    with open(pickle_file_path, 'wb') as file:
        pickle.dump(model, file)



if __name__ == '__main__':
    player_character = melee.Character.MARTH
    opponent_character = melee.Character.CPTFALCON
    stage = melee.Stage.BATTLEFIELD

    print(f'{player_character.name} vs. {opponent_character.name} on {stage.name}')

    f = open('replays.json', 'r')
    j = json.load(f)

    replay_paths = j[f'{player_character.name}_{opponent_character.name}'][stage.name]

    create_model(replay_paths=replay_paths, player_character=player_character,
                 opponent_character=opponent_character, stage=stage)


