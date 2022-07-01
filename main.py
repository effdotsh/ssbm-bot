#!/usr/bin/python3
import copy
from collections import deque

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

player_character = melee.Character.PIKACHU
opponent_character = melee.Character.CPTFALCON
stage = melee.Stage.FINAL_DESTINATION


# nothing_chance = 0.05
def create_model(replay_paths, player_character: melee.Character,
                 opponent_character: melee.Character, stage: melee.Stage, folder: str, lr: float):
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

        player: melee.PlayerState = gamestate.players.get(player_port)

        action_history = deque(maxlen=3)
        last_recorded_player = player
        last_recorded_action = -1
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

            action = generate_output(player)
            if action is None:
                break
            if action == 21 or action == -1:
                continue

            action_history.append(action)

            if action != last_recorded_action and action != -1:
                if action_history[-1] < 11 and action_history[0] >= 11:
                    pass
                elif action_history[-1] >= 11 and action_history[0] < 11 or (
                        action_history[-1] == action_history[0] and action_history[0] < 11):
                    if controller_states_different(player, last_recorded_player):
                        inp = generate_input(gamestate=gamestate, player_port=player_port, opponent_port=opponent_port)
                        if inp is None:
                            break

                        out = np.zeros(21)
                        out[action] = 1

                        X.append(inp)
                        Y.append(out)
                    last_recorded_action = action
                    last_recorded_player = player

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
    # model = Sequential([
    #     Dense(32, activation='tanh', input_shape=(len(X[0]),)),
    #     Dense(32, activation='tanh'),
    #     Dense(len(Y[0]), activation='tanh'),
    # ])

    opt = keras.optimizers.Adam(
        learning_rate=lr,
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

    # folder = 'models'
    pickle_file_path = f'{folder}/{player_character.name}_v_{opponent_character.name}_on_{stage.name}.pkl'

    if not os.path.isdir(folder):
        os.mkdir(f'{folder}/')

    with open(pickle_file_path, 'wb') as file:
        pickle.dump(model, file)


if __name__ == '__main__':
    # print(f'{player_character.name} vs. {opponent_character.name} on {stage.name}')
    # f = open('replays.json', 'r')
    # j = json.load(f)
    # replay_paths = j[f'{player_character.name}_{opponent_character.name}'][stage.name]
    # #
    # create_model(replay_paths=replay_paths, player_character=player_character,
    #              opponent_character=opponent_character, stage=stage, folder='models', lr = 6e-3)

    f = open('replays2.json', 'r')
    j = json.load(f)
    characters = [melee.Character.MARTH, melee.Character.FALCO, melee.Character.CPTFALCON]
    for c1 in characters:
        for c2 in characters:
            if c1 != c2:
                for s in [melee.Stage.BATTLEFIELD, melee.Stage.FINAL_DESTINATION]:
                    print(f'{c1.name} vs. {c2.name} on {s.name}')

                    replay_paths = j[f'{c1.name}_{c2.name}'][s.name]

                    create_model(replay_paths=replay_paths, player_character=c1,
                                 opponent_character=c2, stage=s, folder='models2', lr=2e-4)
