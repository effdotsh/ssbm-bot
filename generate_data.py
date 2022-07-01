#!/usr/bin/python3
import copy
from collections import deque

import melee

import os
import json

import melee as melee
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

def load_data(replay_paths: str, player_character: melee.Character, opponent_character: melee.Character, ):
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
    return X, Y

if __name__ == '__main__':

    # Mass Generate
    f = open('replays2.json', 'r')
    j = json.load(f)
    characters = [melee.Character.FALCO, melee.Character.JIGGLYPUFF, melee.Character.MARTH, melee.Character.CPTFALCON, melee.Character.FOX]

    for c1 in characters:
        for c2 in characters:
            if c1 != c2:
                for s in [melee.Stage.BATTLEFIELD, melee.Stage.FINAL_DESTINATION]:
                    f = open('replays2.json', 'r')
                    j = json.load(f)
                    characters = [melee.Character.MARTH, melee.Character.FALCO, melee.Character.CPTFALCON]

                    replay_paths = j[f'{c1.name}_{c2.name}'][s.name]

                    X, Y = load_data(replay_paths, c1, c2)

                    data_file_path = f'Data/{c1.name}_{c2.name}_on_{s.name}_data.pkl'
                    with open(data_file_path, 'wb') as file:
                        pickle.dump({'X': X, 'Y': Y}, file)
