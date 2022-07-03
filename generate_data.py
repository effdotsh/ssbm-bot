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
    X_player = []
    Y_player = []

    X_opponent = []
    Y_oponnent = []
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
        opponent: melee.PlayerState = gamestate.players.get(player_port)

        player_action_history = deque(maxlen=3)
        opponent_action_history = deque(maxlen=3)

        last_recorded_player = player
        last_recorded_action_player = -1

        last_recorded_opponent = opponent
        last_recorded_action_opponent = -1
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

            # player
            action = generate_output(player)
            inp = generate_input(gamestate=gamestate, player_port=player_port, opponent_port=opponent_port)
            if action is None:
                break
            if action not in [21, -1]:
                player_action_history.append(action)
                if action != last_recorded_action_player and action != -1:
                    if player_action_history[-1] < 11 and player_action_history[0] >= 11:
                        pass
                    elif player_action_history[-1] >= 11 and player_action_history[0] < 11 or (
                            player_action_history[-1] == player_action_history[0] and player_action_history[0] < 11):
                        if controller_states_different(player, last_recorded_player):
                            if inp is None:
                                break

                            out = np.zeros(21)
                            out[action] = 1

                            X_player.append(inp)
                            Y_player.append(out)
                        last_recorded_action_player = action
                        last_recorded_player = player

            #opponent
            action_opponent = generate_output(player)
            inp = generate_input(gamestate=gamestate, player_port=player_port, opponent_port=opponent_port)
            if action_opponent is None:
                break
            if action_opponent not in [21, -1]:
                opponent_action_history.append(action_opponent)
                if action_opponent != last_recorded_action_opponent and action_opponent != -1:
                    if opponent_action_history[-1] < 11 and opponent_action_history[0] >= 11:
                        pass
                    elif opponent_action_history[-1] >= 11 and opponent_action_history[0] < 11 or (
                            opponent_action_history[-1] == opponent_action_history[0] and opponent_action_history[0] < 11):
                        if controller_states_different(opponent, last_recorded_opponent):
                            if inp is None:
                                break

                            out = np.zeros(21)
                            out[action_opponent] = 1

                            X_opponent.append(inp)
                            Y_oponnent.append(out)
                        last_recorded_action_opponent = action_opponent
                        last_recorded_opponent = player

    X_player = np.array(X_player)
    Y_player = np.array(Y_player)
    X_opponent = np.array(X_player)
    Y_oponnent = np.array(Y_player)
    return X_player, Y_player, X_opponent, Y_oponnent

def process_replays(replays: dict, c1: melee.Character, c2: melee.Character, s: melee.Stage):
    print(f'Data/{c1.name}_{c2.name}_on_{s.name}_data.pkl')
    print(f'Data/{c2.name}_{c1.name}_on_{s.name}_data.pkl')


    replay_paths = replays[f'{c1.name}_{c2.name}'][s.name]

    Xp, Yp, Xo, Yo = load_data(replay_paths, c1, c2)

    data_file_path = f'Data/{c1.name}_{c2.name}_on_{s.name}_data.pkl'
    with open(data_file_path, 'wb') as file:
        pickle.dump({'X': Xp, 'Y': Yp}, file)

    data_file_path = f'Data/{c2.name}_{c1.name}_on_{s.name}_data.pkl'
    with open(data_file_path, 'wb') as file:
        pickle.dump({'X': Xo, 'Y': Yo}, file)


if __name__ == '__main__':

    # Mass Generate
    f = open('replays2.json', 'r')
    replays = json.load(f)
    characters = [melee.Character.FALCO, melee.Character.JIGGLYPUFF, melee.Character.MARTH, melee.Character.CPTFALCON, melee.Character.FOX]

    for e, c1 in enumerate(characters):
        for c2 in characters[e+1:]:
            if c1 != c2:
                for s in [melee.Stage.FINAL_DESTINATION]:
                    process_replays(replays, c1, c2, s)