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
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense
import pickle

import Args
from DataHandler import get_ports, controller_states_different, generate_input, generate_output
import MovesList

args = Args.get_args()


def create_model(X: np.ndarray, Y: np.ndarray, player_character: melee.Character, opponent_character: melee.Character,
                 stage: melee.Stage,
                 folder: str, lr: float):
    print(len(X), len(Y))
    print(len(X[0]), len(Y[0]))

    # train
    model = Sequential([
        Dense(128, activation='tanh', input_shape=(len(X[0]),)),
        Dense(128, activation='tanh'),
        # Dense(128, activation='tanh'),
        Dense(len(Y[0]), activation='tanh'),
    ])
    # model = Sequential([
    #     Dense(32, activation='tanh', input_shape=(len(X[0]),)),
    #     Dense(32, activation='tanh'),
    #     Dense(len(Y[0]), activation='tanh'),
    # ])

    # opt = keras.optimizers.Adam(
    #     learning_rate=lr,
    #     name="Adam",
    # )
    # opt = optimizers.Adagrad(
    #     learning_rate=lr,
    #     name="Adagrad",
    # )
    # opt = optimizers.Adadelta(
    #     learning_rate=lr,
    #     name="Adelta",
    # )

    opt = optimizers.RMSprop(
        learning_rate=lr,
        name="RMSprop",
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
    player_character = melee.Character.MARTH
    opponent_character = melee.Character.CPTFALCON
    stage = melee.Stage.FINAL_DESTINATION
    lr = 1e-5

    raw = open(f'Data/{player_character.name}_{opponent_character.name}_on_{stage.name}_data.pkl', 'rb')
    data = pickle.load(raw)
    X = data['X']
    Y = data['Y']
    create_model(X, Y, player_character=player_character,
                 opponent_character=opponent_character, stage=stage, folder='models2', lr=lr)
