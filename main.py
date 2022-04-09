#!/usr/bin/python3
import argparse
import copy
import time

import torch

import Args
import GameManager
import melee
import platform

import os


args = Args.get_args()
start_time = time.time()
if __name__ == '__main__':
    replay_path = '/home/human/Slippi/Game_20220409T153244.slp'
    console = melee.Console(is_dolphin=False,
                            allow_old_version=True,
                            path=replay_path)
    console.connect()



    while True:  # Training loop
        gamestate: melee.GameState = console.step()
        p1: melee.PlayerState = gamestate.players.get(1)
        print(p1.controller_state)
        # print(p1.nickName)