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

from Agent import Agent, Algorithm

args = Args.get_args()
start_time = time.time()
if __name__ == '__main__':
    character = melee.Character.FOX
    opponent = melee.Character.CPTFALCON if not args.compete else character

    if not os.path.isdir(f'{args.model_path}/{character.name}'):
        os.makedirs(f'{args.model_path}/{character.name}')

    game = GameManager.Game(args)
    game.enterMatch(cpu_level=args.cpu_level if not args.compete else 0, opponant_character=opponent,
                    player_character=character,
                    stage=melee.Stage.FINAL_DESTINATION)
    step = args.load_from

    agent1 = Agent(player_port=args.port, opponent_port=args.opponent, game=game, algorithm=Algorithm.PPO,
                   use_wandb=args.wandb)

    while True:  # Training loop
        gamestate = game.get_gamestate()
        agent1.run_frame(gamestate)

        step += 1
