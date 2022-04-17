#!/usr/bin/python3

import Args
import melee

import os
import json
from DataHandler import create_model

args = Args.get_args()

if __name__ == '__main__':
    player_character = melee.Character.MARTH
    opponent_character = melee.Character.MARTH
    stage = melee.Stage.FINAL_DESTINATION

    print(f'{player_character.name} vs. {opponent_character.name} on {stage.name}')

    f = open('replays.json', 'r')
    j = json.load(f)

    replay_paths = j[f'{player_character.name}_{opponent_character.name}'][stage.name]

    create_model(replay_paths=replay_paths, player_character=player_character,
                 opponent_character=opponent_character, stage=stage)
