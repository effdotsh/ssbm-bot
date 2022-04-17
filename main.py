#!/usr/bin/python3

import Args
import melee

import os
import json
from DataHandler import create_model

args = Args.get_args()

if __name__ == '__main__':
    player_character = melee.Character.CPTFALCON
    opponent_character = melee.Character.CPTFALCON
    stage = melee.Stage.FINAL_DESTINATION

    f = open('replays.json', 'r')
    j = json.load(f)

    replay_paths = j[f'{player_character.name}_{opponent_character.name}']
    create_model(replay_paths=replay_paths, player_character=player_character,
                 opponent_character=opponent_character, stage=stage)
