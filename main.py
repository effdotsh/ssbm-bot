#!/usr/bin/python3

import Args
import melee

import os

from ReplayManager import filter_replays
from trainer import train

args = Args.get_args()

if __name__ == '__main__':
    player_character = melee.Character.MARTH
    opponent_character = melee.Character.CPTFALCON
    stage = melee.Stage.BATTLEFIELD

    replay_folder = '/media/human/Data/replays'

    replay_paths = []
    for root, dirs, files in os.walk(replay_folder):
        for name in files:
            replay_paths.append(os.path.join(root, name))

    replay_paths = filter_replays(replay_paths, opponent_character=opponent_character,
                                  player_character=player_character, win_only=False, stage=stage)

    train(replay_paths=replay_paths, player_character=player_character,
          opponent_character=opponent_character)
