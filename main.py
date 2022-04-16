#!/usr/bin/python3

import Args
import melee

import os

from ReplayManager import filter_replays
from trainer import train

args = Args.get_args()

if __name__ == '__main__':
    player_character = melee.Character.JIGGLYPUFF
    opponent_character = melee.Character.CPTFALCON
    stage = melee.Stage.FINAL_DESTINATION

    print(f'{player_character.name} vs. {opponent_character.name} on {stage.name}')
    # replay_folder = '/media/human/Data/replays'
    replay_folder = '/media/human/Data/melee_public_slp_dataset_v2'
    replay_paths = []
    for root, dirs, files in os.walk(replay_folder):
        for name in files:
            replay_paths.append(os.path.join(root, name))

    replay_paths = filter_replays(replay_paths, opponent_character=opponent_character,
                                  player_character=player_character, win_only=False, stage=stage)

    train(replay_paths=replay_paths, player_character=player_character,
          opponent_character=opponent_character, stage=stage)
