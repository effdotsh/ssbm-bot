#!/usr/bin/python3

import Args
import melee

import os

from ReplayManager import filter_replays

args = Args.get_args()
if __name__ == '__main__':
    player_character = melee.Character.FALCO
    opponent_character = melee.Character.FOX


    replay_folder = '/home/human/Documents/replays_lite/'

    replay_paths = []
    for root, dirs, files in os.walk(replay_folder):
        for name in files:
            replay_paths.append(os.path.join(root, name))

    print(len(replay_paths))
    replay_paths = filter_replays(replay_paths, opponent_character=opponent_character, player_character=player_character, win_only=False)
    print(len(replay_paths))


