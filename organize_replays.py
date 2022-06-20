import os
import json
import time

import melee
from tqdm import tqdm

if __name__ == '__main__':
    # replay_folder = '/home/human/Documents/training_data'
    replay_folder= '/home/human/Documents/slippi replays'
    replay_paths = []
    for root, dirs, files in os.walk(replay_folder):
        for name in files:
            replay_paths.append(os.path.join(root, name))



    if not os.path.exists('replays.json'):
        with open('replays.json', 'w') as file:
            json.dump({}, file, indent=4)

    f = open('replays.json', 'r')
    j = json.load(f)

    for path in tqdm(replay_paths):
        console = melee.Console(is_dolphin=False,
                                allow_old_version=True,
                                path=path)
        try:
            console.connect()
        except:
            console.stop()
            print('console failed to connect', path,  time.time())
            continue
        gamestate: melee.GameState = console.step()

        if gamestate is None:
            print('gamestate is none', path,  time.time())
            continue
        
        if gamestate.stage is None:
            print('stage is none', path, time.time())
            continue

        ports = list(gamestate.players.keys())

        if len(ports) != 2:
            print('not two ports ', path,  time.time())
            continue
        p1: melee.PlayerState = gamestate.players.get(ports[0])
        p2: melee.PlayerState = gamestate.players.get(ports[1])

        key = f'{p1.character.name}_{p2.character.name}'
        if key not in j:
            j[key] = {}
        if gamestate.stage.name not in j[key]:
            j[key][gamestate.stage.name] = []
        j[key][gamestate.stage.name].append(path)
        
        if p1.character.name != p2.character.name:
            key = f'{p2.character.name}_{p1.character.name}'
            if key not in j:
                j[key] = {}
            if gamestate.stage.name not in j[key]:
                j[key][gamestate.stage.name] = []
            j[key][gamestate.stage.name].append(path)
        console.stop()

    with open('replays.json', 'w') as file:
        json.dump(j, file, indent=2)