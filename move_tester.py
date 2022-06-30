import time

import Args
import GameManager
import melee
import platform

import os
import DataHandler
import numpy as np

from DataHandler import controller_states_different, generate_input, generate_output, decode_from_model

from collections import deque

args = Args.get_args()

if __name__ == '__main__':
    character = melee.Character.MARTH
    opponent = melee.Character.CPTFALCON if not args.compete else character
    stage = melee.Stage.BATTLEFIELD
    print(f'{character.name} vs. {opponent.name} on {stage.name}')

    game = GameManager.Game(args)
    game.enterMatch(cpu_level=0, opponant_character=opponent,
                    player_character=character,
                    stage=stage, rules=False)

    gamestate = game.get_gamestate()
    player: melee.PlayerState = gamestate.players.get(game.controller.port)
    last_player = player

    history = deque(maxlen=20)

    while True:
        gamestate = game.get_gamestate()
        if gamestate is None:
            continue
        player: melee.PlayerState = gamestate.players.get(game.controller.port)
        # print(player.on_ground)
        if player is None or last_player is None:
            continue

        out = generate_output(player)
        # history.append(out)
        print(out)
        a = np.zeros(21)
        a[out] = 1
        print(decode_from_model([a], player))
        # if controller_states_different(player, last_player):
        #     # print(time.time())
        #     if not(7 <= history[0] < 10 and history[-1] >= 10):
        #         print(history[0])
        # last_player = player


        # print(player.controller_state)
        # print(last_state)
        # inp = generate_input(gamestate=gamestate, player_port=game.controller.port, opponent_port=game.controller.port)
        # print(inp)

