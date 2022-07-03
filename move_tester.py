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

    action_history = deque(maxlen=3)

    last_recorded_player = player
    last_recorded_action = -1
    while True:
        gamestate = game.get_gamestate()
        if gamestate is None:
            continue
        player: melee.PlayerState = gamestate.players.get(game.controller.port)
        if player is None or last_recorded_player is None:
            continue

        out = generate_output(player)
        action_history.append(out)

        if out != last_recorded_action and out != -1:
            if action_history[-1] < 11 and action_history[0] >=11:
                pass
            elif action_history[-1] >= 11 and action_history[0] < 11 or (action_history[-1] == action_history[0] and action_history[0] < 11):
                if controller_states_different(player, last_recorded_player):
                    print(out)
                last_recorded_action=out
                last_recorded_player = player

        elif out == -1:
            last_recorded_action=out
            last_recorded_player = player

