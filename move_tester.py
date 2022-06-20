import time

import Args
import GameManager
import melee
import platform

import os
import DataHandler
import numpy as np

from DataHandler import controller_states_different, generate_input, generate_output

args = Args.get_args()

if __name__ == '__main__':
    character = melee.Character.FOX
    opponent = melee.Character.MARTH if not args.compete else character
    stage = melee.Stage.FINAL_DESTINATION
    print(f'{character.name} vs. {opponent.name} on {stage.name}')

    game = GameManager.Game(args)
    game.enterMatch(cpu_level=0, opponant_character=opponent,
                    player_character=character,
                    stage=stage, rules=False)

    gamestate = game.get_gamestate()
    player: melee.PlayerState = gamestate.players.get(game.controller.port)
    last_state = player.controller_state
    while True:
        gamestate = game.get_gamestate()
        player: melee.PlayerState = gamestate.players.get(game.controller.port)

        # if controller_states_different(player.controller_state, last_state):
        #     print(time.time())
        last_state = player.controller_state

        # print(last_state)
        # inp = generate_input(gamestate=gamestate, player_port=game.controller.port, opponent_port=game.controller_opponent.port)
        # print(inp)

        out = generate_output(player.controller_state)
        print(out)
