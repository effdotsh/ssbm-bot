import time

import Args
import GameManager
import melee
import platform

import os
import DataHandler
import numpy as np

from encoder import decode_from_number

args = Args.get_args()


def validate_action(action, maxes, gamestate: melee, player_port: int):
    move_x, move_y, c, button = decode_from_number(action, maxes)

    player: melee.PlayerState = gamestate.players.get(player_port)
    move_x -= 1
    move_y -= 1

    edge: float = melee.EDGE_POSITION.get(gamestate.stage)
    edge_buffer = 20
    # Prevent running off edge
    if button == 0 and c == 0 and (
            move_x == -1 and player.position.x < -edge + edge_buffer or move_x == 1 and player.position.x > edge - edge_buffer):
        print("Stopping running SD")
        return 120

    if player.character == melee.Character.FOX:
        a = melee.Action
        if player.action in [a.NEUTRAL_B_ATTACKING, a.NEUTRAL_B_ATTACKING_AIR, a.NEUTRAL_B_CHARGING,
                             a.NEUTRAL_B_CHARGING_AIR, a.NEUTRAL_B_FULL_CHARGE, a.NEUTRAL_B_FULL_CHARGE_AIR]:
            if move_x != 0 or move_y != 0:
                print('Stopping laser lockout')
                return 120

        if button == 2 and c == 0 and (
                move_x == -1 and player.position.x < -edge + edge_buffer or move_x == 1 and player.position.x > edge - edge_buffer):
            print("Stopping dash SD")
            return 120

    if player.character == melee.Character.JIGGLYPUFF:
        # Prevent accidental rollout
        if button == 2 and move_x == 0 and move_y == 0:
            return 120

    return action


if __name__ == '__main__':
    character = melee.Character.FOX
    opponent = melee.Character.MARTH if not args.compete else character
    stage = melee.Stage.FINAL_DESTINATION
    print(f'{character.name} vs. {opponent.name} on {stage.name}')

    tree, map = DataHandler.load_model(player_character=character, opponent_character=opponent, stage=stage)

    game = GameManager.Game(args)
    game.enterMatch(cpu_level=0, opponant_character=opponent,
                    player_character=character,
                    stage=stage, rules=False)

    num_buttons = len(DataHandler.buttons) + 1
    axis_size = 3
    num_c = 5
    maxes = [axis_size, axis_size, num_c, num_buttons]

    last_action = 120
    while True:
        gamestate = game.get_gamestate()
        # print('----------')

        action = DataHandler.predict(tree=tree, map=map, gamestate=gamestate, player_port=game.controller.port,
                                     opponent_port=game.controller_opponent.port, maxes=maxes)

        action = validate_action(action, maxes, gamestate, game.controller.port)
        move_x, move_y, c, button = decode_from_number(action, maxes)

        print(gamestate.players.get(game.controller.port).action)
        # print(move_x - 1, move_y - 1, c, button)

        # print(trainer.buttons)
        # print(gamestate.players.get(1).position.x)
        if button > 0:
            game.controller.press_button(DataHandler.buttons[button - 1][0])

        if c == 0:
            game.controller.tilt_analog(melee.Button.BUTTON_C, 0.5, 0.5)
        else:
            for i in range(3):
                if c == 1:
                    game.controller.tilt_analog(melee.Button.BUTTON_C, 0, 0.5)
                elif c == 2:
                    game.controller.tilt_analog(melee.Button.BUTTON_C, 1, 0.5)
                elif c == 3:
                    game.controller.tilt_analog(melee.Button.BUTTON_C, 0.5, 0)
                elif c == 4:
                    game.controller.tilt_analog(melee.Button.BUTTON_C, 0.5, 1)
        game.controller.tilt_analog(melee.Button.BUTTON_MAIN, move_x / 2, move_y / 2)

        if button > 0:
            for i in range(3):
                gamestate = game.get_gamestate()

        last_action = action
        gamestate = game.get_gamestate()

        for b in DataHandler.buttons:
            game.controller.release_button(b[0])
        # game.controller.tilt_analog(melee.Button.BUTTON_C, 0.5, 0.5)
        game.controller.tilt_analog(melee.Button.BUTTON_MAIN, move_x / 2, move_y / 2)

        # game.controller.release_all()
