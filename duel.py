import math
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


def validate_action(action_packed, maxes, gamestate: melee, player_port: int):
    # print(action_packed, maxes)
    move_x, move_y, c, button = action_packed
    idle = decode_from_number(120, maxes)
    player: melee.PlayerState = gamestate.players.get(player_port)
    move_x -= 1
    move_y -= 1

    edge: float = melee.EDGE_POSITION.get(gamestate.stage)
    edge_buffer = 20

    if player.action == melee.Action.TUMBLING:
        print('Exiting Tumble')
        return [1, 1, 0, 3]

    if player.character in [melee.Character.FOX, melee.Character.FALCO]:
        a = melee.Action
        if 'SWORD_DANCE' in player.action.name and abs(player.position.x) > edge - edge_buffer:
            print('Aiming at ledge')
            if player.position.x > 0:
                angle = math.atan2(40-player.position.y, edge-10 - player.position.x)
            else:
                angle = math.atan2(40-player.position.y, -edge+10 - player.position.x)

            return [(math.cos(angle)+1)/2, (math.sin(angle)+1)/2, 0, 0]

        if player.action in [a.NEUTRAL_B_ATTACKING, a.NEUTRAL_B_ATTACKING_AIR, a.NEUTRAL_B_CHARGING,
                             a.NEUTRAL_B_CHARGING_AIR, a.NEUTRAL_B_FULL_CHARGE, a.NEUTRAL_B_FULL_CHARGE_AIR]:
            if move_x != 0 or move_y != 0:
                print('Stopping laser lockout')
                return idle

        if button == 2 and c == 0 and (
                move_x == -1 and player.position.x < -edge + edge_buffer or move_x == 1 and player.position.x > edge - edge_buffer):
            print("Stopping dash SD")
            return idle

        if button == 0 and player.action in [a.FALLING] and (abs(player.position.x) > edge or player.position.y < -10):
            print('forcing recovery', player.position.x, player.position.y, edge)
            return [1, 2, 0, 2]
    if player.character == melee.Character.JIGGLYPUFF:
        # Prevent accidental rollout
        if button == 2 and move_x == 0 and move_y == 0:
            return idle

    # Prevent running off edge
    if button == 0 and c == 0 and move_x == -1 and player.position.x < -edge + edge_buffer:
        print("Stopping running SD")
        return [2, 1, 0, 0]

    if button == 0 and c == 0 and move_x == 1 and player.position.x > edge - edge_buffer:
        print("Stopping running SD")
        return [0, 1, 0, 0]

    return action_packed


if __name__ == '__main__':
    character = melee.Character.FOX
    opponent = melee.Character.MARTH if not args.compete else character
    stage = melee.Stage.FINAL_DESTINATION
    print(f'{character.name} vs. {opponent.name} on {stage.name}')

    tree, map = DataHandler.load_model(player_character=character, opponent_character=opponent, stage=stage)

    game = GameManager.Game(args)
    game.enterMatch(cpu_level=9, opponant_character=opponent,
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

        action_packed = DataHandler.predict(tree=tree, map=map, gamestate=gamestate, player_port=game.controller.port,
                                            opponent_port=game.controller_opponent.port, maxes=maxes)
        action_packed= validate_action(action_packed, maxes, gamestate, game.controller.port)
        move_x, move_y, c, button = action_packed

        # print(gamestate.players.get(game.controller.port).action)
        print(move_x - 1, move_y - 1, c, button)

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

        last_action = action_packed
        gamestate = game.get_gamestate()

        for b in DataHandler.buttons:
            game.controller.release_button(b[0])
        game.controller.tilt_analog(melee.Button.BUTTON_C, 0.5, 0.5)
        if button != 0:
            game.controller.tilt_analog(melee.Button.BUTTON_MAIN, move_x / 2, move_y / 2)

        # game.controller.release_all()
