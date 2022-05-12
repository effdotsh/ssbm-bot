import time

import melee
import GameManager
import Args
from collections import deque
# # path = '/home/human/Slippi/Game_20220413T001024.slp'
# path = "/media/human/Data/replays/Gang-Steals/24/Game_20190309T111522.slp"
# replay = melee.Console(is_dolphin=False,
#                        allow_old_version=True,
#                        path=path, blocking_input=True, online_delay=0)
# replay.connect()
# print(replay.version)
# print(replay.slp_version)
# gamestate = replay.step()

args = Args.get_args()
game = GameManager.Game(args)
# print('-----')
# print(game.console.version)
# print(game.console.slp_version)
#
# ports = list(gamestate.players.keys())
# player1: melee.PlayerState = gamestate.players.get(ports[0])
# player2: melee.PlayerState = gamestate.players.get(ports[1])
#
# game.enterMatch(player_character=player1.character, opponant_character=player2.character)
# for i in range(140):
#     game.console.step()
#
# num=3
# queue = deque(maxlen=num)
gamestate = game.get_gamestate()

while gamestate is not None:
    gamestate = game.get_gamestate()
    # player1: melee.PlayerState = gamestate.players.get(ports[0])
    # player2: melee.PlayerState = gamestate.players.get(ports[1])
    #
    # controllers = [game.controller, game.controller_opponent]
    #
    # for e, p in enumerate([player1, player2]):
    #     p: melee.PlayerState = p
    #     c: melee.ControllerState = p.controller_state
    #     if e == 1:
    #         queue.append(c)
    #         c = queue[0]
    #     if len(queue) >= num:
    #         controller = controllers[e]
    #         controller.release_all()
    #
    #         controller.tilt_analog(melee.Button.BUTTON_MAIN, c.main_stick[0], c.main_stick[1])
    #
    #         controller.tilt_analog(melee.Button.BUTTON_C, c.c_stick[0], c.c_stick[1])
    #
    #         controller.press_shoulder(melee.Button.BUTTON_L, c.l_shoulder)
    #         controller.press_shoulder(melee.Button.BUTTON_R, c.r_shoulder)
    #
    #         buttons = [melee.Button.BUTTON_A, melee.Button.BUTTON_B, melee.Button.BUTTON_X, melee.Button.BUTTON_Y,
    #                    melee.Button.BUTTON_Z, melee.Button.BUTTON_L, melee.Button.BUTTON_R, melee.Button.BUTTON_D_UP,
    #                    melee.Button.BUTTON_D_DOWN, melee.Button.BUTTON_D_LEFT, melee.Button.BUTTON_D_RIGHT]
    #         # buttons = [melee.Button.BUTTON_A]
    #
    #         for b in buttons:
    #             if c.button.get(b):
    #                 controller.press_button(b)
    #                 if b == buttons[0]:
    #                     print(time.time())
    #         controller.flush()
    # game.console.step()
    # gamestate = replay.step()
