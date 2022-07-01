import keras
import melee

from DataHandler import generate_input, generate_output, decode_from_model
import numpy as np

import MovesList


class Bot:
    def __init__(self, model, controller: melee.Controller, opponent_controller: melee.Controller):
        self.opponent_controller = opponent_controller
        self.drop_every = 5
        self.model: keras.Model = model
        self.controller = controller
        self.frame_counter = 0

        self.delay = 0
        self.pause_delay = 0
        self.firefoxing = False

    def validate_action(self, action, gamestate: melee.GameState, port: int):
        # global smash_last
        player: melee.PlayerState = gamestate.players.get(port)
        edge = melee.stages.EDGE_POSITION.get(gamestate.stage)
        vel_y = player.speed_y_self + player.speed_y_attack
        vel_x = player.speed_x_attack + player.speed_air_x_self + player.speed_ground_x_self

        x = np.sign(player.position.x)

        if player.action in MovesList.special_fall_list:
            print('special falling')
            return [[0, 0, 0, 0, 0], -x, 0, 0, 0]

        if player.character == melee.enums.Character.MARTH:
            if player.jumps_left == 0 and (player.position.y < 0 or abs(player.position.x) - edge > 0):
                print('marth autorecover')
                if player.y < -50 or abs(player.position.x) - edge < 20:
                    return [[0, 1, 0, 0, 0], -0.6 * x, 0.85, 0, 0]
                facing = 1 if player.facing else -1
                if facing == x:
                    if vel_x > 0 and x > 0 or vel_x < 0 and x < 0:
                        return [[0, 1, 0, 0, 0], -x, 0, 0, 0]
                        print("side attack")
                    else:
                        return [[0, 0, 0, 0, 0], -x, 0, 0, 0]

                return [[0, 1, 0, 0, 0], -0.6 * x, 0.85, 0, 0]

            if player.jumps_left > 0 and abs(player.position.x) > edge:
                if vel_y < 0:
                    print('mario jumpman mario')
                    return [[1, 0, 0, 0, 0], 0, 0, 0, 0]
                else:
                    return [[0, 0, 0, 0, 0], -x, 0, 0, 0]

        if player.character in [melee.Character.FOX, melee.Character.FALCO]:
            if player.y < -10:
                print('firefoxing')
                if player.action in MovesList.firefoxing:
                    self.firefoxing = True
                if not self.firefoxing:
                    print(player.action)
                    return [[0, 1, 0, 0, 0], 0, 1, 0, 0]
                else:
                    return [[0, 0, 0, 0, 0], -x * 0.71, 0.71, 0, 0]
            else:
                self.firefoxing = False

        return action

    def act(self, gamestate: melee.GameState):
        if self.delay > 0:
            self.delay -= 1
            return
        if self.pause_delay > 0:
            self.pause_delay -= 1
            self.controller.release_all()
            return
        self.controller.release_all()

        player: melee.PlayerState = gamestate.players.get(self.controller.port)
        opponent: melee.PlayerState = gamestate.players.get(self.opponent_controller.port)

        if opponent.action in MovesList.dead_list and player.on_ground:
            return

        self.frame_counter += 1

        inp = generate_input(gamestate, self.controller.port, self.opponent_controller.port)
        a = self.model.predict(np.array([inp]), verbose=0, use_multiprocessing=True)

        a, action = decode_from_model(a, player)

        action = self.validate_action(action, gamestate, self.controller.port)
        b = melee.enums.Button

        print(action)
        button_used = False
        for i in range(len(MovesList.buttons)):
            if action[0][i] == 1:
                button_used = True
                self.controller.press_button(MovesList.buttons[i][0])
            else:
                self.controller.release_button(MovesList.buttons[i][0])

        if action[0][0] == 1:  # jump
            self.controller.tilt_analog_unit(melee.Button.BUTTON_MAIN, 0, 0)
            self.controller.tilt_analog_unit(melee.Button.BUTTON_C, 0, 0)
        else:
            self.controller.tilt_analog_unit(melee.Button.BUTTON_MAIN, action[-4], action[-3])
            self.controller.tilt_analog_unit(melee.Button.BUTTON_C, action[-2], action[-1])



        if a in [11, 12] and player.character in [melee.Character.FALCO, melee.Character.FOX]:
            self.delay += 15



        if button_used:
            self.pause_delay += 3



        self.controller.flush()

        # if self.frame_counter >= self.drop_every:
        #     self.controller.release_all()
        #     self.frame_counter = 0
        #     self.delay += 1
