import time

import gym
import random
from gym import spaces
import math
import numpy as np
import gameManager
import melee

import utils
import copy

class FoxEnv(gym.Env):
    def __init__(self, player_port, opponent_port, game: gameManager.Game):
        self.num_actions = 18

        self.stage = melee.Stage.BATTLEFIELD

        self.game = game

        self.controller: melee.Controller = self.game.getController(player_port)
        self.player_port = player_port
        self.opponent_port = opponent_port
        # controller_opponent = game.getController(args.opponent)

        super(FoxEnv, self).__init__()

        self.gamestate: melee.GameState = self.game.console.step()
        self.old_gamestate = self.game.console.step()

        nun_inputs = self.get_observation(self.gamestate).shape[0]
        self.observation_space = spaces.Box(shape=np.array([nun_inputs]), dtype=np.float, low=-1, high=1)
        self.action_space = spaces.Discrete(self.num_actions)

        self.rewards = []

        self.move_x = 0

        self.kills = 0
        self.deaths = 0

        self.killed_last = 1 # 1 if no 0 if yes

    def step(self, action: int):
        self.kills = 0
        self.deaths = 0
        self.act(action)
        self.old_gamestate = self.gamestate
        self.gamestate = self.game.console.step()

        obs = self.get_observation(self.gamestate)


        r = self.calculate_reward()

        # print(f'{action} - {r}')
        print(f'{action} {r}')
        return [obs, r, False, {}]

    def get_observation(self, gamestate):
        player: melee.PlayerState = gamestate.players.get(self.player_port)
        opponent: melee.PlayerState = gamestate.players.get(self.opponent_port)

        opponent_action = opponent.action
        self_action = player.action

        opponent_attacking = 1 if opponent_action in utils.attacking_list else 0
        self_attacking = 1 if self_action in utils.attacking_list else 0

        player_facing = 1 if player.facing else -1
        opponent_facing = 1 if player.facing else -1

        opponent_vel_x = utils.clamp((opponent.speed_air_x_self + opponent.speed_ground_x_self + opponent.speed_x_attack)/1000, 0, 1)
        opponent_vel_y = utils.clamp((opponent.speed_y_self + opponent.speed_y_attack)/1000, 0, 1)

        self_vel_x = utils.clamp((player.speed_air_x_self + player.speed_ground_x_self + player.speed_x_attack)/1000, 0, 1)
        self_vel_y = utils.clamp((player.speed_y_self + player.speed_y_attack)/1000, 0, 1)

        blastzones = melee.BLASTZONES.get(self.stage)


        obs = np.array([player.x/blastzones[1], player.y/blastzones[2], opponent.x/blastzones[1], opponent.y/blastzones[2], player_facing, opponent_attacking, opponent_vel_x, opponent_vel_y, self_vel_x, self_vel_y])
        # print(player.action)

        return obs

    def calculate_reward(self):
        old_gamestate = self.old_gamestate
        new_gamestate = self.gamestate

        new_player: melee.PlayerState = new_gamestate.players.get(self.player_port)
        new_opponent: melee.PlayerState = new_gamestate.players.get(self.opponent_port)

        old_player: melee.PlayerState = old_gamestate.players.get(self.player_port)
        old_opponent: melee.PlayerState = old_gamestate.players.get(self.opponent_port)

        distance = math.dist((new_player.x, new_player.y), (new_opponent.x, new_opponent.y))

        damage_dealt = max(0, new_opponent.percent - old_opponent.percent)
        damage_recieved = max(0, new_player.percent - old_player.percent)


        blast_thresh = 30
        blastzones = melee.BLASTZONES.get(self.stage)
        # deaths = 1 if new_player.percent < old_player.percent or math.fabs(new_player.x) > blastzones[1] - blast_thresh or new_player.y > blastzones[2] - blast_thresh or new_player.y < blastzones[3] + blast_thresh else 0
        # kills = 1 if new_opponent.percent < old_opponent.percent or math.fabs(new_opponent.x) > blastzones[1] - blast_thresh or new_opponent.y > blastzones[2] - blast_thresh or new_opponent.y < blastzones[3] + blast_thresh else 0

        # print(deaths)
        k = self.kills * self.killed_last
        self.killed_last = -self.kills + 1

        reward = -distance/1000 + (damage_dealt - damage_recieved) * 10 + k * 2000 - self.deaths * 5000
        return reward

    def reset(self):
        self.old_gamestate = self.game.console.step()
        return self.get_observation(self.old_gamestate)

    def act(self, action: int):
        # print(action)
        def wait_for_actionable():
            blast_thresh = 30
            blastzones = melee.BLASTZONES.get(self.stage)

            player: melee.PlayerState = self.gamestate.players.get(self.player_port)
            old_player = player
            while player.action in utils.dead_list or player.action in utils.attacking_list:
                game_state = self.game.console.step()
                player: melee.PlayerState = game_state.players.get(self.player_port)
                opponent: melee.PlayerState = game_state.players.get(self.opponent_port)
                if player.action == melee.Action.ON_HALO_DESCENT and self.deaths == 0:
                    self.deaths = 1
                if opponent.action == melee.Action.ON_HALO_DESCENT or opponent.percent < opponent.percent or math.fabs(opponent.x) > blastzones[1] - blast_thresh or opponent.y > blastzones[2] - blast_thresh or opponent.y < blastzones[3] + blast_thresh:
                    self.kills = 1

                if old_player.action == melee.Action.ON_HALO_DESCENT and player.action == melee.Action.ON_HALO_WAIT:
                    break
                old_player = player
        def frame_delay(num_frames):
            for i in range(num_frames):
                self.game.console.step()
            # fps = 60
            # time.sleep(num_frames/fps)

        def flick_button(button, num_frames):
            self.controller.press_button(button)
            self.controller.flush()
            frame_delay(num_frames)
            self.controller.release_button(button)
            self.controller.flush()


        def flick_axis(button, x, y, num_frames):
            self.controller.tilt_analog_unit(button, x, y)
            frame_delay(num_frames)
            self.controller.tilt_analog_unit(button, 0, 0)

        def button_axis(button, axis, x, y):
            self.controller.tilt_analog_unit(axis, x, y)
            frame_delay(1)

            self.controller.press_button(button)
            self.controller.flush()

            wait_for_actionable()

            self.controller.release_button(button)
            self.controller.tilt_analog_unit(axis, 0, 0)

            self.controller.flush()

        move_stick = melee.Button.BUTTON_MAIN

        if action == 0:  # Move Left
            self.move_x = -1
        elif action == 1:  # Move Right
            self.move_x = 1
        elif action == 2:  # Jump
            flick_button(melee.Button.BUTTON_Y, num_frames=5)

        elif action == 3:  # Drop
            flick_axis(move_stick, 0, -1, num_frames=10)


        elif action == 4:  # Left Smash
            flick_axis(melee.Button.BUTTON_C, -1, 0, num_frames=39)
        elif action == 5:  # Right Smash
            flick_axis(melee.Button.BUTTON_C, 1, 0, num_frames=39)
        elif action == 6:  # Down Smash
            flick_axis(melee.Button.BUTTON_C, 0, -1, num_frames=49)
        elif action == 7:  # Up Smash
            flick_axis(melee.Button.BUTTON_C, 0, 1, num_frames=41)

        elif action == 8:  # Right B
            button_axis(melee.Button.BUTTON_B, melee.Button.BUTTON_MAIN, 1, 0)
        elif action == 9:  # Left B
            button_axis(melee.Button.BUTTON_B, melee.Button.BUTTON_MAIN, -1, 0)
        elif action == 10:  # Down B
            button_axis(melee.Button.BUTTON_B, melee.Button.BUTTON_MAIN, 0, -1)
            frame_delay(4)
            flick_button(melee.Button.BUTTON_Y, num_frames=1)
        ##Up B's
        elif action == 11:  # Up B Up
            button_axis(melee.Button.BUTTON_B, melee.Button.BUTTON_MAIN, 0, 1)
        elif action == 12:  # Up B Down
            button_axis(melee.Button.BUTTON_B, melee.Button.BUTTON_MAIN, 0, 1)
            button_axis(melee.Button.BUTTON_B, melee.Button.BUTTON_MAIN, 0, -1)
        elif action == 13:  # Up B Right
            button_axis(melee.Button.BUTTON_B, melee.Button.BUTTON_MAIN, 0, 1)
            button_axis(melee.Button.BUTTON_B, melee.Button.BUTTON_MAIN, 1, 0)
        elif action == 14:  # Up B Left
            button_axis(melee.Button.BUTTON_B, melee.Button.BUTTON_MAIN, 0, 1)
            button_axis(melee.Button.BUTTON_B, melee.Button.BUTTON_MAIN, -1, 0)
        elif action == 15:  # Up Recover Right Side
            side = melee.EDGE_POSITION.get(self.stage)
            player: melee.PlayerState = self.gamestate.players.get(self.player_port)

            x = side - player.x
            y = 10 - player.y

            button_axis(melee.Button.BUTTON_B, melee.Button.BUTTON_MAIN, 0, 1)
            button_axis(melee.Button.BUTTON_B, melee.Button.BUTTON_MAIN, x, y)

        elif action == 16:  # Up Recover Left Side
            side = melee.EDGE_POSITION.get(self.stage)
            player: melee.PlayerState = self.gamestate.players.get(self.player_port)

            x = -side - player.x
            y = 10 - player.y

            button_axis(melee.Button.BUTTON_B, melee.Button.BUTTON_MAIN, 0, 1)
            button_axis(melee.Button.BUTTON_B, melee.Button.BUTTON_MAIN, x, y)

        elif action == 17:
            self.move_x = 0

        if action not in [11, 12, 13, 14, 15, 3]:
            self.controller.tilt_analog_unit(move_stick, self.move_x, 0)

        if action in [0, 1, 17]:
            flick_axis(melee.Button.BUTTON_MAIN, self.move_x, 0, num_frames=3)
