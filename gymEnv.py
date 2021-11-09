import gym
import random
from gym import spaces
import math
import numpy as np
import gameManager
import melee

import utils

frames_per_step = 100
num_actions = 13


class CharacterEnv(gym.Env):
    def __init__(self, args: gameManager.Args, player_port, opponent_port):

        # Setup Game
        self.game = gameManager.Game(args)
        self.controller = self.game.getController(args.port)
        self.player_port = player_port
        self.opponent_port = opponent_port
        # controller_opponent = game.getController(args.opponent)
        self.game.enterMatch(stage=melee.Stage.FINAL_DESTINATION)

        super(CharacterEnv, self).__init__()

        self.gamestate: melee.GameState = self.game.console.step()
        self.frame = 0
        self.old_gamestate = self.game.console.step()

        nun_inputs = self.get_observation(self.gamestate).shape[0]
        self.observation_space = spaces.Box(shape=np.array([nun_inputs]), dtype=np.float, low=-500, high=500)
        self.action_space = spaces.Discrete(num_actions)

        self.rewards = []

    def step(self, action: int):
        self.frame += 1
        self.frame %= frames_per_step

        self.act(action)
        self.old_gamestate = self.gamestate
        self.gamestate = self.game.console.step()

        obs = self.get_observation(self.gamestate)


        r = self.calculate_reward()
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

        opponent_vel_x = opponent.speed_air_x_self + opponent.speed_ground_x_self + opponent.speed_x_attack
        opponent_vel_y = opponent.speed_y_self + opponent.speed_y_attack

        obs = np.array([player.x, player.y, opponent.x, opponent.y, player_facing, opponent_vel_x, opponent_vel_y, opponent_attacking * opponent_facing, self_attacking, ])
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


        blast_thresh = 20
        blastzones = melee.BLASTZONES.get(melee.Stage.FINAL_DESTINATION)
        deaths = 1 if new_player.percent < old_player.percent or math.fabs(new_player.x) > blastzones[1] - blast_thresh or new_player.y > blastzones[2] - blast_thresh or new_player.y < blastzones[3] + blast_thresh  else 0
        kills = 1 if new_opponent.percent < old_opponent.percent or math.fabs(new_opponent.x) > blastzones[1] - blast_thresh or new_opponent.y > blastzones[2] - blast_thresh or new_opponent.y < blastzones[3] + blast_thresh  else 0
        # print(deaths)

        reward = -distance/5 + (damage_dealt - damage_recieved) * 10 + kills * 1000 - deaths * 5000
        print(reward)
        return reward

    def reset(self):
        self.old_gamestate = self.game.console.step()
        return self.get_observation(self.old_gamestate)

    def act(self, action: int):
        def flick_button(button):
            self.controller.press_button(button)
            gamestate = self.game.console.step()
            self.controller.release_button(button)

        def flick_axis(button, x, y):
            self.controller.tilt_analog_unit(button, x, y)
            gamestate = self.game.console.step()
            self.controller.tilt_analog_unit(button, 0, 0)

        def button_axis(button, axis, x, y):
            self.controller.tilt_analog_unit(axis, x, y)
            gamestate = self.game.console.step()
            self.controller.press_button(button)
            gamestate = self.game.console.step()
            self.controller.release_button(button)
            self.controller.tilt_analog_unit(axis, 0, 0)

        move_stick = melee.Button.BUTTON_MAIN
        if action == 0:  # Move Left
            self.controller.tilt_analog_unit(move_stick, -1, 0)
        elif action == 1:  # Move Right
            self.controller.tilt_analog_unit(move_stick, 1, 0)
        elif action == 2:  # Jump
            flick_button(melee.Button.BUTTON_Y)

        elif action == 3:  # Jab Left
            flick_axis(melee.Button.BUTTON_C, -1, 0)
        elif action == 4:  # Jab Right
            flick_axis(melee.Button.BUTTON_C, 1, 0)
        elif action == 5:  # Jab Down
            flick_axis(melee.Button.BUTTON_C, 0, -1)
        elif action == 6:  # Jab Up
            flick_axis(melee.Button.BUTTON_C, 0, 1)

        elif action == 7:  # Neutral B
            flick_button(melee.Button.BUTTON_B)
        elif action == 8:  # Right B
            button_axis(melee.Button.BUTTON_B, melee.Button.BUTTON_MAIN, 1, 0)
        elif action == 9:  # Left B
            button_axis(melee.Button.BUTTON_B, melee.Button.BUTTON_MAIN, -1, 0)
        elif action == 10:  # Up B
            button_axis(melee.Button.BUTTON_B, melee.Button.BUTTON_MAIN, 0, 1)
        elif action == 11:  # Down B
            button_axis(melee.Button.BUTTON_B, melee.Button.BUTTON_MAIN, 0, -1)

        elif action == 12:  # Sheild
            self.controller.press_button(melee.Button.BUTTON_L)
        else:
            self.controller.release_button(melee.Button.BUTTON_L)