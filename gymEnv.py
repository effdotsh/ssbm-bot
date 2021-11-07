import gym
import random
from gym import spaces
import math
import numpy as np
import gameManager
import melee

import utils

frames_per_step = 100
num_actions = 10


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
        self.gamestate = self.game.console.step()

        obs = self.get_observation(self.gamestate)

        r = 0
        new_step = False
        if (self.frame == 0):
            r = self.calculate_reward()
            new_step = True
            self.rewards.append(r)
        return [obs, r, new_step, {}]

    def get_observation(self, gamestate):
        player: melee.PlayerState = gamestate.players.get(self.player_port)
        opponent: melee.PlayerState = gamestate.players.get(self.opponent_port)

        opponent_action = opponent.action
        opponent_attacking = 1 if opponent_action in utils.attacking_list else 0

        player_facing = 1 if player.facing else -1
        opponent_facing = 1 if player.facing else -1

        opponent_vel_x = opponent.speed_air_x_self + opponent.speed_ground_x_self + opponent.speed_x_attack
        opponent_vel_y = opponent.speed_y_self + opponent.speed_y_attack


        return np.array([player.x, player.y, opponent.x, opponent.y, player_facing, opponent_vel_x, opponent_vel_y, opponent_attacking * opponent_facing])

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

        deaths = 1 if new_player.percent < old_player.percent else 0

        kills = 1 if new_opponent.percent < old_opponent.percent else 0

        # try:
        #     print(f"Stocks: {old_player.}")
        # except:
        #     pass
        return -distance + (damage_dealt - damage_recieved) * 10 + kills * 1000 - deaths * 2000

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

        # print(action)
        move_stick = melee.Button.BUTTON_MAIN
        if action == 0:  # Move Left
            self.controller.tilt_analog_unit(move_stick, -1, 0)
        elif action == 1:  # Move Right
            self.controller.tilt_analog_unit(move_stick, 1, 0)
        elif action == 2:  # Attack Right
            flick_axis(melee.Button.BUTTON_C, 1, 0)
        elif action == 3:  # Attack Left
            flick_axis(melee.Button.BUTTON_C, -1, 0)
        elif action == 4:  # Attack Down
            flick_axis(melee.Button.BUTTON_C, 0, -1)
        elif action == 5:  # Jump
            flick_button(melee.Button.BUTTON_Y)
        elif action == 6:  # Neutral B
            flick_button(melee.Button.BUTTON_B)
        elif action == 7:  # Right B
            button_axis(melee.Button.BUTTON_B, melee.Button.BUTTON_MAIN, 1, 0)
        elif action == 8:  # Left B
            button_axis(melee.Button.BUTTON_B, melee.Button.BUTTON_MAIN, -1, 0)
        elif action == 9:  # Up B
            button_axis(melee.Button.BUTTON_B, melee.Button.BUTTON_MAIN, 0, 1)
