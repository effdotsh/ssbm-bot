import math

import gym
import melee
from gym import spaces
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env

import numpy as np
import game

num_actions = 3  # walk left/right, jump, drop, attack


class Observation:
    player_x: int
    player_y: int
    opponent_x: int
    opponent_y: int
    action_frame: int

    def getValues(self):
        return np.array([self.player_x, self.player_y, self.opponent_x, self.opponent_y, self.action_frame])


class CharacterEnv(gym.Env):
    def __init__(self, game, state: melee.GameState, port, port_opponent):

        self.game = game
        super(CharacterEnv, self).__init__()
        self.action_frame = 1
        self.last_dist = 0
        self.port = port
        self.port_opponent = port_opponent

        self.gamestate = state

        base_observation = self.getObservation()

        self.observation_space = spaces.Box(shape=base_observation.shape, high=1, low=-1)
        self.action_space = spaces.Discrete(num_actions)

        self.controller: melee.Controller = game.getController(port)

    def setGameState(self, state: melee.GameState):
        self.gamestate = state

    def getObservation(self):
        gamestate = self.gamestate
        player_data: melee.PlayerState = gamestate.players.get(self.port)
        opponent_data: melee.PlayerState = gamestate.players.get(self.port_opponent)

        stage_blastzones: melee.Stage = melee.stages.BLASTZONES[gamestate.stage]
        stage_edge_x = stage_blastzones[1]
        stage_edge_y = stage_blastzones[3]

        ob = Observation()
        ob.player_x = player_data.x / stage_edge_x
        ob.player_y = player_data.y / stage_edge_y
        ob.opponent_x = opponent_data.x / stage_edge_x
        ob.opponent_y = opponent_data.y / stage_edge_y
        ob.action_frame = self.action_frame
        return ob.getValues()

    def step(self, action: int):
        self.action_frame *= -1

        if action == 0:  # Walk Left
            self.move(-1, 0)
        elif action == 1:  # Walk Right
            self.move(1, 0)
        # elif action == 2:  # Jump
        #     self.jump()
        # elif action == 3:  # Drop
        #     self.move(0, -1)
        else:  # Attack
            self.move(0, 0)
        self.flush_controls()

        self.gamestate: melee.GameState = self.game.getState()

        player: melee.PlayerState = self.gamestate.players.get(self.port)
        opponent: melee.PlayerState = self.gamestate.players.get(self.port_opponent)

        r = self.reward(player, opponent)
        return [self.getObservation(), r, False, {}]

    def reward(self, player: melee.PlayerState, opponent: melee.PlayerState):
        dist = math.dist((player.x, player.y), (opponent.x, opponent.y))

        r = (self.last_dist - dist) * (10-dist)
        print(r)
        return r

    def reset(self):
        pass

    ################
    ### Controls ###
    ################
    def flush_controls(self):
        if (self.action_frame == -1):
            self.controller.release_button(melee.Button.BUTTON_Y)
            self.controller.release_button(melee.Button.BUTTON_A)
        self.controller.flush()

    def move(self, x: float, y: float):
        self.controller.tilt_analog_unit(melee.Button.BUTTON_MAIN, x, y)

    def jump(self):
        self.controller.press_button(melee.Button.BUTTON_Y)
        self.controller.flush()

    def jab(self):
        self.controller.press_button(melee.Button.BUTTON_A)
        self.controller.flush()
