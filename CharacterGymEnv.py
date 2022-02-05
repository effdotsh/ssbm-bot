import time

import gym
import random

import numpy
from gym import spaces
import math
import numpy as np
import gameManager
import melee

import movesList
import utils
import copy

from movesList import moveset


class CharacterEnv(gym.Env):
    def __init__(self, player_port, opponent_port, game: gameManager.Game):
        self.framedata: melee.framedata.FrameData = melee.framedata.FrameData()
        self.num_actions = len(moveset)
        self.stage = melee.Stage.BATTLEFIELD

        self.game = game

        self.controller: melee.Controller = self.game.getController(player_port)
        self.player_port = player_port
        self.opponent_port = opponent_port
        # controller_opponent = game.getController(args.opponent)

        super(CharacterEnv, self).__init__()

        self.gamestate: melee.GameState = self.game.console.step()
        self.old_gamestate = self.game.console.step()

        self.rewards = []

        self.move_x = 0

        self.kills = 0
        self.deaths = 0
        self.overjump = False  # Reward penalty if agent chooses to jump when it is already out of jumps

        self.obs = self.reset()



        num_inputs = self.get_observation(self.gamestate).shape[0]

        self.observation_space = spaces.Box(shape=np.array([num_inputs]), dtype=np.float, low=-1, high=1)
        self.action_space = spaces.Discrete(self.num_actions)

    def step(self, action: int):
        self.move = moveset[action]

    def set_gamestate(self, gamestate: melee.GameState):
        self.old_gamestate = self.gamestate
        self.gamestate = gamestate

    def get_observation(self, gamestate):
        player: melee.PlayerState = gamestate.players.get(self.player_port)
        opponent: melee.PlayerState = gamestate.players.get(self.opponent_port)

        opponent_attacking = 1 if self.framedata.attack_state(opponent.character, opponent.action,
                                                              opponent.action_frame) != melee.AttackState.NOT_ATTACKING else -1

        player_facing = 1 if player.facing else -1
        opponent_facing = 1 if player.facing else -1

        opponent_vel_x = utils.clamp(
            (opponent.speed_air_x_self + opponent.speed_ground_x_self + opponent.speed_x_attack) / 1000, -1, 1)
        opponent_vel_y = utils.clamp((opponent.speed_y_self + opponent.speed_y_attack) / 1000, -1, 1)

        blastzones = melee.BLASTZONES.get(self.stage)
        edge = melee.EDGE_POSITION.get(self.stage)

        player_on_ground = 1 if player.on_ground else -1
        opponent_on_ground = 1 if opponent.on_ground else -1

        player_off_stage = 1 if player.off_stage else -1
        opponent_off_stage = 1 if opponent.off_stage else -1

        player_jumps_left = player.jumps_left / self.framedata.max_jumps(player.character)
        opponent_jumps_left = opponent.jumps_left / self.framedata.max_jumps(opponent.character)

        player_grabbed = 1 if player.action in [melee.Action.GRABBED, melee.Action.GRABBED_WAIT_HIGH] else -1
        opponent_grabbed = 1 if player.action in [melee.Action.GRABBED, melee.Action.GRABBED_WAIT_HIGH] else -1

        obs = np.array(
            [(edge - player.position.x) / 300, (-edge - player.position.x) / 300, (edge - opponent.position.x) / 300,
             (-edge - opponent.position.x) / 300, player.position.x / blastzones[0],
             opponent.position.x / blastzones[0], player.position.y / 100, opponent.position.y / 100,
             opponent_attacking, player_facing, opponent_attacking, opponent.speed_air_x_self / 10,
             opponent.speed_ground_x_self / 10, opponent.speed_x_attack / 10, opponent.speed_y_attack / 10,
             opponent.speed_y_self, player.speed_air_x_self / 10, player.speed_ground_x_self / 10,
             player.speed_x_attack / 10, player.speed_y_attack / 10, player.speed_y_self, player.percent / 300,
             opponent.percent / 300, player_on_ground, opponent_on_ground, player_off_stage, opponent_off_stage,
             self.move_x, player_jumps_left, opponent_jumps_left, player_grabbed, opponent_grabbed, gamestate.distance/500, 1])

        return obs

    def calculate_reward(self, old_gamestate: melee.GameState, new_gamestate: melee.GameState):
        old_player: melee.PlayerState = old_gamestate.players.get(self.player_port)
        old_opponent: melee.PlayerState = old_gamestate.players.get(self.opponent_port)

        new_player: melee.PlayerState = new_gamestate.players.get(self.player_port)
        new_opponent: melee.PlayerState = new_gamestate.players.get(self.opponent_port)

        out_of_bounds = 0
        edge_position: float = melee.stages.EDGE_POSITION.get(self.game.stage)
        blastzones = melee.stages.BLASTZONES.get(self.game.stage)

        if abs(new_player.x) > edge_position:
            out_of_bounds -= 0.2
        if abs(new_opponent.x) > edge_position:
            out_of_bounds += 0.1
        if new_player.y < blastzones[3] * 0.75 or new_player.y > blastzones[2] * 0.75:
            out_of_bounds -= 0.4
        if new_opponent.y < blastzones[3] * 0.75 or new_opponent.y > blastzones[2] * 0.75:
            out_of_bounds += 0.2


        time_penalty = -1/800 * (new_gamestate.frame - old_gamestate.frame)

        reward = math.tanh((new_opponent.percent - new_player.percent) / 200) + out_of_bounds - new_gamestate.distance/1000

        reward = time_penalty + math.tanh(((new_opponent.percent - old_opponent.percent) - (new_player.percent - old_player.percent))/50)
        # reward = new_player.x / 120
        # reward = (new_player.x - old_player.x)/50
        # print(f'REEEE: {reward}')
        if self.kills >= 1:
            reward = 0.99
            self.move_queue = []
            self.move_x = 0
        if self.deaths >= 1:
            reward = -1
            self.move_queue = []
            self.move_x = 0

        return reward

    def reset(self):
        return self.get_observation(self.gamestate)



    def act(self):
        # Check for deaths
        self.controller.release_all()
        player: melee.PlayerState = self.gamestate.players.get(self.player_port)
        opponent: melee.PlayerState = self.gamestate.players.get(self.opponent_port)

        self.controller.press_button(self.move.button)
        for axis_movement in self.move.axes:
            self.controller.tilt_analog_unit(axis_movement.axis, axis_movement.x, axis_movement.y)


        self.controller.flush()