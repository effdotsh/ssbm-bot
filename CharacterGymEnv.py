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

        self.kills = 0
        self.deaths = 0
        self.overjump = False  # Reward penalty if agent chooses to jump when it is already out of jumps

        self.obs = self.reset()

        self.move = moveset[0]

        num_inputs = self.get_observation(self.gamestate).shape[0]

        self.observation_space = spaces.Box(shape=np.array([num_inputs]), dtype=np.float, low=-1, high=1)
        self.action_space = spaces.Discrete(self.num_actions)

    def step(self, action: int):
        self.move = moveset[action]

    def set_gamestate(self, gamestate: melee.GameState):
        self.old_gamestate = self.gamestate
        self.gamestate = gamestate

    def get_player_obs(self, player: melee.PlayerState):
        # PlayerState Info
        direction_facing = 1 if player.facing else -1

        is_on_ground = 1 if player.on_ground else -1

        is_off_stage = 1 if player.off_stage else -1
        is_grabbed = 1 if player.action in [melee.Action.GRABBED, melee.Action.GRABBED_WAIT_HIGH] else -1

        hitstun_left = player.hitstun_frames_left

        hitlag_left = player.hitlag_left

        invulnerable = 1 if player.invulnerable else -1
        invulnerability_left = player.invulnerability_left

        jumps_left = player.jumps_left / self.framedata.max_jumps(player.character)

        percent = player.percent / 200

        x = player.position.x / 300
        y = player.position.y / 200
        sheild_strength = player.shield_strength / 60
        speed_air_x_self = player.speed_air_x_self
        speed_ground_x_self = player.speed_ground_x_self
        speed_x_attack = player.speed_x_attack
        speed_y_attack = player.speed_y_attack
        speed_y_self = player.speed_y_self

        # FrameData Info
        active_hitbox = 1 if self.framedata.attack_state(player.character, player.action,
                                                         player.action_frame) != melee.AttackState.NOT_ATTACKING else -1
        is_attacking = 1 if self.framedata.is_attack(player.character, player.action) else -1
        is_b_move = 1 if self.framedata.is_bmove(player.character, player.action) else -1
        is_grab = 1 if self.framedata.is_grab(player.character, player.action) else -1
        is_roll = 1 if self.framedata.is_roll(player.character, player.action) else -1
        is_shield = 1 if self.framedata.is_shield(player.action) else -1

        return [direction_facing, is_on_ground, is_off_stage, is_grabbed, hitstun_left, hitlag_left, invulnerable,
                invulnerability_left, jumps_left, percent, x, y, sheild_strength, speed_air_x_self, speed_ground_x_self,
                speed_x_attack, speed_y_attack, speed_y_self, active_hitbox, is_attacking, is_b_move, is_grab, is_roll,
                is_shield]

    def get_observation(self, gamestate):
        player: melee.PlayerState = gamestate.players.get(self.player_port)
        opponent: melee.PlayerState = gamestate.players.get(self.opponent_port)

        obs = []
        obs.append(self.get_player_obs(player))
        obs.append(self.get_player_obs(opponent))
        obs = np.array(obs).flatten()
        return obs

    def calculate_reward(self, new_gamestate: melee.GameState):

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

        reward =utils.clamp(math.tanh((new_opponent.percent - new_player.percent) / 60), -0.8, 0.8)
        # reward = -0.1
        if new_player.action in utils.dead_list:
            reward = -1
            print(new_player.action)
        elif new_opponent.action in utils.dead_list:
            reward = 1
            print(new_opponent.action)

        return reward

    def reset(self):
        return self.get_observation(self.gamestate)

    def act(self):
        # Check for deaths
        self.controller.release_all()
        player: melee.PlayerState = self.gamestate.players.get(self.player_port)
        opponent: melee.PlayerState = self.gamestate.players.get(self.opponent_port)

        if self.move.button is not None:
            self.controller.press_button(self.move.button)
        for axis_movement in self.move.axes:
            if axis_movement.axis is not None:
                self.controller.tilt_analog_unit(axis_movement.axis, axis_movement.x, axis_movement.y)

        self.controller.flush()
