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


class Move:
    def __init__(self, button=None, axis=None, x=0, y=0, num_frames=0):
        self.frames_remaining = num_frames
        self.y = y
        self.x = x
        self.axis = axis
        self.button = button


class CharacterEnv(gym.Env):
    def __init__(self, player_port, opponent_port, game: gameManager.Game):
        self.num_actions = 11

        self.stage = melee.Stage.BATTLEFIELD

        self.game = game

        self.controller: melee.Controller = self.game.getController(player_port)
        self.player_port = player_port
        self.opponent_port = opponent_port
        # controller_opponent = game.getController(args.opponent)

        super(CharacterEnv, self).__init__()

        self.gamestate: melee.GameState = self.game.console.step()
        self.old_gamestate = self.game.console.step()

        nun_inputs = self.get_observation(self.gamestate).shape[0]
        self.observation_space = spaces.Box(shape=np.array([nun_inputs]), dtype=np.float, low=-1, high=1)
        self.action_space = spaces.Discrete(self.num_actions)

        self.rewards = []

        self.move_x = 0

        self.kills = 0
        self.deaths = 0
        self.overjump = False # Reward penalty if agent chooses to jump when it is already out of jumps

        self.obs = self.reset()

        self.move_queue = []
        self.last_action = 0

    def step(self, action: int):

        self.queue_action(action)

        obs = self.get_observation(self.gamestate)

        r = self.calculate_reward(self.old_gamestate, self.gamestate)

        self.kills = 0
        self.deaths = 0


        return [obs, r, self.deaths > 1, {}]  # These returns don't work for this environment, coded differently in main.py

    def set_gamestate(self, gamestate: melee.GameState):
        self.old_gamestate = self.gamestate
        self.gamestate = gamestate

    def get_observation(self, gamestate):
        player: melee.PlayerState = gamestate.players.get(self.player_port)
        opponent: melee.PlayerState = gamestate.players.get(self.opponent_port)

        opponent_action = opponent.action
        self_action = player.action

        opponent_attacking = 1 if opponent_action in utils.attacking_list else 0
        self_attacking = 1 if self_action in utils.attacking_list else 0

        player_facing = 1 if player.facing else -1
        opponent_facing = 1 if player.facing else -1

        opponent_vel_x = max(
            (opponent.speed_air_x_self + opponent.speed_ground_x_self + opponent.speed_x_attack) / 1000, 1)
        opponent_vel_y = max((opponent.speed_y_self + opponent.speed_y_attack) / 1000, 1)

        self_vel_x = max((player.speed_air_x_self + player.speed_ground_x_self + player.speed_x_attack) / 1000, 1)
        self_vel_y = max((player.speed_y_self + player.speed_y_attack) / 1000, 1)

        blastzones = melee.BLASTZONES.get(self.stage)

        obs = np.array(
            [player.x / blastzones[1], player.y / blastzones[2], opponent.x / blastzones[1], opponent.y / blastzones[2],
             player_facing, opponent_attacking, opponent_facing, opponent_vel_x, opponent_vel_y,
             self_vel_x, self_vel_y, opponent.percent/300, player.percent/300, player.jumps_left/2, opponent.jumps_left/2])
        # print(obs)

        return obs

    def calculate_reward(self, old_gamestate, new_gamestate):
        new_player: melee.PlayerState = new_gamestate.players.get(self.player_port)
        new_opponent: melee.PlayerState = new_gamestate.players.get(self.opponent_port)

        old_player: melee.PlayerState = old_gamestate.players.get(self.player_port)
        old_opponent: melee.PlayerState = old_gamestate.players.get(self.opponent_port)

        distance = math.dist((new_player.x, new_player.y), (new_opponent.x, new_opponent.y))

        # damage_dealt = max(0, new_opponent.percent - old_opponent.percent)
        # damage_recieved = max(0, new_player.percent - old_player.percent)

        # blast_thresh = 30
        # blastzones = melee.BLASTZONES.get(self.stage)
        # print(f'P%: {new_player.percent}')
        # if damage_recieved != 0:
        #     print(f'Recieved: {damage_recieved} dmg')
        # if damage_dealt != 0:
        #     print(f'Dealt: {damage_dealt} dmg')

        jump_penalty = 1 if self.overjump == True else 0

        reward = (new_opponent.percent - new_player.percent) / 250 - jump_penalty * 0.3

        if self.kills >= 1:
            reward = 1
        if self.deaths >= 1:
            reward = -2
        # tanh_reward = 2 / (1 + math.pow(math.e, -4.4 * reward)) - 1

        # return tanh_reward
        return reward

    def reset(self):
        # self.old_gamestate = self.game.console.step()
        return self.get_observation(self.gamestate)

    def queue_action(self, action: int):
        self.overjump = False

        # self.controller.release_all()
        # print(action)

        move_stick = melee.Button.BUTTON_MAIN
        c_stick = melee.Button.BUTTON_C

        player_state: melee.PlayerState = self.gamestate.players.get(self.player_port)
        if action == 0:  # Move Left
            move = Move(axis=move_stick, x=-1, y=0, num_frames=10)
        elif action == 1:  # Move Right
            move = Move(axis=move_stick, x=1, y=0, num_frames=10)
        elif action == 2:  # Jump
            move = Move(button=melee.Button.BUTTON_Y, num_frames=10)
            if player_state.jumps_left == 0:
                self.overjump = True
        elif action == 3:  # Drop
            move = Move(axis=move_stick, x=0, y=-1, num_frames=10)

        elif action == 4:  # smash left
            move = Move(axis=c_stick,x=-1, y=0, num_frames=10)
        elif action == 5:  # smash right
            move = Move(axis=c_stick,x=1, y=0, num_frames=10)
        elif action == 6:  # smash up
            move = Move(axis=c_stick,x=0, y=1, num_frames=10)
        elif action == 7:  # smash down
            move = Move(axis=c_stick,x=0, y=-1, num_frames=10)

        elif action == 8:  # special left
            move = Move(axis=move_stick,x=-1, y=0, button=melee.Button.BUTTON_B,num_frames=10)
        elif action == 9:  # special right
            move = Move(axis=move_stick, x=1, y=0, button=melee.Button.BUTTON_B, num_frames=10)
        elif action == 10:  # special down
            m1 = Move(axis=move_stick, x=0, y=-1, button=melee.Button.BUTTON_B, num_frames=4)
            self.move_queue.append(m1)
            move = Move(button=melee.Button.BUTTON_Y, num_frames=5)

        self.last_action = action
        self.move_queue.append(move)

    def act(self):
        #Check for deaths
        player: melee.PlayerState = self.gamestate.players.get(self.player_port)
        opponent: melee.PlayerState = self.gamestate.players.get(self.opponent_port)

        if player.action == melee.Action.ON_HALO_DESCENT and self.deaths == 0:
            self.deaths = 1
        if opponent.action == melee.Action.ON_HALO_DESCENT and self.kills == 0:
            self.kills = 1




        player_state: melee.PlayerState = self.gamestate.players[self.player_port]
        if len(self.move_queue) == 0:
            if player_state.action in utils.attacking_list or player_state.action in utils.dead_list:
                return False  # No queued action but player is either attacking or in special fall
            return True

        action: Move = self.move_queue[0]

        if action.button is not None:
            self.controller.press_button(action.button)

        if action.axis is not None:
            self.controller.tilt_analog_unit(action.axis, action.x, action.y)

        action.frames_remaining -= 1

        if action.frames_remaining <= 0:
            self.move_queue.pop(0)
            self.controller.release_all()

        return False
