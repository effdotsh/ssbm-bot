import time

import gym
import random
from gym import spaces
import math
import numpy as np
import gameManager
import melee

import movesList
import utils
import copy

from movesList import Moves


class Move:
    def __init__(self, button=None, axis=None, x=0, y=0, num_frames=0):
        self.frames_remaining = num_frames
        self.y = y
        self.x = x
        self.axis = axis
        self.button = button


class CharacterEnv(gym.Env):
    def __init__(self, player_port, opponent_port, game: gameManager.Game, moveset):
        self.framedata: melee.framedata.FrameData = melee.framedata.FrameData()

        self.moveset = moveset

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

        self.move_queue = []
        self.last_action = 0
        self.last_action_name = ''

        nun_inputs = self.get_observation(self.gamestate).shape[0]
        self.num_actions = len(self.moveset)
        self.observation_space = spaces.Box(shape=np.array([nun_inputs]), dtype=np.float, low=-1, high=1)
        self.action_space = spaces.Discrete(self.num_actions)

    def step(self, action: int):

        self.queue_action(action)

        obs = self.get_observation(self.gamestate)

        r = self.calculate_reward(self.old_gamestate, self.gamestate)

        self.kills = 0
        self.deaths = 0

        return [obs, r, self.deaths > 1,
                {}]  # These returns don't work for this environment, coded differently in main.py

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
        opponent_jumps_left = opponent_off_stage.jumps_left / self.framedata.max_jumps(opponent.character)

        obs = np.array(
            [(edge - player.position.x) / 300, (-edge - player.position.x) / 300, (edge - opponent.position.x) / 300,
             (-edge - opponent.position.x) / 300, player.position.x / blastzones[0],
             opponent.position.x / blastzones[0], player.position.y / 100, opponent.position.y / 100,
             opponent_attacking, player_facing, opponent_attacking, opponent.speed_air_x_self / 10,
             opponent.speed_ground_x_self / 10, opponent.speed_x_attack / 10, opponent.speed_y_attack / 10,
             opponent.speed_y_self, player.speed_air_x_self / 10, player.speed_ground_x_self / 10,
             player.speed_x_attack / 10, player.speed_y_attack / 10, player.speed_y_self, player.percent / 300,
             opponent.percent / 300, player_on_ground, opponent_on_ground, player_off_stage, opponent_off_stage,
             self.move_x, player_jumps_left, opponent_jumps_left, 1])

        return obs

    def calculate_reward(self, old_gamestate, new_gamestate):
        old_player: melee.PlayerState = old_gamestate.players.get(self.player_port)
        old_opponent: melee.PlayerState = old_gamestate.players.get(self.opponent_port)

        new_player: melee.PlayerState = new_gamestate.players.get(self.player_port)
        new_opponent: melee.PlayerState = new_gamestate.players.get(self.opponent_port)

        damage_dealt = max(0, new_opponent.percent - old_opponent.percent)
        damage_recieved = max(0, new_player.percent - old_player.percent)

        jump_penalty = 1 if self.overjump else 0

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

        reward = (damage_dealt - damage_recieved) / 40 - jump_penalty * 0.3 + out_of_bounds

        if self.kills >= 1:
            reward = 1
        if self.deaths >= 1:
            reward = -1
            self.move_queue = []
            self.move_x = 0
        # tanh_reward = 2 / (1 + math.pow(math.e, -4.4 * reward)) - 1

        # return tanh_reward
        return reward

    def reset(self):
        return self.get_observation(self.gamestate)

    def queue_action(self, action: int):

        action_name = self.moveset[action]
        self.overjump = False
        # self.controller.release_all()
        # print(action)

        move_stick = melee.Button.BUTTON_MAIN
        c_stick = melee.Button.BUTTON_C

        player_state: melee.PlayerState = self.gamestate.players.get(self.player_port)
        if action_name == Moves.WALK_LEFT:  # Move Left
            move = Move(axis=move_stick, x=-1, y=0, num_frames=5)
            self.move_x = -1
        elif action_name == Moves.WALK_RIGHT:  # Move Right
            move = Move(axis=move_stick, x=1, y=0, num_frames=5)
            self.move_x = 1
        elif action_name == Moves.JUMP:  # Jump
            move = Move(button=melee.Button.BUTTON_Y, num_frames=5)
            if player_state.jumps_left == 0:
                self.overjump = True
                print('Overjump')
        elif action_name == Moves.SHORT_JUMP:  # Jump
            move = Move(button=melee.Button.BUTTON_Y, num_frames=2)
            if player_state.jumps_left == 0:
                self.overjump = True
                print('Overjump')

        elif action_name == Moves.DROP:  # Drop
            m1 = Move(num_frames=5)
            self.move_queue.append(m1)
            m2 = Move(axis=move_stick, x=0, y=-1, num_frames=20)
            self.move_queue.append(m2)
            move = Move(num_frames=5)

        elif action_name == Moves.SMASH_LEFT:  # smash left
            move = Move(axis=c_stick, x=-1, y=0, num_frames=5)
        elif action_name == Moves.SMASH_RIGHT:  # smash right
            move = Move(axis=c_stick, x=1, y=0, num_frames=5)
        elif action_name == Moves.SMASH_UP:  # smash up
            move = Move(axis=c_stick, x=0, y=1, num_frames=5)
        elif action_name == Moves.SMASH_DOWN:  # smash down
            move = Move(axis=c_stick, x=0, y=-1, num_frames=5)

        elif action_name == Moves.SPECIAL_LEFT:  # special left
            move = Move(axis=move_stick, x=-1, y=0, button=melee.Button.BUTTON_B, num_frames=5)
        elif action_name == Moves.SPECIAL_RIGHT:  # special right
            move = Move(axis=move_stick, x=1, y=0, button=melee.Button.BUTTON_B, num_frames=5)
        elif action_name == Moves.SPECIAL_DOWN:  # special down
            move = Move(axis=move_stick, x=0, y=-1, button=melee.Button.BUTTON_B, num_frames=5)
        elif action_name == Moves.SPECIAL_UP:  # special down
            move = Move(axis=move_stick, x=0, y=1, button=melee.Button.BUTTON_B, num_frames=5)

        elif action_name == Moves.WAIT:  # wait
            self.move_x = 0
            move = Move(axis=move_stick, x=0, y=0, num_frames=20)


        elif action_name == Moves.FOX_SPECIAL_DOWN:  # special down
            m1 = Move(axis=move_stick, x=0, y=-1, button=melee.Button.BUTTON_B, num_frames=4)
            self.move_queue.append(m1)
            move = Move(button=melee.Button.BUTTON_Y, num_frames=2)
        elif action_name == Moves.FOX_RECOVERY:  # Recovery
            self.move_x = 0
            print(time.time())
            sign = np.sign(player_state.x)
            target_x: float = melee.stages.EDGE_POSITION.get(self.game.stage) * sign
            angle = math.atan2(0.2 - player_state.y, target_x - player_state.x)
            m1 = Move(axis=move_stick, x=0, y=1, button=melee.Button.BUTTON_B, num_frames=15)
            self.move_queue.append(m1)
            move = Move(axis=move_stick, x=math.cos(angle), y=math.sin(angle), button=melee.Button.BUTTON_B,
                        num_frames=50)
        elif action_name == Moves.JAB:  # jab
            move = Move(button=melee.Button.BUTTON_A, num_frames=20)
        else:
            print("ACTION MISSING")
        self.last_action = action
        self.last_action_name = action_name
        self.move_queue.append(move)
        self.move_queue.append(Move(num_frames=3))  # Delay

    def act(self):
        # Check for deaths
        self.controller.release_all()
        player: melee.PlayerState = self.gamestate.players.get(self.player_port)
        opponent: melee.PlayerState = self.gamestate.players.get(self.opponent_port)

        if player.action in utils.dead_list:
            self.deaths = 1
            return False
        if opponent.action in utils.dead_list:
            self.kills = 1

        player_state: melee.PlayerState = self.gamestate.players[self.player_port]

        if player_state.action in [melee.Action.LYING_GROUND_UP, melee.Action.LYING_GROUND_UP_HIT,
                                   melee.Action.LYING_GROUND_DOWN]:
            self.move_queue = [Move(axis=melee.Button.BUTTON_MAIN, x=-1, num_frames=2)]

        if len(self.move_queue) == 0:
            if self.framedata.attack_state(player_state.character, player_state.action,
                                           player_state.action_frame) == melee.AttackState.NOT_ATTACKING and player_state.action not in utils.dead_list and not self.framedata.is_attack(
                player_state.character, player_state.action) and not self.framedata.is_grab(
                player_state.character, player_state.action) and not self.framedata.is_roll(
                player_state.character, player_state.action) and not self.framedata.is_bmove(
                player_state.character, player_state.action):
                return True
            return False

        action: Move = self.move_queue[0]

        if action.frames_remaining <= 0:
            self.move_queue.pop(0)
            self.controller.release_all()
            return False

        if action.button is not None:
            self.controller.press_button(action.button)

        if action.axis is not None:
            self.controller.tilt_analog_unit(action.axis, action.x, action.y)

        action.frames_remaining -= 1

        if action.axis != melee.Button.BUTTON_MAIN:
            self.controller.tilt_analog_unit(melee.Button.BUTTON_MAIN, self.move_x, 0)
        return False
