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

from queue import Queue
from gameManager import Action

class FoxEnv(gym.Env):
    def __init__(self, player_port, opponent_port, game: gameManager.Game, queue: Queue):
        self.queue = queue
        self.num_actions = 17

        self.stage = melee.Stage.BATTLEFIELD

        self.game = game

        self.controller: melee.Controller = self.game.getController(player_port)
        self.player_port = player_port
        self.opponent_port = opponent_port
        # controller_opponent = game.getController(args.opponent)

        super(FoxEnv, self).__init__()

        self.gamestate: melee.GameState = self.game.getGamestate()
        self.old_gamestate = self.game.getGamestate()

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
        self.gamestate = self.game.getGamestate()

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
        self.old_gamestate = self.game.getGamestate()
        return self.get_observation(self.old_gamestate)

    def act(self, action: int):
        def waitToComplete(a: Action):
            # self.queue.put_nowait(a)
            self.game.player_actions[0] = a
            while not a.complete:
                print(time.time())

            # self.queue.get()
            # print('peiwfhnoiwehfoiwebhfo')


        if action == 0:  # Move Left
            action = Action(joystick=melee.Button.BUTTON_MAIN, x=-1, num_frames=5)
        elif action == 1:  # Move Right
            action = Action(joystick=melee.Button.BUTTON_MAIN, x=1, num_frames=5)
        elif action == 2:  # Jump
            action = Action(joystick=melee.Button.BUTTON_MAIN, y=1, num_frames=5)

        elif action == 3:  # Drop
            action = Action(joystick=melee.Button.BUTTON_MAIN, y=-1, num_frames=5)

        elif action == 4:  # Left Smash
            action = Action(joystick=melee.Button.BUTTON_C, x=-1, num_frames=39)

        elif action == 5:  # Right Smash
            action = Action(joystick=melee.Button.BUTTON_C, x=1, num_frames=39)
        elif action == 6:  # Down Smash
            action = Action(joystick=melee.Button.BUTTON_C, y=-1, num_frames=49)
        elif action == 7:  # Up Smash
            action = Action(joystick=melee.Button.BUTTON_C, y=1, num_frames=41)

        elif action == 8:  # Right B
            action = Action(joystick=melee.Button.BUTTON_MAIN, button=melee.Button.BUTTON_B, x=1)
        elif action == 9:  # Left B
            action = Action(joystick=melee.Button.BUTTON_MAIN, button=melee.Button.BUTTON_B, x=-1)
        elif action == 10:  # Down B
            action = Action(joystick=melee.Button.BUTTON_MAIN, button=melee.Button.BUTTON_B, y=-1, num_frames=4, force_end=True)
            waitToComplete(action)
            action = Action(button=melee.Button.BUTTON_Y, num_frames=4)

        ##Up B's
        elif action == 11:  # Up B Up
            action = Action(joystick=melee.Button.BUTTON_MAIN, button=melee.Button.BUTTON_B, y=1)
        elif action == 12:  # Up B Down
            action = Action(joystick=melee.Button.BUTTON_MAIN, button=melee.Button.BUTTON_B, y=1, num_frames=10, force_end=True)
            waitToComplete(action)
            action = Action(joystick=melee.Button.BUTTON_MAIN, y=-1)
        elif action == 13:  # Up B Right
            action = Action(joystick=melee.Button.BUTTON_MAIN, button=melee.Button.BUTTON_B, y=1, num_frames=10, force_end=True)
            waitToComplete(action)
            action = Action(joystick=melee.Button.BUTTON_MAIN, x=1)
        elif action == 14:  # Up B Left
            action = Action(joystick=melee.Button.BUTTON_MAIN, button=melee.Button.BUTTON_B, y=1, num_frames=10, force_end=True)
            waitToComplete(action)
            action = Action(joystick=melee.Button.BUTTON_MAIN, x=-1)
        elif action == 15:  # Up Recover Right Side
            side = melee.EDGE_POSITION.get(self.stage)
            player: melee.PlayerState = self.gamestate.players.get(self.player_port)

            x = side - player.x
            y = 10 - player.y

            action = Action(joystick=melee.Button.BUTTON_MAIN, button=melee.Button.BUTTON_B, y=1, num_frames=10, force_end=True)
            waitToComplete(action)
            action = Action(joystick=melee.Button.BUTTON_MAIN, x=x, y=y)


        elif action == 16:  # Up Recover Left Side
            side = melee.EDGE_POSITION.get(self.stage)
            player: melee.PlayerState = self.gamestate.players.get(self.player_port)

            x = -side - player.x
            y = 10 - player.y

            action = Action(joystick=melee.Button.BUTTON_MAIN, button=melee.Button.BUTTON_B, y=1, num_frames=10, force_end=True)
            waitToComplete(action)
            action = Action(joystick=melee.Button.BUTTON_MAIN, x=x, y=y)

        waitToComplete(action)