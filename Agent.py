import math
import time
from collections import deque

import melee
import numpy as np

import GameManager
import MovesList
from CharacterController import CharacterController

from ModelCreator import create_model, Algorithm, train_every

import wandb


class Agent:
    def __init__(self, player_port: int, opponent_port: int, game: GameManager.Game, algorithm: Algorithm,
                 use_wandb: bool):
        self.use_wandb = use_wandb
        self.algorithm = algorithm
        self.game = game
        self.opponent_port = opponent_port
        self.player_port = player_port
        self.framedata = melee.FrameData()

        self.step = 0

        self.prev_gamestate = self.game.get_gamestate()
        obs = self.get_observation(self.prev_gamestate)
        num_inputs = len(obs)
        moves_list = MovesList.moves_list
        num_actions = len(moves_list)

        self.model = create_model(algorithm=algorithm, num_inputs=num_inputs, num_actions=num_actions)
        self.controller = CharacterController(player_port=self.player_port, game=self.game, moves_list=moves_list)
        self.action = 0

        self.kdr = deque(maxlen=100)
        self.rewards = deque(maxlen=4 * 60 * 60)
        if use_wandb:
            wandb.init(project="SmashBotSC", name=f'{self.algorithm.name}-{int(time.time())}')
        print("wandb logged in")

    def run_frame(self, gamestate: melee.GameState) -> None:
        self.step += 1
        reward = self.get_reward(gamestate, self.prev_gamestate)
        self.rewards.append(reward)

        prev_obs = self.get_observation(self.prev_gamestate)
        obs = self.get_observation(gamestate)
        self.model.learn_experience(prev_obs, self.action, reward, obs, False)

        if self.step % train_every(self.algorithm) == 0:
            self.model.train()

        self.update_kdr(gamestate=gamestate, prev_gamestate=self.prev_gamestate)

        if self.use_wandb:
            obj = {
                'Average Reward': np.mean(self.rewards),
                'Reward': reward,
                'KDR': np.sum(self.kdr),
            }
            model_log = self.model.get_log()
            wandb.log(obj | model_log)

        self.action = self.model.predict(obs)

        self.controller.act(self.action)
        self.prev_gamestate = gamestate

    def update_kdr(self, gamestate, prev_gamestate):
        new_player: melee.PlayerState = gamestate.players.get(self.player_port)
        new_opponent: melee.PlayerState = gamestate.players.get(self.opponent_port)
        old_player: melee.PlayerState = prev_gamestate.players.get(self.player_port)
        old_opponent: melee.PlayerState = prev_gamestate.players.get(self.opponent_port)

        if new_player.action in MovesList.dead_list and old_player.action not in MovesList.dead_list:
            self.kdr.append(-1)
        if new_opponent.action in MovesList.dead_list and old_opponent.action not in MovesList.dead_list:
            self.kdr.append(1)

    def get_player_obs(self, player: melee.PlayerState) -> list:
        x = player.position.x / 100
        y = player.position.y / 100
        percent = player.percent / 150
        sheild = player.shield_strength / 60

        is_attacking = self.framedata.is_attack(player.character, player.action)
        on_ground = player.on_ground

        vel_y = (player.speed_y_self + player.speed_y_attack) / 10
        vel_x = (player.speed_x_attack + player.speed_air_x_self + player.speed_ground_x_self) / 10

        facing = 1 if player.facing else -1

        in_hitstun = 1 if player.hitlag_left else -1
        is_invounrable = 1 if player.invulnerable else -1

        special_fall = 1 if player.action in MovesList.special_fall_list else -1
        is_dead = 1 if player.action in MovesList.dead_list else -1

        jumps_left = player.jumps_left / self.framedata.max_jumps(player.character)

        attack_state = self.framedata.attack_state(player.character, player.action, player.action_frame)
        attack_active = 1 if attack_state == melee.AttackState.ATTACKING else -1
        attack_cooldown = 1 if attack_state == melee.AttackState.COOLDOWN else -1
        attack_windup = 1 if attack_state == melee.AttackState.WINDUP else -1

        return [special_fall, is_dead, vel_x, vel_y, x, y, percent, sheild, on_ground, is_attacking, facing,
                in_hitstun, is_invounrable, jumps_left, attack_windup, attack_active, attack_cooldown]

    def get_observation(self, gamestate: melee.GameState) -> np.ndarray:
        player: melee.PlayerState = gamestate.players.get(self.player_port)
        opponent: melee.PlayerState = gamestate.players.get(self.opponent_port)

        obs = []
        obs.append(self.get_player_obs(player))
        obs.append(self.get_player_obs(opponent))
        return np.array(obs).flatten()

    def get_reward(self, new_gamestate: melee.GameState, old_gamestate: melee.GameState) -> float:
        new_player: melee.PlayerState = new_gamestate.players.get(self.player_port)
        new_opponent: melee.PlayerState = new_gamestate.players.get(self.opponent_port)

        old_player: melee.PlayerState = old_gamestate.players.get(self.player_port)
        old_opponent: melee.PlayerState = old_gamestate.players.get(self.opponent_port)

        damage_dealt = max(new_opponent.percent - old_opponent.percent,0)
        damage_received = max(new_player.percent - old_player.percent,0)

        reward = math.tanh((damage_dealt-damage_received)/25)*0.5

        if new_player.action in MovesList.dead_list:
            reward = -1
        elif new_opponent.action in MovesList.dead_list:
            reward = 1

        return reward
