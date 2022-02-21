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
        self.rewards = deque(maxlen=4*60*60)
        if use_wandb:
            wandb.init(project="SmashBot", name=f'{Algorithm.value}-{time.time()}')

    def run_frame(self, gamestate: melee.GameState) -> None:
        self.step += 1
        reward = self.get_reward(gamestate)
        self.rewards.append(reward)

        prev_obs = self.get_observation(self.prev_gamestate)
        obs = self.get_observation(gamestate)
        self.model.learn_expirience(prev_obs, self.action, reward, obs, False)

        if self.step % train_every(self.algorithm) == 0:
            self.model.train()

        self.update_kdr(gamestate=gamestate, prev_gamestate=self.prev_gamestate)

        if self.use_wandb:
            obj = {
                'Average Reward': np.mean(self.rewards),
                'KDR': np.sum(self.kdr)
            }
            model_log = self.model.log_wandb()
            wandb.log(obj | model_log)

        self.action = self.model.predict(obs)
        self.controller.act(self.action)

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
        x = player.x / 100
        y = player.y / 100
        percent = min(player.percent/150, 1)
        return [x, y, percent]

    def get_observation(self, gamestate: melee.GameState) -> np.ndarray:
        player: melee.PlayerState = gamestate.players.get(self.player_port)
        opponent: melee.PlayerState = gamestate.players.get(self.opponent_port)

        obs = []
        obs.append(self.get_player_obs(player))
        obs.append(self.get_player_obs(opponent))
        return np.array(obs).flatten()

    def get_reward(self, gamestate: melee.GameState) -> float:
        player: melee.PlayerState = gamestate.players.get(self.player_port)
        opponent: melee.PlayerState = gamestate.players.get(self.opponent_port)

        reward = np.clip((opponent.percent - player.percent)/100, -0.8, 0.8)

        if player.action in MovesList.dead_list:
            reward = -1
        elif opponent.action in MovesList.dead_list:
            reward = 1

        return reward
