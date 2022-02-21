from enum import Enum

import melee
import numpy as np

import GameManager
import MovesList


class Algorithm(Enum):
    SAC = 0
    DQN = 1


class Agent:
    def __init__(self, player_port: int, opponent_port: int, game: gameManager.Game, algorithm: Algorithm):
        self.algorithm = algorithm
        self.game = game
        self.opponent_port = opponent_port
        self.player_port = player_port

        self.step = 0

        gamestate = self.game.get_gamestate()
        num_inputs = len(self.get_observation(gamestate))
        moves_list = movesList.moves_list


        self.model = self.create_model()

    def run_frame(self, gamestate: melee.GameState) -> None:
        pass

    def get_observation(self, gamestate: melee.GameState) -> np.ndarray:
        return np.array([0])

    def get_reward(self, gamestate: melee.GameState) -> float:
        return 0

    def create_model(self):
        pass
