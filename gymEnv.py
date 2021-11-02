import gym
import melee
from gym import spaces
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env

import numpy as np
import game


class Observation:
    player_x: int
    player_y: int
    opponent_x: int
    opponent_y: int

    def getValues(self):
        return np.array([self.player_x, self.player_y, self.opponent_x, self.opponent_y])


class CharacterEnv:
    def __init__(self, state: melee.GameState, port, port_opponent):
        super(CharacterEnv, self).__init__()
        self.port = port
        self.port_opponent = port_opponent
        self.gamestate = state

        base_observation = self.getObservation()
        self.observation_space = spaces.Box(shape=base_observation.shape[0], high=1, low=-1)

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

        print(ob.player_x)
        return ob.getValues()
