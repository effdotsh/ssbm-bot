from enum import Enum

from EasyRL.Discrete.DQN.DQN_Inps import DQN
from EasyRL.Discrete.PPO.PPO import PPO
from EasyRL.Discrete.SAC.SAC1.SAC import SAC
from EasyRL.Discrete.QT.QT import QT


class Algorithm(Enum):
    SAC = 0
    DQN = 1
    PPO = 2
    QT=3


def create_model(algorithm: Algorithm, num_inputs: int, num_actions: int):
    if algorithm == Algorithm.SAC:
        model = SAC(obs_dim=num_inputs, action_dim=num_actions)
    elif algorithm == Algorithm.DQN:
        model = DQN(obs_dim=num_inputs, action_dim=num_actions)
    elif algorithm == Algorithm.PPO:
        model = PPO(obs_dim=num_inputs, action_dim=num_actions, batch_size=1000, T_horizon=1200, learning_rate=3e-5,
                    adv_normalization=False)
    elif algorithm == Algorithm.QT:
        model = QT(obs_dim=num_inputs, action_dim=num_actions, learning_rate=3e-5)
    else:
        model = DQN(obs_dim=num_inputs, action_dim=num_actions)

    return model


def train_every(algorithm: Algorithm) -> int:
    if algorithm == algorithm.SAC:
        return 1
    elif algorithm == Algorithm.DQN:
        return 512
    elif algorithm == Algorithm.DQN:
        return -1
    return 1
