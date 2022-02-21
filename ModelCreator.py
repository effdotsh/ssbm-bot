from enum import Enum

from EasyML.DQN.DQN import DQN
from EasyML.SAC.SAC import SAC


class Algorithm(Enum):
    SAC = 0
    DQN = 1


def create_model(algorithm: Algorithm, num_inputs: int, num_actions: int):
    if algorithm == Algorithm.SAC:
        model = SAC(num_inputs=num_inputs, num_actions=num_actions)
    elif algorithm == Algorithm.DQN:
        model = DQN(num_inputs=num_inputs, num_actions=num_actions)
    else:
        model = DQN(num_inputs=num_inputs, num_actions=num_actions)

    return model

def train_every(algorithm: Algorithm) -> int:
    if algorithm == algorithm.SAC:
        return 1
    elif algorithm == Algorithm.DQN:
        return 512

    return 1
