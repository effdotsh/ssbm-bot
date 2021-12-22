#!/usr/bin/python3
import argparse
import math
import gameManager
import melee
import platform

from EasyML.Spartnn import Overseer

from torch import nn

from CharacterEnv import CharacterEnv

dolphin_path = ''
if platform.system() == "Darwin":
    dolphin_path = "/Users/human/Library/Application Support/Slippi Launcher/netplay/Slippi Dolphin.app"
elif platform.system() == "Windows":
    dolphin_path = "C:/Users/human/AppData/Roaming/Slippi Launcher/netplay/"

print(dolphin_path)


def check_port(value):
    ivalue = int(value)
    if ivalue < 1 or ivalue > 4:
        raise argparse.ArgumentTypeError("%s is an invalid controller port. \
                                         Must be 1, 2, 3, or 4." % value)
    return ivalue


parser = argparse.ArgumentParser(description='Example of libmelee in action')
parser.add_argument('--port', '-p', type=check_port,
                    help='The controller port (1-4) your AI will play on',
                    default=1)
parser.add_argument('--opponent', '-o', type=check_port,
                    help='The controller port (1-4) the opponent will play on',
                    default=2)
parser.add_argument('--debug', '-d', action='store_true',
                    help='Debug mode. Creates a CSV of all game states')
parser.add_argument('--address', '-a', default="127.0.0.1",
                    help='IP address of Slippi/Wii')
parser.add_argument('--dolphin_executable_path', '-e',
                    help='The directory where dolphin is',
                    default=dolphin_path)
parser.add_argument('--connect_code', '-t', default="",
                    help='Direct connect code to connect to in Slippi Online')
parser.add_argument('--iso', default='SSBM.iso', type=str,
                    help='Path to melee iso.')

args: gameManager.Args = parser.parse_args()

if __name__ == '__main__':
    game = gameManager.Game(args)
    game.enterMatch(cpu_level=5, opponant_character=melee.Character.FALCO)

    env = CharacterEnv(player_port=args.port, opponent_port=args.opponent, game=game)

    num_inputs = env.obs.shape[0]
    num_choices = env.num_actions

    reward_network = nn.Sequential(
        nn.Linear(num_inputs + num_choices, 20),
        nn.Sigmoid(),
        nn.Linear(20, 10),
        nn.Sigmoid(),
        nn.Linear(10, 10),
        nn.Sigmoid(),
        nn.Linear(10, 1)
    )
    state_network = nn.Sequential(
        nn.Linear(num_inputs + num_choices, 20),
        nn.Sigmoid(),
        nn.Linear(20, 20),
        nn.Sigmoid(),
        nn.Linear(20, num_inputs)
    )

    nn = Overseer(num_inputs=env.obs.shape[0], num_choices=env.num_actions, epsilon_greedy_chance=1,
                  epsilon_greedy_decrease=0.00005, reward_network_layers=reward_network, search_depth=2,
                  discount_factor=0.4, reward_network_learning_rate=0.0001)

    new_state = game.console.step()
    env.set_gamestate(new_state)
    state = new_state

    decision_counter = 0
    while True:

        new_gamestate = game.console.step()
        if new_gamestate is None:
            continue
        if game.console.processingtime * 1000 > 30:
            print("WARNING: Last frame took " + str(game.console.processingtime * 1000) + "ms to process.")


        env.set_gamestate(new_gamestate)
        character_ready = env.act()
        if character_ready:
            obs = env.get_observation(state)
            new_obs = env.get_observation(new_state)

            reward = env.calculate_reward(old_gamestate=state, new_gamestate=new_state)

            nn.learn_state(chosen_action=env.last_action, old_state=obs, new_state=new_obs)
            nn.learn_reward(chosen_action=env.last_action, inputs=obs, observed_reward=reward)

            state = new_state

            action = nn.predict(new_obs)
            env.step(action)
            decision_counter += 1

        state = new_gamestate
        if (decision_counter % 30 == 0):
            nn.log(100)
            decision_counter += 1
