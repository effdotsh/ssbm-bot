#!/usr/bin/python3
import argparse
import math
import time

import gameManager
import melee
import platform

# from EasyML.Spartnn import Overseer

from EasyML.DQNTorch import DQNAgent

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

flipper = False

rewards = []
start_time = time.time()
if __name__ == '__main__':
    game = gameManager.Game(args)
    game.enterMatch(cpu_level=6, opponant_character=melee.Character.MARTH)

    env = CharacterEnv(player_port=args.port, opponent_port=args.opponent, game=game)

    num_inputs = env.obs.shape[0]
    num_actions = env.num_actions

    model = DQNAgent(num_inputs=num_inputs, num_outputs=num_actions, min_replay_size=10_000, minibatch_size=128,
                     learning_rate=0.00007, update_target_every=5, discount_factor=0.99995, epsilon_decay=0.9997, epsilon=1)

    gamestate = game.console.step()
    prev_gamestate = gamestate
    env.set_gamestate(gamestate)
    action = 0
    tot_steps = 0
    while True:

        current_state = env.reset()
        episode_reward = 0
        step = 0
        done = False

        while not done:

            gamestate = game.console.step()
            # if gamestate is None:
            #     continue
            # if game.console.processingtime * 1000 > 30:
            #     print("WARNING: Last frame took " + str(game.console.processingtime * 1000) + "ms to process.")

            env.set_gamestate(gamestate)

            character_ready = env.act()
            if character_ready:
                # update model from previous move
                reward = env.calculate_reward(prev_gamestate, gamestate)
                print(f'{round(time.time() - start_time, 1)}: {reward}')
                episode_reward += reward
                old_obs = env.get_observation(prev_gamestate)
                obs = env.get_observation(gamestate)
                done = env.deaths >= 1 or env.kills >=1

                #Don't let the model think that being where the spawn gateis is the bad thing
                model.update_replay_memory((old_obs, action, reward, obs if env.deaths==0 else old_obs, done))

                model.train(done)
                step += 1

                action = model.predict(env.get_observation(gamestate), True)
                env.step(action)
                tot_steps += 1

                prev_gamestate = gamestate

        print('##################################')
        print(f'Epsilon Greedy: {model.epsilon}')
        print(f'Total Steps: {tot_steps}')
        print(f'Replay Size: {len(model.replay_memory)}')
        print(f'Average Reward: {episode_reward/step}')
