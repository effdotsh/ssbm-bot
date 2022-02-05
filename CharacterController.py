import time

import melee

import gameManager
import movesList
from CharacterGymEnv import CharacterEnv

# from EasyML.Spartnn import Overseer
from EasyML.DQNTorch import DQNAgent


class CharacterController:
    def __init__(self, port: int, opponent_port: int, game: gameManager.Game, min_replay_size=1500,
                 minibatch_size=128, max_replay_size=300_000,
                 learning_rate=0.00004, update_target_every=5, discount_factor=0.999, epsilon_decay=0.9997, epsilon=1,
                 update_model=True):
        self.update_model = update_model
        self.game = game
        self.env = CharacterEnv(player_port=port, opponent_port=opponent_port, game=game)

        num_inputs = self.env.obs.shape[0]
        num_actions = self.env.num_actions
        #
        self.model = DQNAgent(num_inputs=num_inputs, num_outputs=num_actions, min_replay_size=min_replay_size,
                              minibatch_size=minibatch_size, max_replay_size=max_replay_size,
                              learning_rate=learning_rate, update_target_every=update_target_every,
                              discount_factor=discount_factor, epsilon_decay=epsilon_decay, epsilon=epsilon)

        # self.model = DQNAgent(num_inputs=num_inputs, num_outputs=num_actions)
        # self.model = Overseer(num_inputs=num_inputs, num_outputs=num_actions, min_replay_size=1000, batch_size=64, search_depth=1, update_every=500)

        self.current_state = self.env.reset()
        self.episode_reward = 0
        self.step = 1
        self.tot_steps = 0
        self.done = False
        gamestate = self.game.console.step()
        while gamestate is None:
            gamestate = self.game.console.step()

        self.prev_gamestate = gamestate

        self.action = 0
        self.start_time = time.time()

    def run_frame(self, gamestate: melee.GameState, log: bool):
        if gamestate is None:
            return
        # if gamestate is None:
        #     continue
        # if game.console.processingtime * 1000 > 30:
        #     print("WARNING: Last frame took " + str(game.console.processingtime * 1000) + "ms to process.")

        self.env.set_gamestate(gamestate)

        self.env.act()
        self.env.controller.flush()
        # print(gamestate.players.get(env.player_port).action)
        # print(gamestate.players.get(env.player_port).action)
        #
        # self.done = self.env.deaths >= 1
        if self.step % 60*5 == 0 and log:
            print('##################################')
            print(f'Epsilon Greedy: {self.model.epsilon}')
            print(f'Total Steps: {self.tot_steps}')
            print(f'Replay Size: {len(self.model.replay_memory)}')
            print(f'Average Reward: {self.episode_reward / self.step}')
            print(f'Num Updates: {self.model.num_updates}')
            print('##################################')
        # update model from previous move
        reward = self.env.calculate_reward(self.prev_gamestate, gamestate)
        # reward = env.calculate_state_reward(gamestate) - env.calculate_state_reward(prev_gamestate)
        self.episode_reward += reward
        old_obs = self.env.get_observation(self.prev_gamestate)
        obs = self.env.get_observation(gamestate)

        done = self.env.deaths >= 1 or self.env.kills >= 1


        self.model.update_replay_memory((old_obs, self.action, reward, obs, False))


        self.step += 1

        action = self.model.predict(obs, out_eps=False)
        self.action = action
        self.env.step(action)

        self.tot_steps += 1

        self.prev_gamestate = gamestate
        self.env.kills = 0
        self.env.deaths = 0

        if self.update_model and self.tot_steps % 1024 == 0:
            self.model.train(True)
            # self.model.log(200)

            self.episode_reward = 0
            self.step = 1
            self.done = False
