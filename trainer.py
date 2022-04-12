import os

import numpy as np
import melee

import MovesList
from ReplayManager import get_ports

from torch import nn
import torch
from tqdm import tqdm

from torchsample.modules import ModuleTrainer

framedata = melee.FrameData()


class Network(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.l1 = nn.Linear(obs_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 128)
        self.l4 = nn.Linear(128, action_dim)
        # self.layers = nn.Sequential(
        #     nn.Linear(obs_dim, 256),
        #     nn.LeakyReLU(),
        #     nn.Linear(256, 256),
        #     nn.LeakyReLU(),
        #     nn.Linear(256, action_dim)
        # )

    def forward(self, x):
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        x = torch.relu(self.l3(x))
        x = torch.relu(self.l4(x))

        return x


def get_player_obs(player: melee.PlayerState) -> list:
    x = player.position.x / 100
    y = player.position.y / 100
    percent = player.percent / 150
    shield = player.shield_strength / 60

    is_attacking = framedata.is_attack(player.character, player.action)
    on_ground = player.on_ground

    vel_y = (player.speed_y_self + player.speed_y_attack) / 10
    vel_x = (player.speed_x_attack + player.speed_air_x_self + player.speed_ground_x_self) / 10

    facing = 1 if player.facing else -1
    return [x, y, percent, shield, is_attacking, on_ground, vel_x, vel_y, facing]
    in_hitstun = 1 if player.hitlag_left else -1
    is_invulnerable = 1 if player.invulnerable else -1

    special_fall = 1 if player.action in MovesList.special_fall_list else -1
    is_dead = 1 if player.action in MovesList.dead_list else -1

    jumps_left = player.jumps_left / framedata.max_jumps(player.character)

    attack_state = framedata.attack_state(player.character, player.action, player.action_frame)
    attack_active = 1 if attack_state == melee.AttackState.ATTACKING else -1
    attack_cooldown = 1 if attack_state == melee.AttackState.COOLDOWN else -1
    attack_windup = 1 if attack_state == melee.AttackState.WINDUP else -1

    is_bmove = 1 if framedata.is_bmove(player.character, player.action) else -1

    stock = player.stock / 4

    return [special_fall, is_dead, vel_x, vel_y, x, y, percent, shield, on_ground, is_attacking, facing,
            in_hitstun, is_invulnerable, jumps_left, attack_windup, attack_active, attack_cooldown, is_bmove, stock]


def generate_input(gamestate: melee.GameState, player_port: int, opponent_port: int):
    player: melee.PlayerState = gamestate.players.get(player_port)
    opponent: melee.PlayerState = gamestate.players.get(opponent_port)

    obs = []
    obs.append(get_player_obs(player))
    obs.append(get_player_obs(opponent))
    return np.array(obs).flatten()


buttons = [[melee.Button.BUTTON_A], [melee.Button.BUTTON_B], [melee.Button.BUTTON_X, melee.Button.BUTTON_Y],
           [melee.Button.BUTTON_Z]]


def generate_output(gamestate: melee.GameState, player_port: int):
    controller: melee.ControllerState = gamestate.players.get(player_port).controller_state
    state = []
    for combo in buttons:
        active = False
        for b in combo:
            active = active or controller.button.get(b)
        state.append(1 if active else 0)

    # state.append(controller.l_shoulder + controller.r_shoulder)
    # state.append((controller.c_stick[0] - 0.5) * 2)
    # state.append((controller.c_stick[1] - 0.5) * 2)
    state.append(1 if (controller.main_stick[0] - 0.5) * 2 > 0.2 else -1 if (controller.main_stick[0] - 0.5) * 2 < -0.2 else 0)
    state.append(1 if (controller.main_stick[1] - 0.5) * 2 > 0.2 else -1 if (controller.main_stick[1] - 0.5) * 2 < -0.2 else 0)

    return np.array(state)


def load_data(replay_paths, player_character: melee.Character,
              opponent_character: melee.Character):
    X = []
    Y = []
    for path in tqdm(replay_paths):
        console = melee.Console(is_dolphin=False,
                                allow_old_version=True,
                                path=path)
        try:
            console.connect()
        except:
            console.stop()
            continue

        gamestate: melee.GameState = console.step()
        player_port, opponent_port = get_ports(gamestate, player_character=player_character,
                                               opponent_character=opponent_character)
        if player_port == -1:
            continue

        while gamestate is not None:
            inp = generate_input(gamestate=gamestate, player_port=player_port, opponent_port=opponent_port)
            out = generate_output(gamestate=gamestate, player_port=player_port)
            for val in out[:4]:
                if val != 0:
                    X.append(inp)
                    Y.append(out)
                    # print(out)
                    break
            gamestate: melee.GameState = console.step()

    return np.array(X), np.array(Y)


def train(replay_paths, player_character: melee.Character, opponent_character: melee.Character,
          batch_size=1024):
    X, Y = load_data(replay_paths=replay_paths, player_character=player_character,
                                     opponent_character=opponent_character)

    input_dim = len(X[0])
    output_dim = len(Y[0])


    model = Network(input_dim, output_dim)
    trainer = ModuleTrainer(model)

    optim = torch.optim.Adam(model.parameters(),
                             lr=3e-4)
    trainer.compile(loss='mse_loss',
                         optimizer=optim)
    trainer.fit(torch.Tensor(X), torch.Tensor(Y), batch_size=128, verbose=1, shuffle=True)

    save_dir = 'models/'
    model_name = f'{player_character.name}_v_{opponent_character.name}_on_BATTLEFIELD'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)


    torch.save(model, os.path.join(save_dir, model_name))