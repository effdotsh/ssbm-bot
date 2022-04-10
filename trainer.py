import numpy as np
import melee
import torch
from torch import nn, optim

import MovesList
from ReplayManager import get_ports
from network import Network

framedata = melee.FrameData()


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
           [melee.Button.BUTTON_Z], [melee.Button.BUTTON_L, melee.Button.BUTTON_R]]


def generate_output(gamestate: melee.GameState, player_port: int):
    controller: melee.ControllerState = gamestate.players.get(player_port).controller_state
    state = []
    for combo in buttons:
        active = False
        for b in combo:
            active = active or controller.button.get(b)
        state.append(1 if active else 0)

    state.append(controller.l_shoulder + controller.r_shoulder)
    state.append((controller.c_stick[0] - 0.5) * 2)
    state.append((controller.c_stick[1] - 0.5) * 2)
    state.append((controller.main_stick[0] - 0.5) * 2)
    state.append((controller.main_stick[1] - 0.5) * 2)

    return np.array(state)


class Datapoint:
    def __init__(self, x, y):
        self.x = x
        self.y = y


def update_buffer(buffer: np.ndarray, replay_paths, min_buffer_size: int, player_character: melee.Character,
                  opponent_character: melee.Character):
    while len(buffer) < min_buffer_size and len(replay_paths) >= 1:
        path = replay_paths[0]
        replay_paths = replay_paths[1:]

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
            buffer = np.append(buffer, [Datapoint(inp, out)])

            gamestate: melee.GameState = console.step()
        print(len(buffer))
    return buffer


def train(replay_paths, min_buffer_size: int, player_character: melee.Character, opponent_character: melee.Character,
          batch_size=512):
    buffer = update_buffer(buffer=np.array([], dtype=Datapoint), replay_paths=replay_paths,
                           min_buffer_size=min_buffer_size,
                           player_character=player_character, opponent_character=opponent_character)
    print(buffer[:2])
    model = Network(input_dim=len(buffer[0].x), output_dim=len(buffer[0].y))

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=3e-2)
    while len(buffer) > 0:
        buffer = update_buffer(buffer=buffer, replay_paths=replay_paths, min_buffer_size=min_buffer_size,
                               player_character=player_character, opponent_character=opponent_character)

        ep_loss = 0
        for _ in range(batch_size):
            i = np.random.randint(0, len(buffer) - 1)
            inp, target = torch.Tensor(buffer[i].x), torch.Tensor(buffer[i].y)
            buffer = np.delete(buffer, i)

            output = model(inp)

            loss = criterion(output, target)

            # backpropogate through the loss gradiants
            loss.backward()

            # update model weights
            optimizer.step()

            # remove current gradients for next iteration
            optimizer.zero_grad()

            # append to loss
            ep_loss += loss


        print(ep_loss / batch_size)
