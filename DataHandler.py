import math
import os
import pickle
import time

import numpy as np
import melee
from tensorflow import keras

import MovesList

from tqdm import tqdm

from keras.models import Sequential
from keras.layers import Dense

framedata = melee.FrameData()


def controller_states_different(new: melee.ControllerState, old: melee.ControllerState):
    # for b in MovesList.buttons:
    for b in melee.enums.Button:
        if new.button.get(b) != old.button.get(b) and new.button.get(b):
            return True
    if new.c_stick != old.c_stick and (new.c_stick[0] * 2 - 1) ** 2 + (new.c_stick[1] * 2 - 1) ** 2 >= 0.95:
        return True
    if new.main_stick != old.main_stick and (new.main_stick[0] * 2 - 1) ** 2 + (new.main_stick[1] * 2 - 1) ** 2 >= 0.95:
        return True
    return False


def get_ports(gamestate: melee.GameState, player_character: melee.Character, opponent_character: melee.Character):
    if gamestate is None:
        return -1, -1
    ports = list(gamestate.players.keys())
    if len(ports) != 2:
        return -1, -1
    player_port = ports[0]
    opponent_port = ports[1]
    p1: melee.PlayerState = gamestate.players.get(player_port)
    p2: melee.PlayerState = gamestate.players.get(opponent_port)

    if p1.character == player_character and p2.character == opponent_character:
        player_port = ports[0]
        opponent_port = ports[1]
    elif p2.character == player_character and p1.character == opponent_character:
        player_port = ports[1]
        opponent_port = ports[0]
    else:
        print(p1.character, p2.character)
        player_port = -1
        opponent_port = -1
    return player_port, opponent_port


def get_player_obs(player: melee.PlayerState, gamestate: melee.GameState) -> list:
    x = player.position.x / 100
    y = player.position.y / 20

    percent = player.percent / 100
    shield = player.shield_strength / 60

    edge = melee.EDGE_POSITION.get(gamestate.stage)

    offstage = 1 if abs(player.position.x) > edge - 1 else -1
    tumbling = 1 if player.action in [melee.Action.TUMBLING] else -1
    is_attacking = 1 if framedata.is_attack(player.character, player.action) else -1
    on_ground = 1 if player.on_ground else -1

    vel_y = (player.speed_y_self + player.speed_y_attack) / 20
    vel_x = (player.speed_x_attack + player.speed_air_x_self + player.speed_ground_x_self) / 20

    facing = 1 if player.facing else -1
    # return [x, y, percent, shield, is_attacking, on_ground, vel_x, vel_y, facing]
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

    return [tumbling, offstage, special_fall, is_dead, shield, on_ground, is_attacking,
            x, y,
            vel_x, vel_y, percent,
            facing, in_hitstun, is_invulnerable, jumps_left, attack_windup, attack_active, attack_cooldown, is_bmove]


def generate_input(gamestate: melee.GameState, player_port: int, opponent_port: int):
    player: melee.PlayerState = gamestate.players.get(player_port)
    opponent: melee.PlayerState = gamestate.players.get(opponent_port)
    if player is None or opponent is None:
        return None

    direction = 1 if player.position.x < opponent.position.x else -1

    firefoxing = 1 if player.character in [melee.Character.FOX,
                                           melee.Character.FALCO] and player.action in MovesList.firefoxing else -1

    obs = [
        # player.position.x - opponent.position.x, player.position.y - opponent.position.y,
        firefoxing, direction, 1]
    obs += get_player_obs(player, gamestate)
    obs += get_player_obs(opponent, gamestate)

    return np.array(obs).flatten()


buttons = [[melee.Button.BUTTON_A], [melee.Button.BUTTON_B], [melee.Button.BUTTON_X, melee.Button.BUTTON_Y],
           [melee.Button.BUTTON_Z], [melee.Button.BUTTON_L, melee.Button.BUTTON_R]]


def generate_output(controller: melee.ControllerState):
    b = melee.enums.Button
    buttons = [[b.BUTTON_X, b.BUTTON_Y], [b.BUTTON_L, b.BUTTON_R], [b.BUTTON_Z], [b.BUTTON_A], [b.BUTTON_B]]

    button = 5  # 6 options
    for e, group in enumerate(buttons):
        for btn in group:
            if controller.button.get(btn):
                button = e
                break
        if button == e:
            break

    c_x = 1
    c_y = 1
    if controller.c_stick[0] < 0.25:
        c_x = 0
    elif controller.c_stick[0] > 0.75:
        c_x = 2
    if controller.c_stick[1] < 0.25:
        c_y = 0
    elif controller.c_stick[1] > 0.75:
        c_y = 2
    c_stick = 3 * c_x + c_y

    move_x = 1
    move_y = 1
    if controller.main_stick[0] < 0.25:
        move_x = 0
    elif controller.main_stick[0] > 0.75:
        move_x = 2
    if controller.main_stick[1] < 0.25:
        move_y = 0
    elif controller.main_stick[1] > 0.75:
        move_y = 2
    move_stick = 3 * move_x + move_y # 9 options

    sticks = move_stick*9 + c_stick

    action = button * 81 + sticks
    return action


# nothing_chance = 0.05
def create_model(replay_paths, player_character: melee.Character,
                 opponent_character: melee.Character, stage: melee.Stage):
    pickle_file_path = f'models/{player_character.name}_v_{opponent_character.name}_on_{stage.name}.pkl'

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
            print('console failed to connect', path, time.time())
            continue

        gamestate: melee.GameState = console.step()
        player_port, opponent_port = get_ports(gamestate, player_character=player_character,
                                               opponent_character=opponent_character)
        if player_port == -1:
            print('bad port', path, gamestate.players.keys(), time.time())

            continue

        last_input = gamestate.players.get(player_port).controller_state
        last_input_opp = gamestate.players.get(opponent_port).controller_state
        while True:
            try:
                gamestate: melee.GameState = console.step()
            except:
                break
            if gamestate is None or gamestate.stage is None:
                break

            new_input = gamestate.players.get(player_port).controller_state
            if not controller_states_different(new_input, last_input):
                continue
            last_input = new_input

            inp = generate_input(gamestate=gamestate, player_port=player_port, opponent_port=opponent_port)
            action = generate_output(new_input)
            if inp is None:
                break
            if action is None:
                break

            X.append(inp)
            Y.append(action)

    X = np.array(X)
    Y = np.array(Y)

    print(len(X), len(Y))
    print(len(X[0]), len(Y[0]))

    # train
    model = Sequential([
        Dense(64, activation='tanh', input_shape=(len(X[0]),)),
        Dense(64, activation='tanh'),
        Dense(len(Y[0]), activation='tanh'),
    ])

    opt = keras.optimizers.Adam(
        learning_rate=3e-4,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
        amsgrad=False,
        name="Adam",
    )

    model.compile(
        optimizer=opt,
        loss='mean_squared_error',
        metrics=['accuracy'],
    )

    model.fit(
        X,  # training data
        Y,  # training targets
    )

    if not os.path.isdir('models'):
        os.mkdir('models/')

    with open(pickle_file_path, 'wb') as file:
        pickle.dump(model, file)
