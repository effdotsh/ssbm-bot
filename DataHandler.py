import os
import pickle
import time

import numpy as np
import melee

import MovesList

from tqdm import tqdm

import sklearn
from sklearn.neighbors import KDTree

framedata = melee.FrameData()


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
    x = player.position.x
    y = player.position.y * 3

    percent = player.percent * 10
    shield = player.shield_strength

    edge = melee.EDGE_POSITION.get(gamestate.stage)

    offstage = 999999 if abs(player.position.x) > edge - 1 else 0
    tumbling = 999999 if player.action in [melee.Action.TUMBLING] else 0
    is_attacking = 999999 if framedata.is_attack(player.character, player.action) else 0
    on_ground = 999999 if player.on_ground else 0

    vel_y = (player.speed_y_self + player.speed_y_attack)
    vel_x = (player.speed_x_attack + player.speed_air_x_self + player.speed_ground_x_self)

    facing = 999999.0 if player.facing else 0
    # return [x, y, percent, shield, is_attacking, on_ground, vel_x, vel_y, facing]
    in_hitstun = 999999.0 if player.hitlag_left else 0
    is_invulnerable = 999999 if player.invulnerable else 0

    special_fall = 99999999.0 if player.action in MovesList.special_fall_list else 0
    is_dead = 999999999 if player.action in MovesList.dead_list else 0

    jumps_left = player.jumps_left * 500

    attack_state = framedata.attack_state(player.character, player.action, player.action_frame)
    attack_active = 999999.0 if attack_state == melee.AttackState.ATTACKING else 0
    attack_cooldown = 999999.0 if attack_state == melee.AttackState.COOLDOWN else 0
    attack_windup = 999999.0 if attack_state == melee.AttackState.WINDUP else 0

    is_bmove = 999999.0 if framedata.is_bmove(player.character, player.action) else 0

    return [tumbling, offstage, special_fall, is_dead, percent, shield, on_ground, is_attacking,
            # x, y, vel_x, vel_y,
            facing, in_hitstun, is_invulnerable, jumps_left, attack_windup, attack_active, attack_cooldown, is_bmove]


def generate_input(gamestate: melee.GameState, player_port: int, opponent_port: int):
    player: melee.PlayerState = gamestate.players.get(player_port)
    opponent: melee.PlayerState = gamestate.players.get(opponent_port)
    if player is None or opponent is None:
        return None
    x_dist = player.position.x - opponent.position.x
    d = 999999
    if x_dist < 5:
        d = 999999 * 2
    elif x_dist < 10:
        d = 999999 * 3
    elif x_dist < 30:
        d = 999999 * 4

    direction = 999999 if player.position.x < opponent.position.x else 0

    firefoxing = 999999 if player.character in [melee.Character.FOX, melee.Character.FALCO] and player.action in [
        melee.Action.SWORD_DANCE_3_MID, melee.Action.SWORD_DANCE_3_LOW, melee.Action.SWORD_DANCE_3_HIGH,
        melee.Action.SWORD_DANCE_3_LOW_AIR, melee.Action.SWORD_DANCE_3_MID_AIR,
        melee.Action.SWORD_DANCE_3_HIGH_AIR] else 0


    obs = [direction, firefoxing, x_dist]
    obs += get_player_obs(player, gamestate)
    obs += get_player_obs(opponent, gamestate)

    return np.array(obs).flatten()


buttons = [[melee.Button.BUTTON_A], [melee.Button.BUTTON_B], [melee.Button.BUTTON_X, melee.Button.BUTTON_Y],
           [melee.Button.BUTTON_Z], [melee.Button.BUTTON_L, melee.Button.BUTTON_R]]


def generate_output(gamestate: melee.GameState, player_port: int):
    controller: melee.ControllerState = gamestate.players.get(player_port).controller_state
    b = melee.Button

    A = 1 if controller.button.get(b.BUTTON_A) else 0
    B = 1 if controller.button.get(b.BUTTON_B) else 0
    X = 1 if controller.button.get(b.BUTTON_X) else 0
    Y = 1 if controller.button.get(b.BUTTON_Y) else 0
    Z = 1 if controller.button.get(b.BUTTON_Z) else 0
    L = 1 if controller.button.get(b.BUTTON_L) else 0
    R = 1 if controller.button.get(b.BUTTON_R) else 0
    MAIN_STICK = controller.main_stick
    C_STICK = controller.c_stick
    L_SHOULDER = controller.l_shoulder
    R_SHOULDER = controller.r_shoulder

    action = [A, B, X, Y, Z, L, R, MAIN_STICK, C_STICK, L_SHOULDER, R_SHOULDER]

    return action


# nothing_chance = 0.05
def create_model(replay_paths, player_character: melee.Character,
                 opponent_character: melee.Character, stage: melee.Stage):
    pickle_file_path = f'models/{player_character.name}_v_{opponent_character.name}_on_{stage.name}.pkl'
    X = []
    map = {}

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

        last_input = gamestate.players.get(player_port).character_controller
        while True:
            try:
                gamestate: melee.GameState = console.step()
            except:
                break
            if gamestate is None or gamestate.stage is None:
                break
            inp = generate_input(gamestate=gamestate, player_port=player_port, opponent_port=opponent_port)
            action = generate_output(gamestate=gamestate, player_port=player_port)
            if action == dud:
                continue
            if inp is None:
                break

            if action not in MovesList.bad_moves:
                X.append(inp)
                key = str(list(inp.astype(int)))
                map |= {key: action}

            # if player_character == opponent_character:
            #     inp = generate_input(gamestate=gamestate, player_port=opponent_port, opponent_port=player_port)
            #     action = generate_output(gamestate=gamestate, player_port=opponent_port)
            #     if action == dud:
            #         continue
            #     if inp is None:
            #         break
            #     if action not in MovesList.bad_moves:
            #         X.append(inp)
            #         key = str(list(inp.astype(int)))
            #         map |= {key: action}



    tree = KDTree(X)
    if not os.path.isdir('models'):
        os.mkdir('models/')
    with open(pickle_file_path, 'wb') as file:
        pickle.dump({'tree': tree, 'map': map}, file)
    return tree, map


def load_model(player_character: melee.Character,
               opponent_character: melee.Character, stage: melee.Stage):
    pickle_file_path = f'models/{player_character.name}_v_{opponent_character.name}_on_{stage.name}.pkl'
    if os.path.exists(pickle_file_path):
        with open(pickle_file_path, 'rb') as file:
            data = pickle.load(file)
            return data['tree'], data['map']
    else:
        print("Model does not exist")
        quit()

dud = [0, 0, 0, 0, 0, 0, 0, (0.5, 0.5), (0.5, 0.5), 0.0, 0.0]

def predict(tree: KDTree, map: dict, gamestate: melee.GameState, player_port: int, opponent_port: int, num_points=500):
    inp = generate_input(gamestate=gamestate, player_port=player_port, opponent_port=opponent_port)
    dist, ind = tree.query([inp], k=num_points)

    player: melee.PlayerState = gamestate.players.get(player_port)
    opponent: melee.PlayerState = gamestate.players.get(opponent_port)
    # [0, 0, 0, 0, 0, 0, 0, (0.5, 0.5), (0.5, 0.5), 0.0, 0.0]
    # [A, B, X, Y, Z, L, R, MAIN_STICK, C_STICK, L_SHOULDER, R_SHOULDER]

    point = list(np.array(tree.data[0]).astype(int))
    action = map[str(point)]
    first = True
    for e, i in enumerate(ind[0]):
        p = list(np.array(tree.data[i]).astype(int))
        a = map[str(p)]

        if dist[0][e] > 30:
            print('nothing', time.time())
            if player.x < opponent.x:
                return [0, 0, 0, 0, 0, 0, 0, (1, 0.5), (0.5, 0.5), 0.0, 0.0]
            return [0, 0, 0, 0, 0, 0, 0, (0, 0.5), (0.5, 0.5), 0.0, 0.0]

            # break

        if a != dud:
            return a
        elif first:
            first = False
            action = a

    return action
