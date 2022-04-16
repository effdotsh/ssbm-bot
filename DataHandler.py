import os
import pickle

import numpy as np
import melee

import MovesList
from ReplayManager import get_ports

from tqdm import tqdm

from encoder import encode_to_number, decode_from_number

import sklearn
from sklearn.neighbors import KDTree

framedata = melee.FrameData()


def get_player_obs(player: melee.PlayerState, gamestate: melee.GameState) -> list:
    x = player.position.x
    y = player.position.y

    percent = player.percent / 10
    shield = player.shield_strength / 20

    edge = melee.EDGE_POSITION.get(gamestate.stage)

    offstage = 999999 if abs(player.position.x) > edge else 0

    is_attacking = 10 if framedata.is_attack(player.character, player.action) else 0
    on_ground = 5 if player.on_ground else 0

    vel_y = (player.speed_y_self + player.speed_y_attack)
    vel_x = (player.speed_x_attack + player.speed_air_x_self + player.speed_ground_x_self)

    facing = 200.0 if player.facing else 0
    # return [x, y, percent, shield, is_attacking, on_ground, vel_x, vel_y, facing]
    in_hitstun = 20.0 if player.hitlag_left else 0
    is_invulnerable = 5 if player.invulnerable else 0

    special_fall = 300.0 if player.action in MovesList.special_fall_list else 0
    is_dead = 99999999 if player.action in MovesList.dead_list else 0

    jumps_left = player.jumps_left * 500

    attack_state = framedata.attack_state(player.character, player.action, player.action_frame)
    attack_active = 5.0 if attack_state == melee.AttackState.ATTACKING else 0
    attack_cooldown = 5.0 if attack_state == melee.AttackState.COOLDOWN else 0
    attack_windup = 5.0 if attack_state == melee.AttackState.WINDUP else 0

    is_bmove = 10.0 if framedata.is_bmove(player.character, player.action) else 0

    return [special_fall, is_dead, vel_x, vel_y, x, y, percent, shield, on_ground, is_attacking, facing,
            in_hitstun, is_invulnerable, jumps_left, attack_windup, attack_active, attack_cooldown, is_bmove]


def generate_input(gamestate: melee.GameState, player_port: int, opponent_port: int):
    player: melee.PlayerState = gamestate.players.get(player_port)
    opponent: melee.PlayerState = gamestate.players.get(opponent_port)

    obs = []
    obs += get_player_obs(player, gamestate)
    obs += get_player_obs(opponent, gamestate)

    return np.array(obs).flatten()


buttons = [[melee.Button.BUTTON_A], [melee.Button.BUTTON_B], [melee.Button.BUTTON_X, melee.Button.BUTTON_Y],
           [melee.Button.BUTTON_Z], [melee.Button.BUTTON_L, melee.Button.BUTTON_R]]


def generate_output(gamestate: melee.GameState, player_port: int):
    controller: melee.ControllerState = gamestate.players.get(player_port).controller_state
    button = 0
    num_buttons = len(buttons) + 1
    for e, b in enumerate(buttons):
        if controller.button.get(b[0]):
            button = e + 1
            break

    move_x = 0 if controller.main_stick[0] < 0.3 else 1 if controller.main_stick[0] < 0.7 else 2
    move_y = 0 if controller.main_stick[1] < 0.3 else 1 if controller.main_stick[1] < 0.7 else 2
    num_move_single_axis = 3

    num_c = 5
    c = 0
    if controller.c_stick[0] < 0.3:
        c = 1
    elif controller.c_stick[0] > 0.7:
        c = 2
    elif controller.c_stick[1] < 0.3:
        c = 3
    elif controller.c_stick[1] > 0.7:
        c = 4

    maxes = [num_move_single_axis, num_move_single_axis, num_c, num_buttons]
    action = encode_to_number([move_x, move_y, c, button], maxes)

    # state = np.zeros(np.prod(maxes))
    # state[action] = 1

    return action, maxes, np.array([move_x, move_y, c, button])

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
            continue

        gamestate: melee.GameState = console.step()
        player_port, opponent_port = get_ports(gamestate, player_character=player_character,
                                               opponent_character=opponent_character)
        if player_port == -1:
            continue

        while gamestate is not None:
            inp = generate_input(gamestate=gamestate, player_port=player_port, opponent_port=opponent_port)
            action, maxes, check_arr = generate_output(gamestate=gamestate, player_port=player_port)

            if action != 120:
                X.append(inp)
                key = str(list(inp.astype(int)))

                map |= {key: action}

            gamestate: melee.GameState = console.step()

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


def predict(tree: KDTree, map: dict, gamestate: melee.GameState, player_port: int, opponent_port: int, num_points=7):
    inp = generate_input(gamestate=gamestate, player_port=player_port, opponent_port=opponent_port)
    dist, ind = tree.query([inp], k=num_points)
    print(dist[0][0])

    votes = []
    for i in ind[0]:
        point = list(np.array(tree.data[i]).astype(int))
        vote = map[str(point)]
        # if vote != 120:
        votes.append(vote)

    vals, counts = np.unique(votes, return_counts=True)

    # find mode
    mode_value = np.argwhere(counts == np.max(counts))

    return vals[mode_value].flatten()[0]
