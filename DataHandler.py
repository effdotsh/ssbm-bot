import numpy as np
import melee
from tensorflow import keras

import MovesList

framedata = melee.FrameData()

low_analog = 0.2
high_analog = 0.8


def controller_states_different(new_player: melee.PlayerState, old_player: melee.PlayerState):
    new: melee.ControllerState = new_player.controller_state
    old: melee.ControllerState = old_player.controller_state

    # if generate_output(new_player) == generate_output(old_player):
    #     return False

    for btns in MovesList.buttons:
        # for b in melee.enums.Button:
        for b in btns:
            if new.button.get(b) != old.button.get(b) and new.button.get(b):
                return True

    if new.c_stick[0] < low_analog and old.c_stick[0] >= low_analog:
        return True

    if new.c_stick[0] > high_analog and old.c_stick[0] <= high_analog:
        return True

    if new.c_stick[1] < low_analog and old.c_stick[1] >= low_analog:
        return True

    if new.c_stick[1] > high_analog and old.c_stick[1] <= high_analog:
        return True

    if new.main_stick[0] < low_analog and old.main_stick[0] >= low_analog:
        return True

    if new.main_stick[0] > high_analog and old.main_stick[0] <= high_analog:
        return True

    if new.main_stick[1] < low_analog and old.main_stick[1] >= low_analog:
        return True

    if new.main_stick[1] > high_analog and old.main_stick[1] <= high_analog:
        return True

    return False

    # return generate_output(new) != generate_output(old)


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
    y = player.position.y / 50
    shield = player.shield_strength / 60

    percent = player.percent / 100
    vel_y = (player.speed_y_self + player.speed_y_attack)
    vel_x = (player.speed_x_attack + player.speed_air_x_self + player.speed_ground_x_self)
    is_attacking = 1 if framedata.is_attack(player.character, player.action) else 0

    # return [x, y, shield, percent, vel_x, vel_y, is_attacking]
    edge = melee.EDGE_POSITION.get(gamestate.stage)

    offstage = 1 if abs(player.position.x) > edge - 1 else -1
    tumbling = 1 if player.action in [melee.Action.TUMBLING] else -1
    on_ground = 1 if player.on_ground else -1

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

    stock = player.stock
    return [
        tumbling,
        offstage,
        special_fall,
        is_dead,
        shield, on_ground, is_attacking,
        x, y,
        vel_x, vel_y,
        # percent,
        facing,
        in_hitstun,
        is_invulnerable,
        jumps_left,
        attack_windup, attack_active, attack_cooldown,
        is_bmove,
        (abs(player.position.x) - edge) / 20,
        # stock
    ]


def generate_input(gamestate: melee.GameState, player_port: int, opponent_port: int):
    player: melee.PlayerState = gamestate.players.get(player_port)
    opponent: melee.PlayerState = gamestate.players.get(opponent_port)
    if player is None or opponent is None:
        return None

    direction = 1 if player.position.x < opponent.position.x else -1

    firefoxing = 1 if player.character in [melee.Character.FOX,
                                           melee.Character.FALCO] and player.action in MovesList.firefoxing else -1

    obs = [
        (player.position.x - opponent.position.x) / 20, (player.position.y - opponent.position.y) / 10,
        # firefoxing, direction, 1
        1 if player.position.x > opponent.position.x else -1,
        1 if player.position.y > opponent.position.y else -1,

        abs(player.position.x - opponent.position.x) - 3.5
    ]
    obs += get_player_obs(player, gamestate)
    obs += get_player_obs(opponent, gamestate)

    return np.array(obs).flatten()


def generate_output(player: melee.PlayerState):

    controller: melee.ControllerState = player.controller_state
    action_counter = 0


    # Jump
    if controller.button.get(melee.Button.BUTTON_X) or controller.button.get(melee.Button.BUTTON_Y):
        return action_counter
    action_counter += 1

    # Shield
    if controller.button.get(melee.Button.BUTTON_L) or controller.button.get(melee.Button.BUTTON_R):
        return action_counter
    action_counter += 1

    # grab
    if controller.button.get(melee.Button.BUTTON_Z):
        return action_counter
    action_counter += 1

    # C Stick
    if controller.c_stick[0] < low_analog:
        return action_counter
    if controller.c_stick[0] > high_analog:
        return action_counter + 1
    if controller.c_stick[1] < low_analog:
        return action_counter + 2
    if controller.c_stick[1] > high_analog:
        return action_counter + 3
    action_counter += 4

    # Either move stick pressed with or without B
    b_used = False
    a_used = False
    if controller.button.get(melee.Button.BUTTON_B):
        action_counter += 4
        b_used = True
    elif controller.button.get(melee.Button.BUTTON_A):
        action_counter += 9
        a_used = True
    # Move Stick
    if controller.main_stick[0] < low_analog:
        return action_counter
    elif controller.main_stick[0] > high_analog:
        return action_counter + 1
    elif controller.main_stick[1] < low_analog:
        return action_counter + 2
    elif controller.main_stick[1] > high_analog:
        return action_counter + 3
    action_counter += 4

    if player is not None:
        if player.character in [melee.Character.FOX, melee.Character.FALCO]:
            if b_used:    # Don't use lasers as fox/falco
                return -1

    if not b_used and not a_used:
        return -1
    return action_counter


def decode_from_model(action: np.ndarray, player: melee.PlayerState = None):
    action = action[0]
    # if player is not None and player.position.y > 0 and abs(player.position.x) < 100:
    if player is not None and player.position.y > 0:
        reduce = [7,8,9,10,1]
        for i in reduce:
            action[i] /= 5.6

        if player.character in [melee.Character.FOX, melee.Character.FALCO, melee.Character.MARTH]:
            action[14]/=100
    action[0]/=4
    if not player.on_ground:
        action[1] /= 100
    # action[2] /= 100

    a = np.argmax(action)
    # [[BUTTON_X, BUTTON_B, BUTTON_L, BUTTON_A, BUTTON_Z], move_x, move_y, c_x, c_y]
    if a == 0:  # jump
        return [[1, 0, 0, 0, 0], 0, 0, 0, 0]
    elif a == 1:  # shield
        return [[0, 0, 1, 0, 0], 0, 0, 0, 0]
    elif a == 2:  # grab
        return [[0, 0, 0, 0, 1], 0, 0, 0, 0]

    # c_stick
    elif a == 3:
        return [[0, 0, 0, 0, 0], 0, 0, -1, 0]
    elif a == 4:
        return [[0, 0, 0, 0, 0], 0, 0, 1, 0]
    elif a == 5:
        return [[0, 0, 0, 0, 0], 0, 0, 0, -1]
    elif a == 6:
        return [[0, 0, 0, 0, 0], 0, 0, 0, 1]


    # move stick
    elif a == 7:
        return [[0, 0, 0, 0, 0], -1, 0, 0, 0]
    elif a == 8:
        return [[0, 0, 0, 0, 0], 1, 0, 0, 0]
    elif a == 9:
        return [[0, 0, 0, 0, 0], 0, -1, 0, 0]
    elif a == 10:
        return [[0, 0, 0, 0, 0], 0, 1, 0, 0]

    # b moves
    elif a == 11:
        return [[0, 1, 0, 0, 0], -1, 0, 0, 0]
    elif a == 12:
        return [[0, 1, 0, 0, 0], 1, 0, 0, 0]
    elif a == 13:
        return [[0, 1, 0, 0, 0], 0, -1, 0, 0]
    elif a == 14:
        # b reverse not possible in the action space
        if player is not None and player.character == melee.enums.Character.MARTH:
            vel_y = player.speed_y_self + player.speed_y_attack

            if player.jumps_left == 0 and player.position.y < -20 and vel_y < 0:
                x = np.sign(player.position.x)
                return [[0, 1, 0, 0, 0], -0.6 * x, 0.85, 0, 0]
        return [[0, 1, 0, 0, 0], 0, 1, 0, 0]
    elif a == 15:
        return [[0, 1, 0, 0, 0], 0, 0, 0, 0]

    # a moves
    elif a == 16:
        return [[0, 0, 0, 1, 0], -1, 0, 0, 0]
    elif a == 17:
        return [[0, 0, 0, 1, 0], 1, 0, 0, 0]
    elif a == 18:
        return [[0, 0, 0, 1, 0], 0, -1, 0, 0]
    elif a == 19:
        return [[0, 0, 0, 1, 0], 0, 1, 0, 0]
    elif a == 20:
        return [[0, 0, 0, 1, 0], 0, 0, 0, 0]

    print('NO ACTION FOUND !!!!')
    return [[0, 0, 0, 0, 0], 0, 0, 0, 0]
