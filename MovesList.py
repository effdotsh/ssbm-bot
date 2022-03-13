from enum import Enum
import melee


class Move:
    def __init__(self, button, axes):
        self.axes: list[AxisInput] = axes
        self.button: melee.Button = button


class AxisInput:
    def __init__(self, axis: melee.Button, x: float, y: float):
        self.axis = axis
        self.x = x
        self.y = y


# 8 main directions, origin, and a slow walk left/right
move_stick_vectors = [(0., 0.), (-1., 0.), (1., 0.), (0., 1.), (0., -1.)]

buttons: list[melee.Button] = [None, melee.Button.BUTTON_B, melee.Button.BUTTON_A, melee.Button.BUTTON_X]

moves_list = []
for button in buttons:
    for v in move_stick_vectors:
        moveAxis = AxisInput(axis=melee.Button.BUTTON_MAIN, x=v[0], y=v[1])
        move = Move(button=button, axes=[moveAxis])
        moves_list.append(move)
moves_list = moves_list[1:]

dead_list = [melee.Action.DEAD_FLY, melee.Action.DEAD_FLY_SPLATTER,
             melee.Action.DEAD_FLY_SPLATTER_FLAT, melee.Action.DEAD_FLY_SPLATTER_FLAT_ICE,
             melee.Action.DEAD_FLY_SPLATTER_ICE, melee.Action.DEAD_FLY_STAR, melee.Action.DEAD_FLY_STAR_ICE,
             melee.Action.DEAD_LEFT, melee.Action.DEAD_RIGHT, melee.Action.DEAD_UP, melee.Action.DEAD_DOWN,
             melee.Action.ON_HALO_DESCENT]

special_fall_list = [melee.Action.SPECIAL_FALL_BACK, melee.Action.SPECIAL_FALL_FORWARD, melee.Action.LANDING_SPECIAL,
                     melee.Action.DEAD_FALL]
