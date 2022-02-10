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
move_stick_vectors = [(float(int(i / 3) - 1), float(i % 3 - 1)) for i in range(9)]
move_stick_vectors.append((-0.5, 0))
move_stick_vectors.append((0.5, 0))

# c_stick_vectors = [(0., 0.), (-1., 0.), (0., 1.), (1., 0.), (0., -1.)]

buttons: list[melee.Button] = [None, melee.Button.BUTTON_B, melee.Button.BUTTON_A, melee.Button.BUTTON_Z,
                               melee.Button.BUTTON_L, melee.Button.BUTTON_X]

m = len(move_stick_vectors)
# c = len(c_stick_vectors)
b = len(buttons)

# move, c, button
# move_indexes = [[int(i / (b * c)) % m, int(i / b) % c, i % b] for i in range(m * c * b)]
move_indexes = [[int(i / b), i % b] for i in range(m * b)]

moveset = []
for set in move_indexes:
    moveAxis = AxisInput(axis=melee.Button.BUTTON_MAIN, x=move_stick_vectors[set[0]][0],
                         y=move_stick_vectors[set[0]][1])
    # cAxis = AxisInput(axis=melee.Button.BUTTON_C, x=c_stick_vectors[set[1]][0], y=c_stick_vectors[set[1]][1])

    button = buttons[set[1]]
    # move = Move(button=button, axes=[moveAxis, cAxis])
    move = Move(button=button, axes=[moveAxis])

    moveset.append(move)


