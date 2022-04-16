import Args
import GameManager
import melee
import platform

import os
import torch
import trainer
import numpy as np

from encoder import decode_from_number

args = Args.get_args()

if __name__ == '__main__':
    character = melee.Character.JIGGLYPUFF
    opponent = melee.Character.CPTFALCON if not args.compete else character
    stage = melee.Stage.FINAL_DESTINATION

    model = torch.load(f"models/{character.name}_v_{opponent.name}_on_{stage.name}")
    # print(loaded)
    # model = network.Network(19*2, 10)
    # model.load_state_dict(loaded)
    game = GameManager.Game(args)
    game.enterMatch(cpu_level=9, opponant_character=opponent,
                    player_character=character,
                    stage=stage, rules=False)

    num_buttons = len(trainer.buttons) + 1
    axis_size = 3
    num_c = 5
    maxes = [axis_size, axis_size, num_c, num_buttons]
    with torch.no_grad():
        while True:
            gamestate = game.get_gamestate()

            inp = trainer.generate_input(gamestate, 1, 2)
            out = model(torch.Tensor(inp)).detach().numpy()

            action = np.argmax(out)
            move_x, move_y, c, button = decode_from_number(action, maxes)
            print(action)
            # print(button)
            # print(move_x)
            # print(move_y)
            # print(c)
            # print(out)
            print('----------')
            # print(trainer.buttons)
            if button > 0:
                game.controller.press_button(trainer.buttons[button - 1][0])

            if c == 0:
                game.controller.tilt_analog(melee.Button.BUTTON_C, 0.5, 0.5)
            else:
                if c == 1:
                    game.controller.tilt_analog(melee.Button.BUTTON_C, 0, 0.5)
                elif c == 2:
                    game.controller.tilt_analog(melee.Button.BUTTON_C, 1, 0.5)
                elif c == 3:
                    game.controller.tilt_analog(melee.Button.BUTTON_C, 0.5, 0)
                elif c == 4:
                    game.controller.tilt_analog(melee.Button.BUTTON_C, 0.5, 1)
                for i in range(5):
                    gamestate = game.get_gamestate()

            game.controller.tilt_analog(melee.Button.BUTTON_MAIN, move_x / 2, move_y / 2)
            gamestate = game.get_gamestate()
            for b in trainer.buttons:
                game.controller.release_button(b[0])
            game.controller.tilt_analog(melee.Button.BUTTON_C, 0.5, 0.5)

            # game.controller.release_all()
