import Args
import GameManager
import melee
import platform

import os
import torch
import trainer
import numpy as np

args = Args.get_args()
if __name__ == '__main__':
    character = melee.Character.CPTFALCON
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

    with torch.no_grad():
        while True:
            gamestate = game.get_gamestate()
            game.controller.release_all()

            inp = trainer.generate_input(gamestate, 1, 2)
            out = model(torch.Tensor(inp)).detach().numpy()
            print(out)
            for i in range(len(trainer.buttons)-1):
                if out[i] > 0.3:
                    game.controller.press_button(trainer.buttons[i][0])
                    print(trainer.buttons[i][0])
                    break
            if out[-5] > 0.3:
                game.controller.press_shoulder(melee.Button.BUTTON_R, 1)
            game.controller.tilt_analog_unit(melee.Button.BUTTON_MAIN, np.sign(out[-2]) if abs(out[-2]) > 0.2 else 0,
                                             np.sign(out[1]) if abs(out[-1]) > 0.2 else 0)
            game.controller.tilt_analog_unit(melee.Button.BUTTON_C, np.sign(out[-4]) if abs(out[-4]) > 0.5 else 0,
                                             np.sign(out[-3]) if abs(out[-3]) > 0.5 else 0)

            # game.controller.flush()
            gamestate = game.get_gamestate()
            game.controller.release_all()

            # gamestate = game.get_gamestate()
