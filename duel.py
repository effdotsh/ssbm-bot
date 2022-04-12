import Args
import GameManager
import melee
import platform

import os
from tensorflow import keras

import trainer
import numpy as np

args = Args.get_args()
if __name__ == '__main__':
    character = melee.Character.FALCO
    opponent = melee.Character.FOX if not args.compete else character

    model = keras.models.load_model("model/final/")
    print(model.summary())
    game = GameManager.Game(args)
    game.enterMatch(cpu_level=1, opponant_character=opponent,
                    player_character=character,
                    stage=melee.Stage.BATTLEFIELD)

    while True:
        gamestate = game.get_gamestate()
        inp = trainer.generate_input(gamestate, 1, 2)
        out = model.predict(np.array([inp]))[0]
        print(out)
        for i in range(len(trainer.buttons)):
            if out[i] > 0.3:
                game.controller.press_button(trainer.buttons[i][0])
        if out[-5] > 0.3:
            game.controller.press_shoulder(melee.Button.BUTTON_R, 1)
        game.controller.tilt_analog_unit(melee.Button.BUTTON_MAIN, np.sign(out[-2]) if abs(out[-2]) > 0.2 else 0,
                                         np.sign(out[1]) if abs(out[-1]) > 0.2 else 0)
        game.controller.tilt_analog_unit(melee.Button.BUTTON_C, np.sign(out[-4]) if abs(out[-4]) > 0.5 else 0,
                                         np.sign(out[-3]) if abs(out[-3]) > 0.5 else 0)
