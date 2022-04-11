import Args
import GameManager
import melee
import platform

import os


args = Args.get_args()
if __name__ == '__main__':
    character = melee.Character.FOX
    opponent = melee.Character.CPTFALCON if not args.compete else character

    if not os.path.isdir(f'{args.model_path}/{character.name}'):
        os.makedirs(f'{args.model_path}/{character.name}')

    game = GameManager.Game(args)
    game.enterMatch(cpu_level=args.cpu_level if not args.compete else 0, opponant_character=opponent,
                    player_character=character,
                    stage=melee.Stage.FINAL_DESTINATION)
