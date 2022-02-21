import melee

import GameManager
import MovesList


class CharacterController:
    def __init__(self, player_port: int, opponent_port: int, game: gameManager.Game, moveslist: list):
        self.moveslist = moveslist
        self.game = game
        self.opponent_port = opponent_port
        self.player_port = player_port

        self.gamestate = self.game.get_gamestate()

    def act(self, action_index) -> None:
        controller = self.game.getController(self.player_port)
        action: movesList.Move = movesList[action_index]

        if action.button is not None:
            controller.press_button(action.button)
        for axis_movement in action.axes:
            if axis_movement.axis is not None:
                controller.tilt_analog_unit(axis_movement.axis, axis_movement.x, axis_movement.y)
