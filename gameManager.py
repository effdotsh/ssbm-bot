import signal
import sys
import time

import melee

import utils
from utils import clamp

from queue import Queue

class Args:
    port: int
    opponent: int
    address: str
    debug: bool
    dolphin_executable_path: str
    connect_code: str
    iso: str


class Action:
    def __init__(self, button: melee.Button = None, joystick: melee.Button = None, num_frames=0, force_end=False, x=0,
                 y=0):
        # if no button/joystick is inputted then no action needs to be complete
        self.complete: bool = button is None and joystick is None

        self.joystick = joystick
        self.button = button

        self.num_frames = num_frames
        self.force_end = force_end

        self.x = x
        self.y = y
        self.frame_counter = 0


class Game:
    def __init__(self, args: Args, p1_queue: Queue):
        self.p1_queue = p1_queue
        self.args: Args = args

        self.first_match_started = False
        # This logger object is useful for retroactively debugging issues in your bot
        #   You can write things to it each frame, and it will create a CSV file describing the match
        self.log = None
        if args.debug:
            self.log = melee.Logger()

        # Create our Console object.
        #   This will be one of the primary objects that we will interface with.
        #   The Console represents the virtual or hardware system Melee is playing on.
        #   Through this object, we can get "GameState" objects per-frame so that your
        #       bot can actually "see" what's happening in the game
        self.console: melee.Console = melee.Console(path=args.dolphin_executable_path,
                                                    slippi_address=args.address,
                                                    logger=self.log)

        # Create our Controller object
        #   The controller is the second primary object your bot will interact with
        #   Your controller is your way of sending button presses to the game, whether
        #   virtual or physical.
        self.p1_controller = melee.Controller(console=self.console,
                                              port=args.port,
                                              type=melee.ControllerType.STANDARD)

        self.p2_controller = melee.Controller(console=self.console,
                                              port=args.opponent,
                                              type=melee.ControllerType.STANDARD)

        self.controllers: list[melee.Controller] = [self.p1_controller, self.p2_controller]

        signal.signal(signal.SIGINT, self.signal_handler)

        # Run the console
        self.console.run(iso_path=self.args.iso)

        # Connect to the console
        print("Connecting to console...")
        if not self.console.connect():
            print("ERROR: Failed to connect to the console.")
            sys.exit(-1)
        print("Console connected")

        # Plug our controller in
        #   Due to how named pipes work, this has to come AFTER running dolphin
        #   NOTE: If you're loading a movie file, don't connect the controller,
        #   dolphin will hang waiting for input and never receive it
        print("Connecting controller to console...")
        if not self.p1_controller.connect():
            print("ERROR: Failed to connect the controller.")
            sys.exit(-1)
        if not self.p2_controller.connect():
            print("ERROR: Failed to connect the controller.")
            sys.exit(-1)
        print("Controller connected")

        self.player_actions: list[Action] = [Action(), Action()]
        self.gamestate = self.getState()

    # This isn't necessary, but makes it so that Dolphin will get killed when you ^C
    def signal_handler(self, sig, frame):
        self.console.stop()
        if self.args.debug:
            self.log.writelog()
            print("")  # because the ^C will be on the terminal
            print("Log file created: " + self.log.filename)
        print("Shutting down cleanly...")
        sys.exit(0)

    def getState(self) -> melee.GameState:
        gamestate = self.console.step()
        if gamestate is None:
            print("No gamestate")

        # The console object keeps track of how long your bot is taking to process frames
        #   And can warn you if it's taking too long
        if self.console.processingtime * 1000 > 12:
            print("WARNING: Last frame took " + str(self.console.processingtime * 1000) + "ms to process.")

        if gamestate.menu_state not in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH] and self.first_match_started:
            while gamestate.menu_state == gamestate.menu_state.POSTGAME_SCORES:
                melee.MenuHelper.skip_postgame()
            while gamestate.menu_state not in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
                self.enterMatch()
        return gamestate

    def set_rules(self):
        # Sets rules to be time with no time limit
        def move_cursor(x, y):
            while True:
                gamestate = self.getState()
                moveX = clamp(x - self.cursor_x, -1, 1)
                moveY = clamp(y - self.cursor_y, -1, 1)

                print(moveX)
                self.cursor_x += moveX
                self.cursor_y += moveY
                self.p1_controller.tilt_analog_unit(melee.Button.BUTTON_MAIN, moveX, moveY)
                if (x == self.cursor_x and y == self.cursor_y):
                    return

        def flick_button(button):
            gamestate = self.getState()
            self.p1_controller.press_button(button)
            gamestate = self.getState()
            self.p1_controller.release_all()
            gamestate = self.getState()

        def flick_axis(button, x, y):
            gamestate = self.getState()
            self.p1_controller.tilt_analog_unit(button, x, y)
            gamestate = self.getState()
            t = time.time()
            while time.time() - t < 0.2:
                gamestate = self.getState()
            self.p1_controller.tilt_analog_unit(button, 0, 0)
            gamestate = self.getState()

        # Select pichu, a character needs to be selected before rules can be selected
        t = time.time()
        while time.time() - t < 1:
            gamestate = self.getState()
            melee.MenuHelper.choose_character(melee.Character.PICHU, gamestate, self.p1_controller)
            # melee.MenuHelper.
            if self.log:
                self.log.skipframe()

        self.cursor_x = 0
        self.cursor_y = 0

        move_cursor(20, 30)

        flick_button(melee.Button.BUTTON_A)

        time.sleep(1)
        flick_axis(melee.Button.BUTTON_MAIN, -1, 0)
        flick_axis(melee.Button.BUTTON_MAIN, 0, -1)
        flick_axis(melee.Button.BUTTON_MAIN, -1, 0)
        flick_axis(melee.Button.BUTTON_MAIN, -1, 0)

        flick_button(melee.Button.BUTTON_B)
        time.sleep(0.3)
        flick_button(melee.Button.BUTTON_B)

    def enterMatch(self, player_character: melee.Character = melee.Character.FOX,
                   opponant_character: melee.Character = melee.Character.FOX,
                   stage: melee.Stage = melee.Stage.BATTLEFIELD, cpu_level: int = 0):
        costume = 0
        # "step" to the next frame
        gamestate = self.getState()

        # Set unlimited time
        self.set_rules()

        # # What menu are we in?
        t = time.time()
        # self.controller_opponent.st
        while gamestate.menu_state not in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
            gamestate = self.getState()

            melee.MenuHelper.menu_helper_simple(gamestate,
                                                self.p1_controller,
                                                player_character,
                                                stage,
                                                self.args.connect_code,
                                                costume=costume,
                                                autostart=False,
                                                swag=False)
            melee.MenuHelper.menu_helper_simple(gamestate,
                                                self.p2_controller,
                                                opponant_character,
                                                stage,
                                                self.args.connect_code,
                                                cpu_level=cpu_level,
                                                costume=costume,
                                                autostart=True,
                                                swag=False)  # If we're not in game, don't log the frame
            # melee.MenuHelper.
            if self.log:
                self.log.skipframe()

        self.gamestate = self.getState()

        # self.first_match_started = True

    def getController(self, port) -> melee.Controller:
        if (port == self.args.port):
            return self.p1_controller
        return self.p2_controller

    def setAction(self, player_index: int, action: Action):
        if self.player_actions[player_index].complete:
            self.player_actions[player_index] = action

    def gameLoop(self):
        self.gamestate = self.getState()

        # try:
        #     self.player_actions[0] = self.p1_queue.get(block=False, timeout=0.05)
        # except:
        #     pass

        for controller, action in zip(self.controllers, self.player_actions):
            if not action.complete:
                player: melee.PlayerState = self.gamestate.players.get(controller.port)

                action.frame_counter += 1
                if action.joystick is not None:
                    controller.tilt_analog_unit(action.joystick, action.x, action.y)
                if action.button is not None:
                    controller.press_button(action.button)

                if (action.frame_counter >= action.num_frames) and player.action not in utils.dead_list and player.action not in utils.attacking_list:

                    action.complete = True
                    self.p1_queue.put(True)
                    controller.release_all()

    def getGamestate(self):
        return self.gamestate
