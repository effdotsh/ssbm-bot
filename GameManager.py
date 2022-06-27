import signal
import sys
import time

import melee

import numpy as np

class Args:
    compete: bool

    load_from = -1

    port: int
    opponent: int
    address: str
    debug: bool
    dolphin_executable_path: str
    connect_code: str
    iso: str
    model_path: str
    cpu_level: int
    wandb: bool


class Game:
    def __init__(self, args: Args):
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
                                                    logger=self.log, polling_mode=False, online_delay=0,
                                                    blocking_input=True)

        # Create our Controller object
        #   The controller is the second primary object your bot will interact with
        #   Your controller is your way of sending button presses to the game, whether
        #   virtual or physical.
        self.controller = melee.Controller(console=self.console,
                                           port=args.port,
                                           type=melee.ControllerType.STANDARD)

        self.opponent_controller = melee.Controller(console=self.console,
                                                    port=args.opponent,
                                                    type=melee.ControllerType.STANDARD)

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
        if not self.controller.connect():
            print("ERROR: Failed to connect the controller.")
            sys.exit(-1)
        if not self.opponent_controller.connect():
            print("ERROR: Failed to connect the controller.")
            sys.exit(-1)
        print("Controller connected")

        costume = 0

    # This isn't necessary, but makes it so that Dolphin will get killed when you ^C
    def signal_handler(self, sig, frame):
        self.console.stop()
        if self.args.debug:
            self.log.writelog()
            print("")  # because the ^C will be on the terminal
            print("Log file created: " + self.log.filename)
        print("Shutting down cleanly...")
        sys.exit(0)

    def get_gamestate(self) -> melee.GameState:
        gamestate = self.console.step()
        while gamestate is None:
            gamestate = self.console.step()
            print("No gamestate")

        # The console object keeps track of how long your bot is taking to process frames
        #   And can warn you if it's taking too long
        # if self.console.processingtime * 1000 > 20:
        #     print("WARNING: Last frame took " + str(self.console.processingtime * 1000) + "ms to process.")

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
                gamestate = self.get_gamestate()
                moveX = np.clip(x - self.cursor_x, -1, 1)
                moveY = np.clip(y - self.cursor_y, -1, 1)

                self.cursor_x += moveX
                self.cursor_y += moveY
                self.controller.tilt_analog_unit(melee.Button.BUTTON_MAIN, moveX, moveY)
                if (x == self.cursor_x and y == self.cursor_y):
                    return

        def flick_button(button):
            gamestate = self.get_gamestate()
            self.controller.press_button(button)
            gamestate = self.get_gamestate()
            self.controller.release_all()
            gamestate = self.get_gamestate()

        def flick_axis(button, x, y):
            gamestate = self.get_gamestate()
            self.controller.tilt_analog_unit(button, x, y)
            gamestate = self.get_gamestate()
            self.controller.tilt_analog_unit(button, 0, 0)
            gamestate = self.get_gamestate()

        # Select pichu, a character needs to be selected before rules can be selected
        t = time.time()
        while time.time() - t < 1:
            gamestate = self.get_gamestate()
            melee.MenuHelper.choose_character(melee.Character.PICHU, gamestate, self.controller)
            # melee.MenuHelper.
            if self.log:
                self.log.skipframe()

        self.cursor_x = 0
        self.cursor_y = 0

        move_cursor(20, 30)

        flick_button(melee.Button.BUTTON_A)

        t = time.time()
        while time.time() - t < 1:
            gmaestate = self.get_gamestate()

        flick_axis(melee.Button.BUTTON_MAIN, -1, 0)
        flick_axis(melee.Button.BUTTON_MAIN, 0, -1)
        flick_axis(melee.Button.BUTTON_MAIN, -1, 0)
        flick_axis(melee.Button.BUTTON_MAIN, -1, 0)

        # flick_axis(melee.Button.BUTTON_MAIN, -1, 0)

        flick_button(melee.Button.BUTTON_B)
        flick_button(melee.Button.BUTTON_B)

    def enterMatch(self, player_character: melee.Character = melee.Character.FOX,
                   opponant_character: melee.Character = melee.Character.FOX,
                   stage: melee.Stage = melee.Stage.BATTLEFIELD, cpu_level: int = 0, rules:bool = True):
        self.stage = stage
        # "step" to the next frame
        gamestate = self.get_gamestate()


        # Set unlimited time
        if rules:
            self.set_rules()

        # # What menu are we in?
        t = time.time()
        # self.controller_opponent.st
        while time.time()-t < 1:
            gamestate = self.get_gamestate()

            melee.MenuHelper.menu_helper_simple(gamestate,
                                                self.controller,
                                                player_character,
                                                stage,
                                                self.args.connect_code,
                                                costume=0,
                                                autostart=False,
                                                swag=False)
        c = 0
        while gamestate.menu_state not in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
            c+=1
            gamestate = self.get_gamestate()
            melee.MenuHelper.menu_helper_simple(gamestate,
                                                self.controller,
                                                player_character,
                                                stage,
                                                self.args.connect_code,
                                                costume=0,
                                                autostart=False,
                                                swag=self.args.connect_code =="")
            if self.args.connect_code =="":
                print('urmom')
                melee.MenuHelper.menu_helper_simple(gamestate,
                                                    self.opponent_controller,
                                                    opponant_character,
                                                    stage,
                                                    self.args.connect_code,
                                                    cpu_level=cpu_level,
                                                    costume=0,
                                                    autostart=True,
                                                    swag=True)
            elif c > 60*3:
                self.controller.press_button(melee.Button.BUTTON_START)
                break
        while gamestate.menu_state not in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
            pass
        # p1: melee.PlayerState = gamestate.players.get(self.controller.port)
            # p2: melee.PlayerState = gamestate.players.get(self.controller_opponent.port)
            # if p1.coin_down and p2.coin_down:
            #     self.controller.press_button(melee.Button.BUTTON_START)
        # self.first_match_started = True

    def getController(self, port) -> melee.Controller:
        if (port == self.args.port):
            return self.controller
        return self.opponent_controller
