import signal
import sys
import melee

class Args:
    port: int
    opponent: int
    address: str
    debug: bool
    dolphin_executable_path: str
    connect_code: str
    iso: str

class Game:
    def __init__(self, args: Args):
        self.args: Args = args
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
        self.console = melee.Console(path=args.dolphin_executable_path,
                                slippi_address=args.address,
                                logger=self.log)

        # Create our Controller object
        #   The controller is the second primary object your bot will interact with
        #   Your controller is your way of sending button presses to the game, whether
        #   virtual or physical.
        self.controller = melee.Controller(console=self.console,
                                      port=args.port,
                                      type=melee.ControllerType.STANDARD)

        self.controller_opponent = melee.Controller(console=self.console,
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
        if not self.controller_opponent.connect():
            print("ERROR: Failed to connect the controller.")
            sys.exit(-1)
        print("Controller connected")

        costume = 0
        framedata = melee.framedata.FrameData()

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
        return gamestate


    def enterMatch(self, player1: melee.Character = melee.Character.FOX, player2: melee.Character = melee.Character.FALCO, stage: melee.Stage = melee.Stage.BATTLEFIELD):
        costume = 0
        # "step" to the next frame
        gamestate = self.getState()
        # What menu are we in?
        while gamestate.menu_state not in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
            gamestate = self.getState()

            melee.MenuHelper.menu_helper_simple(gamestate,
                                                self.controller,
                                                player1,
                                                stage,
                                                self.args.connect_code,
                                                costume=costume,
                                                autostart=True,
                                                swag=False)
            melee.MenuHelper.menu_helper_simple(gamestate,
                                                self.controller_opponent,
                                                player2,
                                                stage,
                                                self.args.connect_code,
                                                costume=costume,
                                                autostart=True,
                                                swag=False)  # If we're not in game, don't log the frame
            # melee.MenuHelper.
            if self.log:
                self.log.skipframe()

    def getController(self, port) -> melee.Controller:
        if (port == self.args.port):
            return self.controller
        return self.controller_opponent