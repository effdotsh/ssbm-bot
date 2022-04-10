import melee
from tqdm import tqdm


def get_ports(gamestate: melee.GameState, player_character: melee.Character, opponent_character: melee.Character):

    ports = list(gamestate.players.keys())
    if len(ports) != 2:
        return -1, -1
    player_port = ports[0]
    opponent_port = ports[1]
    p1: melee.PlayerState = gamestate.players.get(player_port)
    p2: melee.PlayerState = gamestate.players.get(opponent_port)

    if p1.character == player_character and p2.character == opponent_character:
        player_port = ports[0]
        opponent_port = ports[1]
    elif p2.character == player_character and p1.character == opponent_character:
        player_port = ports[1]
        opponent_port = ports[0]
    else:
        player_port=-1
        opponent_port = -1
    return player_port, opponent_port


def valid_replay(replay_path, player_character: melee.Character, opponent_character: melee.Character, win_only=False):
    split = replay_path.split('.')
    if split[-1] != 'slp':
        return False

    console = melee.Console(is_dolphin=False,
                            allow_old_version=True,
                            path=replay_path)
    try:
        console.connect()
    except:
        console.stop()
        return False
    gamestate: melee.GameState = console.step()

    player_port, opponent_port = get_ports(gamestate=gamestate, player_character=player_character, opponent_character=opponent_character)

    if player_port != -1:
        if not win_only:
            console.stop()
            return True
    else:
        console.stop()
        return False

    while gamestate is not None:  # Training loop
        # p1: melee.PlayerState = gamestate.players.get(player_port)
        p2: melee.PlayerState = gamestate.players.get(opponent_port)

        if p2.stock == 0:
            console.stop()
            return True
        gamestate: melee.GameState = console.step()

    console.stop()
    return False


def filter_replays(replays, player_character: melee.Character, opponent_character: melee.Character, win_only=False):
    return [r for r in tqdm(replays) if
            valid_replay(replay_path=r, player_character=player_character, opponent_character=opponent_character,
                         win_only=win_only)]
