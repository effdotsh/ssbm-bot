import melee

dead_list = [melee.Action.DEAD_FLY, melee.Action.DEAD_FLY_SPLATTER,
             melee.Action.DEAD_FLY_SPLATTER_FLAT, melee.Action.DEAD_FLY_SPLATTER_FLAT_ICE,
             melee.Action.DEAD_FLY_SPLATTER_ICE, melee.Action.DEAD_FLY_STAR, melee.Action.DEAD_FLY_STAR_ICE,
             melee.Action.DEAD_LEFT, melee.Action.DEAD_RIGHT, melee.Action.DEAD_UP, melee.Action.DEAD_DOWN,
             melee.Action.ON_HALO_DESCENT]

special_fall_list = [melee.Action.SPECIAL_FALL_BACK, melee.Action.SPECIAL_FALL_FORWARD, melee.Action.LANDING_SPECIAL,
                     melee.Action.DEAD_FALL]

bad_moves = []

smashes = [melee.Action.DOWNSMASH, melee.Action.FSMASH_LOW, melee.Action.FSMASH_MID, melee.Action.FSMASH_HIGH,
           melee.Action.FSMASH_MID_HIGH, melee.Action.FSMASH_MID_LOW, melee.Action.UPSMASH]

firefoxing = [
    melee.Action.SWORD_DANCE_3_MID, melee.Action.SWORD_DANCE_3_LOW, melee.Action.SWORD_DANCE_3_HIGH,
    melee.Action.SWORD_DANCE_3_LOW_AIR, melee.Action.SWORD_DANCE_3_MID_AIR,
    melee.Action.SWORD_DANCE_3_HIGH_AIR, melee.Action.SWORD_DANCE_4_MID]


buttons = [melee.enums.Button.BUTTON_A, melee.enums.Button.BUTTON_B, melee.enums.Button.BUTTON_X, melee.enums.Button.BUTTON_Y]