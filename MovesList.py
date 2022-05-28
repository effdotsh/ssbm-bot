import melee

dead_list = [melee.Action.DEAD_FLY, melee.Action.DEAD_FLY_SPLATTER,
             melee.Action.DEAD_FLY_SPLATTER_FLAT, melee.Action.DEAD_FLY_SPLATTER_FLAT_ICE,
             melee.Action.DEAD_FLY_SPLATTER_ICE, melee.Action.DEAD_FLY_STAR, melee.Action.DEAD_FLY_STAR_ICE,
             melee.Action.DEAD_LEFT, melee.Action.DEAD_RIGHT, melee.Action.DEAD_UP, melee.Action.DEAD_DOWN,
             melee.Action.ON_HALO_DESCENT]

special_fall_list = [melee.Action.SPECIAL_FALL_BACK, melee.Action.SPECIAL_FALL_FORWARD, melee.Action.LANDING_SPECIAL,
                     melee.Action.DEAD_FALL]


bad_moves = []