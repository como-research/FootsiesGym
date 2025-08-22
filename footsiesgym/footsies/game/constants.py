import dataclasses


@dataclasses.dataclass
class EnvActions:
    NONE = 0
    BACK = 1
    FORWARD = 2
    ATTACK = 3
    BACK_ATTACK = 4
    FORWARD_ATTACK = 5
    SPECIAL_CHARGE = 6


@dataclasses.dataclass
class GameActions:
    NONE = 0
    LEFT = 1
    RIGHT = 2
    ATTACK = 3
    LEFT_ATTACK = 4
    RIGHT_ATTACK = 5


@dataclasses.dataclass
class ActionBits:
    NONE: int = 0
    LEFT: int = 1 << 0
    RIGHT: int = 1 << 1
    ATTACK: int = 1 << 2
    LEFT_ATTACK: int = LEFT | ATTACK
    RIGHT_ATTACK: int = RIGHT | ATTACK


ACTION_TO_BITS = {
    GameActions.NONE: ActionBits.NONE,
    GameActions.LEFT: ActionBits.LEFT,
    GameActions.RIGHT: ActionBits.RIGHT,
    GameActions.ATTACK: ActionBits.ATTACK,
    GameActions.LEFT_ATTACK: ActionBits.LEFT_ATTACK,
    GameActions.RIGHT_ATTACK: ActionBits.RIGHT_ATTACK,
}


@dataclasses.dataclass
class ActionID:
    STAND = 0
    FORWARD = 1
    BACKWARD = 2
    DASH_FORWARD = 10
    DASH_BACKWARD = 11
    N_ATTACK = 100
    B_ATTACK = 105
    N_SPECIAL = 110
    B_SPECIAL = 115
    DAMAGE = 200
    GUARD_M = 301
    GUARD_STAND = 305
    GUARD_CROUCH = 306
    GUARD_BREAK = 310
    GUARD_PROXIMITY = 350
    DEAD = 500
    WIN = 510


FOOTSIES_ACTION_IDS = {
    "STAND": ActionID.STAND,
    "FORWARD": ActionID.FORWARD,
    "BACKWARD": ActionID.BACKWARD,
    "DASH_FORWARD": ActionID.DASH_FORWARD,
    "DASH_BACKWARD": ActionID.DASH_BACKWARD,
    "N_ATTACK": ActionID.N_ATTACK,
    "B_ATTACK": ActionID.B_ATTACK,
    "N_SPECIAL": ActionID.N_SPECIAL,
    "B_SPECIAL": ActionID.B_SPECIAL,
    "DAMAGE": ActionID.DAMAGE,
    "GUARD_M": ActionID.GUARD_M,
    "GUARD_STAND": ActionID.GUARD_STAND,
    "GUARD_CROUCH": ActionID.GUARD_CROUCH,
    "GUARD_BREAK": ActionID.GUARD_BREAK,
    "GUARD_PROXIMITY": ActionID.GUARD_PROXIMITY,
    "DEAD": ActionID.DEAD,
    "WIN": ActionID.WIN,
}
