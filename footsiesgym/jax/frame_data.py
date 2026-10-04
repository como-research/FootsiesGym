"""Fighter F00's frame data from the Unity project, as read-only lookup tables.

Loads ``frame_data.json`` (extracted from the Unity assets by
``scripts/extract_unity_frame_data.py``) and flattens the per-action lists
into dense tables indexed by ``[action, frame]``. The data is fixed: every
table is a read-only numpy array, and ``tables()`` returns them as JAX
arrays for use inside traced code.

Actions are referred to by *index* into ``ACTION_IDS``, which is sorted
ascending and therefore matches the one-hot order of
``constants.FOOTSIES_ACTION_IDS`` used by the observation encoders. Attacks
are referred to by index into ``ATTACK_NAMES``.

All C# list lookups are reproduced exactly:
  * ``GetHitboxData``/``GetHurtboxData`` return every active box in asset
    order -> fixed slots in asset order plus a validity mask.
  * ``GetPushboxData``/``GetMovementData`` return the *first* active entry.
  * ``RequestAction``'s cancel scan only ever stores ``bufferActionID`` (its
    return value is unused) -> one boolean per (action, frame, requested).
"""

import json
import pathlib
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

f32 = np.float32

_DATA = json.loads((pathlib.Path(__file__).parent / "frame_data.json").read_text())
_FIGHTER = _DATA["fighter"]
_ACTIONS = sorted(_DATA["actions"], key=lambda a: a["action_id"])
_ATTACKS = _DATA["attacks"]

# ── Constants ────────────────────────────────────────────────────────────

FIXED_DELTA_TIME = f32(1.0) / f32(60.0)  # Fighter.FIXED_DELTA_TIME = 1.0f / 60.0f
BATTLE_AREA_WIDTH = f32(10.0)  # BattleSimulation.battleAreaWidth
MAX_SPRITE_SHAKE_FRAME = 6  # Fighter.maxSpriteShakeFrame
INPUT_LEFT, INPUT_RIGHT, INPUT_ATTACK = 1, 2, 4  # InputDefine
ACTION_TYPE_GUARD = 3  # ActionType.Guard

START_GUARD_HEALTH = _FIGHTER["start_guard_health"]
FORWARD_MOVE_SPEED = f32(_FIGHTER["forward_move_speed"])
BACKWARD_MOVE_SPEED = f32(_FIGHTER["backward_move_speed"])
DASH_ALLOW_FRAME = _FIGHTER["dash_allow_frame"]
SPECIAL_ATTACK_HOLD_FRAME = _FIGHTER["special_attack_hold_frame"]
CAN_CANCEL_ON_WHIFF = _FIGHTER["can_cancel_on_whiff"]
BASE_HURTBOX_RECT = np.array(_FIGHTER["base_hurtbox_rect"], dtype=f32)
BASE_PUSHBOX_RECT = np.array(_FIGHTER["base_pushbox_rect"], dtype=f32)

# Only the most recent inputs are ever read: the special-attack check reads
# input[0 .. hold-1] and the dash checks at most input[2 * dashAllowFrame - 2].
# (The C# buffer is 180 long; the rest is never read.)
INPUT_HISTORY = max(SPECIAL_ATTACK_HOLD_FRAME, 2 * DASH_ALLOW_FRAME - 1)

ACTION_IDS = np.array([a["action_id"] for a in _ACTIONS], dtype=np.int32)
ACTION_NAMES = tuple(a["name"] for a in _ACTIONS)
NUM_ACTIONS = len(_ACTIONS)
ATTACK_NAMES = tuple(a["attackName"] for a in _ATTACKS)
_ACTION_INDEX = {int(aid): i for i, aid in enumerate(ACTION_IDS)}
_ATTACK_INDEX = {a["attackID"]: i for i, a in enumerate(_ATTACKS)}

# Action indices referenced by the game logic (CommonActionID)
STAND = _ACTION_INDEX[0]
FORWARD = _ACTION_INDEX[1]
BACKWARD = _ACTION_INDEX[2]
DASH_FORWARD = _ACTION_INDEX[10]
DASH_BACKWARD = _ACTION_INDEX[11]
N_ATTACK = _ACTION_INDEX[100]
B_ATTACK = _ACTION_INDEX[105]
N_SPECIAL = _ACTION_INDEX[110]
B_SPECIAL = _ACTION_INDEX[115]
GUARD_BREAK = _ACTION_INDEX[310]
GUARD_PROXIMITY = _ACTION_INDEX[350]
WIN = _ACTION_INDEX[510]

# Frame axis: every frame an action can be queried at (0..frameCount) and
# every authored range end. Lookups clip the frame to this range.
NUM_FRAMES = 1 + max(
    [a["frame_count"] for a in _ACTIONS]
    + [
        e["frames"][1]
        for a in _ACTIONS
        for key in ("hitboxes", "hurtboxes", "pushboxes", "movements", "cancels")
        for e in a[key]
    ]
)
MAX_HITBOXES = max(len(a["hitboxes"]) for a in _ACTIONS)
MAX_HURTBOXES = max(len(a["hurtboxes"]) for a in _ACTIONS)


# ── Tables ───────────────────────────────────────────────────────────────


class FrameData(NamedTuple):
    # Per action, (NUM_ACTIONS,): ActionData fields
    frame_count: np.ndarray
    is_loop: np.ndarray
    loop_from_frame: np.ndarray
    always_cancelable: np.ndarray
    action_type: np.ndarray
    # Per action and frame, (NUM_ACTIONS, NUM_FRAMES, ...); rects are x, y, w, h
    hitbox_rect: np.ndarray  # (..., MAX_HITBOXES, 4)
    hitbox_valid: np.ndarray
    hitbox_attack: np.ndarray  # attack index
    hitbox_proximity: np.ndarray
    hurtbox_rect: np.ndarray  # (..., MAX_HURTBOXES, 4)
    hurtbox_valid: np.ndarray
    pushbox_rect: np.ndarray  # (..., 4)
    has_movement: np.ndarray
    movement_velocity: np.ndarray
    # velocity * FIXED_DELTA_TIME as float32 (hi, lo, tiny); see exact_delta()
    movement_delta: np.ndarray  # (..., 3)
    cancel_buffer: np.ndarray  # (..., NUM_ACTIONS): requesting [-1] buffers it
    # Per attack, (len(ATTACK_NAMES),): AttackData fields (actions as indices)
    attack_damage_action: np.ndarray
    attack_guard_action: np.ndarray
    attack_number_of_hit: np.ndarray
    attack_vital_damage: np.ndarray
    attack_guard_damage: np.ndarray
    attack_hit_stun: np.ndarray
    attack_guard_stun: np.ndarray
    attack_guard_break_stun: np.ndarray
    # Fighter.GetSpecialAttackProgress() for 0..hold consecutive held frames.
    # Precomputed with IEEE numpy division: XLA does not guarantee correctly
    # rounded float32 division (on CPU it multiplies by an approximate
    # reciprocal).
    special_attack_progress: np.ndarray


def exact_delta(speed) -> np.ndarray:
    """``speed * FIXED_DELTA_TIME`` as float32 ``(hi, lo, tiny)``.

    The product of two float32 values has at most 48 significant bits, so it
    is exact in float64 and splits exactly into ``hi + lo``. ``tiny`` is half
    a float64 ulp of the product: Mono rounds ``position + product`` to
    float64 first, which returns the product itself, then ``hi``, whenever
    ``|position| <= tiny`` (see ``fighter._move``).
    """
    product = np.float64(f32(speed)) * np.float64(FIXED_DELTA_TIME)
    hi = f32(product)
    lo = f32(product - np.float64(hi))
    assert np.float64(hi) + np.float64(lo) == product
    tiny = f32(np.ldexp(1.0, np.frexp(product)[1] - 54)) if product else f32(0)
    return np.array([hi, lo, tiny], dtype=f32)


def _active(entry, frame):
    lo, hi = entry["frames"]
    return lo <= frame <= hi


def _build() -> FrameData:
    per_action = lambda key, dtype: np.array([a[key] for a in _ACTIONS], dtype=dtype)
    per_attack = lambda key: np.array([a[key] for a in _ATTACKS], dtype=np.int32)
    shape = (NUM_ACTIONS, NUM_FRAMES)
    t = dict(
        hitbox_rect=np.zeros(shape + (MAX_HITBOXES, 4), f32),
        hitbox_valid=np.zeros(shape + (MAX_HITBOXES,), bool),
        hitbox_attack=np.zeros(shape + (MAX_HITBOXES,), np.int32),
        hitbox_proximity=np.zeros(shape + (MAX_HITBOXES,), bool),
        hurtbox_rect=np.zeros(shape + (MAX_HURTBOXES, 4), f32),
        hurtbox_valid=np.zeros(shape + (MAX_HURTBOXES,), bool),
        pushbox_rect=np.zeros(shape + (4,), f32),
        has_movement=np.zeros(shape, bool),
        movement_velocity=np.zeros(shape, f32),
        movement_delta=np.zeros(shape + (3,), f32),
        cancel_buffer=np.zeros(shape + (NUM_ACTIONS,), bool),
    )
    for ai, a in enumerate(_ACTIONS):
        for fr in range(NUM_FRAMES):
            for k, h in enumerate(a["hitboxes"]):
                if _active(h, fr):
                    t["hitbox_rect"][ai, fr, k] = np.array(h["rect"], f32)
                    t["hitbox_valid"][ai, fr, k] = True
                    t["hitbox_attack"][ai, fr, k] = _ATTACK_INDEX[h["attack_id"]]
                    t["hitbox_proximity"][ai, fr, k] = h["proximity"]
            for k, h in enumerate(a["hurtboxes"]):
                if _active(h, fr):
                    rect = (
                        BASE_HURTBOX_RECT
                        if h["use_base_rect"]
                        else np.array(h["rect"], f32)
                    )
                    t["hurtbox_rect"][ai, fr, k] = rect
                    t["hurtbox_valid"][ai, fr, k] = True
            push = next((p for p in a["pushboxes"] if _active(p, fr)), None)
            if push is not None:
                rect = (
                    BASE_PUSHBOX_RECT
                    if push["use_base_rect"]
                    else np.array(push["rect"], f32)
                )
                t["pushbox_rect"][ai, fr] = rect
            elif fr < a["frame_count"]:
                # Unity dereferences GetPushboxData() unconditionally.
                raise ValueError(f"{a['name']} has no pushbox on frame {fr}")
            move = next((m for m in a["movements"] if _active(m, fr)), None)
            if move is not None:
                t["has_movement"][ai, fr] = True
                velocity = f32(move["velocity_x"])
                t["movement_velocity"][ai, fr] = velocity
                t["movement_delta"][ai, fr] = exact_delta(velocity)
            for c in a["cancels"]:
                if _active(c, fr) and (c["execute"] or c["buffer"]):
                    for target in c["action_ids"]:
                        t["cancel_buffer"][ai, fr, _ACTION_INDEX[target]] = True

    to_action = np.vectorize(_ACTION_INDEX.__getitem__, otypes=[np.int32])
    held = np.arange(SPECIAL_ATTACK_HOLD_FRAME + 1, dtype=f32)
    data = FrameData(
        frame_count=per_action("frame_count", np.int32),
        is_loop=per_action("is_loop", bool),
        loop_from_frame=per_action("loop_from_frame", np.int32),
        always_cancelable=per_action("always_cancelable", bool),
        action_type=per_action("type", np.int32),
        **t,
        attack_damage_action=to_action(per_attack("damageActionID")),
        attack_guard_action=to_action(per_attack("guardActionID")),
        attack_number_of_hit=per_attack("numberOfHit"),
        attack_vital_damage=per_attack("vitalHealthDamage"),
        attack_guard_damage=per_attack("guardHealthDamage"),
        attack_hit_stun=per_attack("hitStunFrame"),
        attack_guard_stun=per_attack("guardStunFrame"),
        attack_guard_break_stun=per_attack("guardBreakStunFrame"),
        special_attack_progress=(held / f32(SPECIAL_ATTACK_HOLD_FRAME)).astype(f32),
    )
    for table in data:
        table.setflags(write=False)
    return data


FRAME_DATA = _build()

# Walking displacement per frame, exactly (FORWARD / BACKWARD actions). Walking
# backward subtracts, i.e. adds the negated delta.
FORWARD_DELTA = exact_delta(FORWARD_MOVE_SPEED)
BACKWARD_DELTA = exact_delta(-BACKWARD_MOVE_SPEED)


def tables() -> FrameData:
    """``FRAME_DATA`` as JAX arrays (embedded as constants when traced)."""
    return jax.tree.map(jnp.asarray, FRAME_DATA)
