"""ctypes bindings for the CPU build of ``csrc/`` (``libfootsies``).

States are opaque byte buffers (numpy ``uint8`` arrays of shape
``(n, sizeof)``); only ``RawState`` and ``EnvConfig`` are mirrored here, as
numpy structured dtypes whose sizes are checked against the library.
"""

import ctypes
import functools

import numpy as np

from .build import build_cpu

_P = ctypes.c_void_p
_I = ctypes.c_int

_SIGNATURES = {
    "footsies_sizeof_game": ([], _I),
    "footsies_sizeof_env_state": ([], _I),
    "footsies_sizeof_env_config": ([], _I),
    "footsies_sizeof_raw_state": ([], _I),
    "footsies_max_delay_steps": ([], _I),
    "footsies_obs_dim": ([], _I),
    "footsies_set_num_threads": ([_I], None),
    "footsies_get_num_threads": ([], _I),
    "footsies_mono_move": (
        [_P, _P, ctypes.c_float, ctypes.c_float, ctypes.c_int64],
        None,
    ),
    "footsies_game_new": ([_P, _I], None),
    "footsies_game_reset": ([_P, _P, _I], None),
    "footsies_game_step": ([_P, _P, _P, _I, _I], None),
    "footsies_game_raw": ([_P, _P, _I], None),
    "footsies_env_reset": ([_P, _P, _I], None),
    "footsies_env_observe": ([_P, _P, _I], None),
    "footsies_env_step": ([_P, _P, _P, _P, _P, _P, _P, _I, _I], None),
    "footsies_env_info": ([_P, _P, _P, _I], None),
    "footsies_env_games": ([_P, _P, _I], None),
    "footsies_batch_width": ([], _I),
    "footsies_batch_new": ([_P, _I], _P),
    "footsies_batch_free": ([_P], None),
    "footsies_batch_reset": ([_P, _P], None),
    "footsies_batch_step": ([_P, _P, _P, _P, _P, _P, _I], None),
    "footsies_batch_game_step": ([_P, _P, _I], None),
    "footsies_batch_get": ([_P, _P], None),
    "footsies_batch_set": ([_P, _P], None),
}

_PER_PLAYER = [
    ("position_x", np.float32),
    ("velocity_x", np.float32),
    ("special_attack_progress", np.float32),
    ("vital_health", np.int32),
    ("guard_health", np.int32),
    ("current_action_id", np.int32),
    ("current_action_frame", np.int32),
    ("current_action_frame_count", np.int32),
    ("current_action_hit_count", np.int32),
    ("current_hit_stun_frame", np.int32),
    ("sprite_shake_position", np.int32),
    ("max_sprite_shake_frame", np.int32),
    ("current_frame_advantage", np.int32),
    ("is_dead", np.uint8),
    ("is_action_end", np.uint8),
    ("is_always_cancelable", np.uint8),
    ("is_in_hit_stun", np.uint8),
    ("is_face_right", np.uint8),
    ("would_next_forward_input_dash", np.uint8),
    ("would_next_backward_input_dash", np.uint8),
]
_PER_GAME = [
    ("round_states", np.int32),
    ("frame_counts", np.int32),
    ("dones", np.uint8),
    ("rewards", np.int8),
]
PER_PLAYER_FIELDS = tuple(name for name, _ in _PER_PLAYER)

# Mirrors of the C structs (align=True gives the C layout).
RAW_STATE = np.dtype([(n, t, (2,)) for n, t in _PER_PLAYER] + _PER_GAME, align=True)
REWARD_TABLE_SHAPE = (2, 2, 2, 2, 4, 4, 2)
ENV_CONFIG = np.dtype(
    [
        ("frame_skip", np.int32),
        ("action_delay_frames", np.int32),
        ("max_t", np.int32),
        ("guard_break_reward_nonzero", np.uint8),
        ("use_reward_budget", np.uint8),
        ("reward_table", np.float32, (int(np.prod(REWARD_TABLE_SHAPE)),)),
    ],
    align=True,
)


class Library:
    """The loaded library, with its struct sizes."""

    def __init__(self, path):
        self.path = str(path)
        self.lib = ctypes.CDLL(self.path)
        for name, (args, res) in _SIGNATURES.items():
            fn = getattr(self.lib, name)
            fn.argtypes, fn.restype = args, res
            setattr(self, name.removeprefix("footsies_"), fn)
        self.game_size = self.sizeof_game()
        self.env_state_size = self.sizeof_env_state()
        self.max_delay_steps = self.max_delay_steps()
        self.obs_dim = self.obs_dim()
        for dtype, size in (
            (RAW_STATE, self.sizeof_raw_state()),
            (ENV_CONFIG, self.sizeof_env_config()),
        ):
            if dtype.itemsize != size:
                raise RuntimeError(f"struct layout mismatch: {dtype} vs {size} bytes")


@functools.cache
def load() -> Library:
    """The CPU library, built on first use."""
    return Library(build_cpu())


def ptr(a: np.ndarray):
    """Pointer to a C-contiguous array's data (keep ``a`` alive for the call)."""
    assert a.flags.c_contiguous, "arrays passed to C must be C-contiguous"
    return a.ctypes.data


def raw_to_fields(raw: np.ndarray) -> dict[str, np.ndarray]:
    """A ``RAW_STATE`` array as BatchRawState fields (``p1_*``, ``p2_*``, ...)."""
    fields = {}
    for name in PER_PLAYER_FIELDS:
        for i in range(2):
            fields[f"p{i + 1}_{name}"] = np.ascontiguousarray(raw[name][:, i])
    for name, _ in _PER_GAME:
        fields[name] = np.ascontiguousarray(raw[name])
    return fields
