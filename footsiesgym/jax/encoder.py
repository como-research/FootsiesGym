"""JAX port of ``footsiesgym.footsies.encoder.VectorizedEncoder``.

``ObservationEncoder.encode`` turns one game's ``FootsiesGame.raw_state()``
into the two player-centric observations, bit-identical to the numpy
encoder: ``[common (1), self (37 public + 3 + num_actions + 1 privileged),
opponent (37 public)]``.

XLA's float32 division is not correctly rounded, so every quotient comes
from a table computed with IEEE numpy division: integer features use a
precomputed ``k / d`` for every integer in range, velocity uses its finite
set of values, and division by a power of two is an exact multiply.
"""

import jax
import jax.numpy as jnp
import numpy as np

from footsiesgym.footsies.encoder import NormalizationConstants as N

from . import frame_data as fd

NUM_GUARD_CLASSES = 4
_INT_RANGE = 1024  # integer features are clipped to [-R, R] before lookup


def _int_quotients(divisor: float) -> np.ndarray:
    return np.arange(-_INT_RANGE, _INT_RANGE + 1, dtype=np.float32) / np.float32(
        divisor
    )


def _divide(table: np.ndarray, values):
    return jnp.asarray(table)[jnp.clip(values, -_INT_RANGE, _INT_RANGE) + _INT_RANGE]


def _one_hot(value, values):
    return (jnp.asarray(values) == value).astype(jnp.float32)


class ObservationEncoder:
    """Builds the (88,) observation of each player from a game's raw state."""

    FRAME = _int_quotients(N.meaningful_frame_count)
    HIT_STUN = _int_quotients(N.meaningful_hit_stun_frame)
    SPRITE_SHAKE = _int_quotients(N.meaningful_sprite_shake_frame)
    ADVANTAGE = _int_quotients(N.meaningful_frame_advantage)
    VELOCITIES = np.unique(np.append(fd.FRAME_DATA.movement_velocity, np.float32(0)))
    VELOCITY = VELOCITIES / np.float32(N.meaningful_velocity_x)
    assert N.stage_width == 8.0 and N.max_x_value == 4.0  # exact multiplies below

    def __init__(self, num_actions: int = 9):
        self.num_actions = num_actions

    @property
    def observation_size(self) -> int:
        return 79 + self.num_actions

    def encode(self, raw: dict, prev_actions, holding_special) -> dict[str, jax.Array]:
        """Observations of both players.

        :param raw: ``FootsiesGame.raw_state()``.
        :param prev_actions: (2,) previously selected env actions of P1, P2.
        :param holding_special: (2,) special-charge toggle state of P1, P2.
        """
        distance = jnp.abs(raw["p1_position_x"] - raw["p2_position_x"]) * jnp.float32(
            0.125
        )
        common = distance[None]
        p1_public, p1_private = self._player(
            raw, "p1", prev_actions[0], holding_special[0]
        )
        p2_public, p2_private = self._player(
            raw, "p2", prev_actions[1], holding_special[1]
        )
        return {
            "p1": jnp.concatenate([common, p1_public, p1_private, p2_public]),
            "p2": jnp.concatenate([common, p2_public, p2_private, p1_public]),
        }

    def _player(self, raw: dict, player: str, prev_action, holding_special):
        """(public, privileged) features of one player."""
        get = lambda name: raw[f"{player}_{name}"]
        scalar = lambda x: jnp.asarray(x, jnp.float32)[None]
        frame, count = get("current_action_frame"), get("current_action_frame_count")
        velocity = jnp.asarray(self.VELOCITY)[
            jnp.argmax(jnp.asarray(self.VELOCITIES) == get("velocity_x"))
        ]
        public = [
            scalar(get("position_x") * jnp.float32(0.25)),
            scalar(velocity),
            scalar(get("is_dead")),
            scalar(get("vital_health")),
            _one_hot(get("guard_health"), np.arange(NUM_GUARD_CLASSES)),
            _one_hot(get("current_action_id"), fd.ACTION_IDS),
            scalar(_divide(self.FRAME, frame)),
            scalar(_divide(self.FRAME, count)),
            scalar(_divide(self.FRAME, count - frame)),
            scalar(get("is_action_end")),
            scalar(get("is_always_cancelable")),
            scalar(get("current_action_hit_count")),
            scalar(_divide(self.HIT_STUN, get("current_hit_stun_frame"))),
            scalar(get("is_in_hit_stun")),
            scalar(get("sprite_shake_position")),
            scalar(_divide(self.SPRITE_SHAKE, get("max_sprite_shake_frame"))),
            scalar(get("is_face_right")),
            scalar(_divide(self.ADVANTAGE, get("current_frame_advantage"))),
        ]
        privileged = [
            scalar(get("would_next_forward_input_dash")),
            scalar(get("would_next_backward_input_dash")),
            scalar(jnp.minimum(get("special_attack_progress"), jnp.float32(1.0))),
            _one_hot(prev_action, np.arange(self.num_actions)),
            scalar(holding_special),
        ]
        return jnp.concatenate(public), jnp.concatenate(privileged)
