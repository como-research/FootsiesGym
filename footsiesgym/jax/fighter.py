"""Port of the Unity ``Fighter`` class (FootsiesV2 ``Assets/Script/Fighter.cs``).

``Fighter`` holds one fighter's state, and its methods are the C# methods of
the same names. Every method returns a new ``Fighter`` rather than mutating
it. ``FootsiesGame`` vmaps the methods over its two fighters.

Quirks of the C# code that are reproduced on purpose:
  * ``setup_battle_start`` leaves hit stun, frame advantage,
    ``input_backward`` and ``reserve_proximity_guard`` untouched, so they
    carry over from the previous round.
  * ``velocity_x`` is only written by movement data and persists otherwise.
  * Unity's Mono runtime evaluates float expressions in double precision
    (see ``_move``).
"""

import jax
import jax.numpy as jnp

from . import frame_data as fd
from ._struct import Struct, struct

# Fighter.NotifyDamaged results (DamageResult)
DAMAGE, GUARD, GUARD_BREAK = 1, 2, 3


@struct
class Boxes(Struct):
    """A fighter's collision boxes for the current frame.

    Rects are (x, y, w, h) with x the horizontal centre (``BoxBase``).
    """

    hit_rect: jax.Array  # (MAX_HITBOXES, 4)
    hit_valid: jax.Array
    hit_attack: jax.Array  # attack index
    hit_proximity: jax.Array
    hurt_rect: jax.Array  # (MAX_HURTBOXES, 4)
    hurt_valid: jax.Array
    push_rect: jax.Array  # (4,)

    def shifted(self, dx, apply) -> "Boxes":
        """Every box moved horizontally by ``dx`` where ``apply`` holds."""
        shift = lambda rect: rect.at[..., 0].set(
            jnp.where(apply, rect[..., 0] + dx, rect[..., 0])
        )
        return self.replace(
            hit_rect=shift(self.hit_rect),
            hurt_rect=shift(self.hurt_rect),
            push_rect=shift(self.push_rect),
        )

    def hits(self, other: "Boxes"):
        """Per hitbox: whether it overlaps any of ``other``'s hurtboxes."""
        overlap = _boxes_overlap(self.hit_rect[:, None], other.hurt_rect[None, :])
        return jnp.any(overlap & other.hurt_valid[None, :], axis=1)


@struct
class Fighter(Struct):
    position_x: jax.Array  # float32 (position.y is always 0)
    velocity_x: jax.Array  # float32
    face_right: jax.Array  # bool, isFaceRight
    vital_health: jax.Array
    guard_health: jax.Array
    action: jax.Array  # currentActionID, as an action index
    action_frame: jax.Array
    hit_count: jax.Array
    frame_advantage: jax.Array
    hit_stun: jax.Array
    sprite_shake: jax.Array
    has_won: jax.Array  # bool
    input_backward: jax.Array  # bool, isInputBackward
    reserve_proximity_guard: jax.Array  # bool, isReserveProximityGuard
    buffer_action: jax.Array  # bufferActionID, action index or -1
    reserve_damage_action: jax.Array  # reserveDamageActionID, action index or -1
    inputs: jax.Array  # (INPUT_HISTORY,) input bits, [0] = newest

    @classmethod
    def new(cls) -> "Fighter":
        """Field values of a freshly constructed C# Fighter."""
        zero = jnp.int32(0)
        return cls(
            position_x=jnp.float32(0),
            velocity_x=jnp.float32(0),
            face_right=jnp.bool_(False),
            vital_health=zero,
            guard_health=zero,
            action=zero,
            action_frame=zero,
            hit_count=zero,
            frame_advantage=zero,
            hit_stun=zero,
            sprite_shake=zero,
            has_won=jnp.bool_(False),
            input_backward=jnp.bool_(False),
            reserve_proximity_guard=jnp.bool_(False),
            buffer_action=jnp.int32(-1),
            reserve_damage_action=jnp.int32(-1),
            inputs=jnp.zeros(fd.INPUT_HISTORY, jnp.int32),
        )

    def setup_battle_start(self, start_x, face_right) -> "Fighter":
        return self.replace(
            position_x=jnp.asarray(start_x, jnp.float32),
            face_right=jnp.asarray(face_right, bool),
            vital_health=jnp.int32(1),
            guard_health=jnp.int32(fd.START_GUARD_HEALTH),
            has_won=jnp.bool_(False),
            velocity_x=jnp.float32(0),
            inputs=jnp.zeros_like(self.inputs),
        ).set_current_action(fd.STAND)

    # ── Current action ───────────────────────────────────────────────────

    @property
    def is_dead(self):
        return self.vital_health <= 0

    @property
    def frame_count(self):
        """currentActionFrameCount."""
        return fd.tables().frame_count[self.action]

    @property
    def is_action_end(self):
        return self.action_frame >= self.frame_count

    @property
    def is_always_cancelable(self):
        return fd.tables().always_cancelable[self.action]

    @property
    def frames_left(self):
        """Frames left in the current action, as GetFrameAdvantage counts them."""
        return jnp.where(
            self.is_always_cancelable, 0, self.frame_count - self.action_frame
        )

    @property
    def _frame(self):
        """The action frame as an index into the frame-data tables."""
        return jnp.clip(self.action_frame, 0, fd.NUM_FRAMES - 1)

    def set_current_action(self, action) -> "Fighter":
        return self.replace(
            action=jnp.asarray(action, jnp.int32),
            action_frame=jnp.int32(0),
            hit_count=jnp.int32(0),
            buffer_action=jnp.int32(-1),
            reserve_damage_action=jnp.int32(-1),
            sprite_shake=jnp.int32(0),
        )

    def request_action(self, action, enabled=True) -> "Fighter":
        """RequestAction (callers never use its return value)."""
        action = jnp.asarray(action, jnp.int32)
        end = self.is_action_end
        same = self.action == action
        cancelable = self.is_always_cancelable
        start = enabled & (end | (~same & cancelable))
        buffer = (
            enabled
            & ~end
            & ~same
            & ~cancelable
            & fd.tables().cancel_buffer[self.action, self._frame, action]
        )
        f = self.set_current_action(action).where(start, self)
        return f.replace(buffer_action=jnp.where(buffer, action, f.buffer_action))

    def increment_action_frame(self) -> "Fighter":
        T = fd.tables()
        shake = -self.sprite_shake
        shake = shake + jnp.where(shake > 0, -1, 1)
        shake = jnp.where(jnp.abs(self.sprite_shake) > 0, shake, self.sprite_shake)

        frame = self.action_frame + 1
        loop = (frame >= self.frame_count) & T.is_loop[self.action]
        frame = jnp.where(loop, T.loop_from_frame[self.action], frame)

        in_stun = self.hit_stun > 0
        return self.replace(
            sprite_shake=shake,
            hit_stun=jnp.where(in_stun, self.hit_stun - 1, self.hit_stun),
            action_frame=jnp.where(in_stun, self.action_frame, frame),
        )

    # ── Input ────────────────────────────────────────────────────────────

    def update_input(self, input_bits) -> "Fighter":
        newest = jnp.asarray(input_bits, jnp.int32)[None]
        return self.replace(inputs=jnp.concatenate([newest, self.inputs[:-1]]))

    def _directions(self, inputs):
        """IsForwardInput / IsBackwardInput, elementwise."""
        forward_bit = jnp.where(self.face_right, fd.INPUT_RIGHT, fd.INPUT_LEFT)
        backward_bit = jnp.where(self.face_right, fd.INPUT_LEFT, fd.INPUT_RIGHT)
        return (inputs & forward_bit) > 0, (inputs & backward_bit) > 0

    @property
    def would_next_forward_input_dash(self):
        forward, backward = self._directions(self.inputs)
        return _dash_check(forward, backward, first=0)

    @property
    def would_next_backward_input_dash(self):
        forward, backward = self._directions(self.inputs)
        return _dash_check(backward, forward, first=0)

    @property
    def special_attack_progress(self):
        """GetSpecialAttackProgress."""
        held = (self.inputs[: fd.SPECIAL_ATTACK_HOLD_FRAME] & fd.INPUT_ATTACK) > 0
        return fd.tables().special_attack_progress[jnp.sum(jnp.cumprod(held))]

    def update_action_request(self) -> "Fighter":
        forward, backward = self._directions(self.inputs)
        # inputDown[0] / inputUp[0] (derived from the history, as in UpdateInput)
        changed = self.inputs[0] ^ self.inputs[1]
        down, up = changed & self.inputs[0], changed & ~self.inputs[0]
        attack_held = (self.inputs & fd.INPUT_ATTACK) > 0

        # Early returns, in priority order
        won = self.request_action(fd.WIN)
        use_reserved = (self.reserve_damage_action != -1) & (self.hit_stun <= 0)
        reserved = self.set_current_action(self.reserve_damage_action)
        can_cancel = fd.CAN_CANCEL_ON_WHIFF | (self.hit_count > 0)
        use_buffered = (self.buffer_action != -1) & can_cancel & (self.hit_stun <= 0)
        buffered = self.set_current_action(self.buffer_action)

        # Attacks
        is_forward, is_backward = forward[0], backward[0]
        directional = is_forward | is_backward
        special = ((up & fd.INPUT_ATTACK) > 0) & jnp.all(
            attack_held[1 : fd.SPECIAL_ATTACK_HOLD_FRAME]
        )
        attack = (down & fd.INPUT_ATTACK) > 0
        in_normal = (
            (self.action == fd.N_ATTACK) | (self.action == fd.B_ATTACK)
        ) & ~self.is_action_end
        normal = jnp.where(
            in_normal, fd.N_SPECIAL, jnp.where(directional, fd.B_ATTACK, fd.N_ATTACK)
        )
        f = self.request_action(
            jnp.where(directional, fd.B_SPECIAL, fd.N_SPECIAL), special
        )
        f = f.request_action(normal, ~special & attack)

        # Dashes
        forward_down, backward_down = self._directions(down)
        forward_dash = forward_down & _dash_check(forward, backward, first=1)
        backward_dash = backward_down & _dash_check(backward, forward, first=1)
        f = f.request_action(fd.DASH_FORWARD, forward_dash)
        f = f.request_action(fd.DASH_BACKWARD, ~forward_dash & backward_dash)

        # Walking and guarding
        f = f.replace(input_backward=is_backward)
        backward_action = jnp.where(
            f.reserve_proximity_guard, fd.GUARD_PROXIMITY, fd.BACKWARD
        )
        move = jnp.where(
            is_forward & is_backward,
            fd.STAND,
            jnp.where(
                is_forward,
                fd.FORWARD,
                jnp.where(is_backward, backward_action, fd.STAND),
            ),
        )
        f = f.request_action(move).replace(reserve_proximity_guard=jnp.bool_(False))

        f = buffered.where(use_buffered, f)
        f = reserved.where(use_reserved, f)
        return won.where(self.has_won, f)

    # ── Movement and boxes ───────────────────────────────────────────────

    @property
    def _sign(self):
        return jnp.where(self.face_right, jnp.float32(1), jnp.float32(-1))

    def update_movement(self) -> "Fighter":
        T = fd.tables()
        has_data = T.has_movement[self.action, self._frame]
        velocity = T.movement_velocity[self.action, self._frame]
        walking_forward = self.action == fd.FORWARD
        walking_backward = self.action == fd.BACKWARD
        by_data = ~walking_forward & ~walking_backward & has_data
        # The per-frame displacement speed * dt as (hi, lo, tiny); see frame_data.
        delta = jnp.where(
            walking_forward,
            jnp.asarray(fd.FORWARD_DELTA),
            jnp.where(
                walking_backward,
                jnp.asarray(fd.BACKWARD_DELTA),
                T.movement_delta[self.action, self._frame],
            ),
        )
        moves = walking_forward | walking_backward | (by_data & (velocity != 0))
        moved = _move(self.position_x, delta, self._sign)
        active = self.hit_stun <= 0
        return self.replace(
            position_x=jnp.where(active & moves, moved, self.position_x),
            velocity_x=jnp.where(active & by_data, velocity, self.velocity_x),
        )

    def boxes(self) -> Boxes:
        """UpdateBoxes / ApplyCurrentActionData."""
        T = fd.tables()
        a, frame = self.action, self._frame

        def place(rect):  # TransformToFightRect (position.y is always 0)
            x = self.position_x + rect[..., 0] * self._sign
            return jnp.stack([x, rect[..., 1], rect[..., 2], rect[..., 3]], axis=-1)

        return Boxes(
            hit_rect=place(T.hitbox_rect[a, frame]),
            hit_valid=T.hitbox_valid[a, frame],
            hit_attack=T.hitbox_attack[a, frame],
            hit_proximity=T.hitbox_proximity[a, frame],
            hurt_rect=place(T.hurtbox_rect[a, frame]),
            hurt_valid=T.hurtbox_valid[a, frame],
            push_rect=place(T.pushbox_rect[a, frame]),
        )

    def apply_position_change(self, boxes: Boxes, dx, apply) -> tuple["Fighter", Boxes]:
        """ApplyPositionChange: the fighter and its boxes move by dx."""
        moved = self.replace(
            position_x=jnp.where(apply, self.position_x + dx, self.position_x)
        )
        return moved, boxes.shifted(dx, apply)

    # ── Attacking and being hit ──────────────────────────────────────────

    def can_attack_hit(self, attack):
        return self.hit_count < fd.tables().attack_number_of_hit[attack]

    def notify_attack_hit(self) -> "Fighter":
        return self.replace(hit_count=self.hit_count + 1)

    def notify_damaged(self, attack) -> tuple["Fighter", jax.Array]:
        """NotifyDamaged: the damaged fighter and the DamageResult."""
        T = fd.tables()
        guard_damage = T.attack_guard_damage[attack]
        guard = jnp.where(
            guard_damage > 0, self.guard_health - guard_damage, self.guard_health
        )
        guard_break = (guard_damage > 0) & (guard < 0)
        f = self.replace(guard_health=jnp.where(guard_break, 0, guard))

        guarding = (f.action == fd.BACKWARD) | (
            T.action_type[f.action] == fd.ACTION_TYPE_GUARD
        )
        blocked = f.set_current_action(T.attack_guard_action[attack])
        blocked = blocked.replace(
            reserve_damage_action=jnp.where(
                guard_break, fd.GUARD_BREAK, blocked.reserve_damage_action
            )
        )

        vital_damage = T.attack_vital_damage[attack]
        vital = jnp.where(
            vital_damage > 0, f.vital_health - vital_damage, f.vital_health
        )
        vital = jnp.where((vital_damage > 0) & (vital <= 0), 0, vital)
        damaged = f.replace(vital_health=vital).set_current_action(
            T.attack_damage_action[attack]
        )

        result = jnp.where(guarding, jnp.where(guard_break, GUARD_BREAK, GUARD), DAMAGE)
        return blocked.where(guarding, damaged), result

    def notify_in_proximity_guard_range(self) -> "Fighter":
        return self.replace(
            reserve_proximity_guard=self.reserve_proximity_guard | self.input_backward
        )

    def set_hit_stun(self, frames) -> "Fighter":
        return self.replace(hit_stun=jnp.asarray(frames, jnp.int32))

    def set_sprite_shake_frame(self, frames) -> "Fighter":
        frames = jnp.minimum(frames, fd.MAX_SPRITE_SHAKE_FRAME)
        return self.replace(sprite_shake=frames * jnp.where(self.face_right, -1, 1))


def hit_stun_frame(result, attack):
    """GetHitStunFrame: stun frames for both fighters after a hit."""
    T = fd.tables()
    return jnp.where(
        result == GUARD,
        T.attack_guard_stun[attack],
        jnp.where(
            result == GUARD_BREAK,
            T.attack_guard_break_stun[attack],
            T.attack_hit_stun[attack],
        ),
    )


def _move(position_x, delta, sign):
    """``position.x += speed * sign * FIXED_DELTA_TIME`` as Unity's Mono runs it.

    Mono evaluates float expressions in double precision and rounds to
    float32 only when storing to a field, so the product is never rounded on
    its own: the result is ``round32(round64(position_x + product))``. With
    the exact product given as ``sign * (hi + lo)`` with ``delta = (hi, lo,
    tiny)`` from frame_data:

      * if ``|position_x| <= tiny``, rounding to float64 gives the product
        itself (a tie goes to it too, as its last float64 bit is 0), so the
        result is ``hi``;
      * otherwise rounding to float64 first never changes the result for the
        positions and speeds of this game, so the exact sum is rounded once.

    ``testing/test_jax_backend.py`` checks this against float64 for every
    float32 position in [-8, 8]. Only float32 error-free transformations
    are used, so the result is exact on every backend, without float64.
    """
    hi, lo, tiny = delta[0] * sign, delta[1] * sign, delta[2]  # sign is +-1: exact
    s, e = _two_sum(position_x, hi)  # s + e == position_x + hi
    t, t_error = _two_sum(e, lo)  # t + t_error == e + lo
    # Round t to odd, so that the single rounding of s + t below is correct.
    t_bits = jax.lax.bitcast_convert_type(t, jnp.int32)
    inexact_even = (t_error != 0) & ((t_bits & 1) == 0)
    toward = jnp.where(t_error > 0, jnp.float32(jnp.inf), jnp.float32(-jnp.inf))
    t = jnp.where(inexact_even, jnp.nextafter(t, toward), t)
    return jnp.where(jnp.abs(position_x) <= tiny, hi, s + t)


def _two_sum(a, b):
    """Knuth's TwoSum: ``s = fl(a + b)`` and the exact error ``a + b - s``."""
    s = a + b
    b_virtual = s - a
    a_virtual = s - b_virtual
    return s, (a - a_virtual) + (b - b_virtual)


def _dash_check(toward, away, first: int):
    """Shared loop of Check*DashInput (first=1) and WouldNext*InputDash (first=0).

    Scans the input history from ``first``: the first input pointing ``away``
    fails, the first pointing ``toward`` succeeds iff one of the next
    dashAllowFrame-1 inputs is neutral.
    """
    n = fd.DASH_ALLOW_FRAME
    neutral = ~toward & ~away
    last = n if first == 1 else n - 1
    result = jnp.bool_(False)
    for i in reversed(range(first, last)):
        any_neutral = jnp.any(neutral[i + 1 : i + n])
        result = jnp.where(away[i], False, jnp.where(toward[i], any_neutral, result))
    return result


def _boxes_overlap(a, b):
    """BoxBase.Overlaps (x = centre, inclusive) for (x, y, w, h) rects."""
    a_x_min, a_x_max = a[..., 0] - a[..., 2] * 0.5, a[..., 0] + a[..., 2] * 0.5
    b_x_min, b_x_max = b[..., 0] - b[..., 2] * 0.5, b[..., 0] + b[..., 2] * 0.5
    a_y_min, a_y_max = a[..., 1], a[..., 1] + a[..., 3]
    b_y_min, b_y_max = b[..., 1], b[..., 1] + b[..., 3]
    return (
        (b_x_max >= a_x_min)
        & (b_x_min <= a_x_max)
        & (b_y_max >= a_y_min)
        & (b_y_min <= a_y_max)
    )
