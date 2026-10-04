"""Port of the Unity ``BattleSimulation`` (FootsiesV2 ``BattleSimulation.cs``).

``FootsiesGame`` is one game: the two fighters plus the round state. Its
methods reproduce ``BattleSimulation`` frame for frame and return a new
game rather than mutating it::

    game = FootsiesGame.new()
    game = game.step(p1_input, p2_input)       # one frame
    game = game.step_n(p1_input, p2_input, 4)  # StepN
    fields = game.raw_state()                  # BatchRawState

Inputs are ``InputDefine`` bit masks (left=1, right=2, attack=4). The
fighters are stored stacked (leaves have a leading axis of size 2, P1 then
P2). Batch games with ``jax.vmap(FootsiesGame.step)``.
"""

import jax
import jax.numpy as jnp
import numpy as np

from . import frame_data as fd
from ._struct import Struct, struct
from .fighter import Boxes, Fighter, hit_stun_frame

# BattleSimulation.RoundStateType
ROUND_FIGHT = 2
ROUND_KO = 3

START_X = np.array([-2.0, 2.0], dtype=np.float32)
FACE_RIGHT = np.array([True, False])  # P1 faces right, P2 faces left
STAGE_MIN_X = np.float32(-fd.BATTLE_AREA_WIDTH / 2)
STAGE_MAX_X = np.float32(fd.BATTLE_AREA_WIDTH / 2)


@struct
class FootsiesGame(Struct):
    fighters: Fighter  # P1 and P2, stacked
    round_state: jax.Array
    frame_count: jax.Array
    done: jax.Array  # bool: the episode ended on the last step
    reward: jax.Array  # +1 P1 won, -1 P2 won, 0 otherwise

    @classmethod
    def new(cls) -> "FootsiesGame":
        """A freshly constructed BattleSimulation (the constructor calls Reset)."""
        fresh = cls(
            fighters=Fighter.stack(Fighter.new(), Fighter.new()),
            round_state=jnp.int32(0),
            frame_count=jnp.int32(0),
            done=jnp.bool_(False),
            reward=jnp.int32(0),
        )
        return fresh.reset()

    def reset(self) -> "FootsiesGame":
        """Reset(): fields SetupBattleStart skips carry over (see fighter.py)."""
        setup = jax.vmap(Fighter.setup_battle_start)
        return self.replace(
            fighters=setup(
                self.fighters, jnp.asarray(START_X), jnp.asarray(FACE_RIGHT)
            ),
            round_state=jnp.int32(ROUND_FIGHT),
            frame_count=jnp.int32(-1),
            done=jnp.bool_(False),
            reward=jnp.int32(0),
        )

    def step(self, p1_input, p2_input) -> "FootsiesGame":
        """Step: advance one frame."""
        game = self.reset().where(self.round_state != ROUND_FIGHT, self)
        fighters = jax.vmap(Fighter.update_input)(
            game.fighters, jnp.stack([p1_input, p2_input])
        )

        left = fighters.frames_left
        fighters = fighters.replace(
            frame_advantage=jnp.stack([left[1] - left[0], left[0] - left[1]])
        )

        fighters = jax.vmap(Fighter.increment_action_frame)(fighters)
        fighters = jax.vmap(Fighter.update_action_request)(fighters)
        fighters = jax.vmap(Fighter.update_movement)(fighters)
        boxes = jax.vmap(Fighter.boxes)(fighters)

        fighters, boxes = _push_character_vs_character(fighters, boxes)
        fighters, boxes = jax.vmap(_push_vs_background)(fighters, boxes)

        # Both checks use this frame's boxes, from before either hit landed.
        fighters = _check_attack(fighters, boxes, attacker=0, damaged=1)
        fighters = _check_attack(fighters, boxes, attacker=1, damaged=0)

        dead = fighters.vital_health <= 0
        done = dead[0] | dead[1]
        reward = jnp.where(dead[0] & ~dead[1], -1, jnp.where(dead[1] & ~dead[0], 1, 0))
        return game.replace(
            fighters=fighters,
            round_state=jnp.where(done, ROUND_KO, ROUND_FIGHT).astype(jnp.int32),
            frame_count=game.frame_count + 1,
            done=done,
            reward=reward.astype(jnp.int32),
        )

    def step_n(self, p1_input, p2_input, n_frames: int) -> "FootsiesGame":
        """StepN: repeat the inputs for n frames, stopping early at a KO."""

        def frame(_, carry):
            game, running = carry
            game = game.step(p1_input, p2_input).where(running, game)
            return game, running & ~game.done

        game, _ = jax.lax.fori_loop(0, n_frames, frame, (self, jnp.bool_(True)))
        return game

    def raw_state(self) -> dict[str, jax.Array]:
        """The BatchRawState fields of this game, keyed like the proto."""
        f = self.fighters
        frame_count = f.frame_count
        # These read the input history, so they are evaluated per fighter.
        dash_forward, dash_backward, special_progress = jax.vmap(
            lambda x: (
                x.would_next_forward_input_dash,
                x.would_next_backward_input_dash,
                x.special_attack_progress,
            )
        )(f)
        per_player = {
            "position_x": f.position_x,
            "is_dead": f.vital_health <= 0,
            "vital_health": f.vital_health,
            "guard_health": f.guard_health,
            "current_action_id": jnp.asarray(fd.ACTION_IDS)[f.action],
            "current_action_frame": f.action_frame,
            "current_action_frame_count": frame_count,
            "is_action_end": f.action_frame >= frame_count,
            "is_always_cancelable": f.is_always_cancelable,
            "current_action_hit_count": f.hit_count,
            "current_hit_stun_frame": f.hit_stun,
            "is_in_hit_stun": f.hit_stun > 0,
            "sprite_shake_position": f.sprite_shake,
            "max_sprite_shake_frame": jnp.full(2, fd.MAX_SPRITE_SHAKE_FRAME, jnp.int32),
            "velocity_x": f.velocity_x,
            "is_face_right": f.face_right,
            "current_frame_advantage": f.frame_advantage,
            "would_next_forward_input_dash": dash_forward,
            "would_next_backward_input_dash": dash_backward,
            "special_attack_progress": special_progress,
        }
        fields = {
            f"p{i + 1}_{name}": v[i] for name, v in per_player.items() for i in range(2)
        }
        fields.update(
            round_states=self.round_state,
            dones=self.done,
            rewards=self.reward,
            frame_counts=self.frame_count,
        )
        return fields


def _push_character_vs_character(fighters: Fighter, boxes: Boxes):
    """UpdatePushCharacterVsCharacter.

    Uses UnityEngine.Rect semantics (x = left edge, strict overlap), because
    the C# code calls Rect.Overlaps / Rect.xMin / Rect.xMax on the raw rects.
    """
    x1, y1, w1, h1 = boxes.push_rect[0]
    x2, y2, w2, h2 = boxes.push_rect[1]
    overlaps = (x2 + w2 > x1) & (x2 < x1 + w1) & (y2 + h2 > y1) & (y2 < y1 + h1)
    p1_x, p2_x = fighters.position_x
    p1_left = overlaps & (p1_x < p2_x)
    p1_right = overlaps & (p1_x > p2_x)
    left_overlap = (x1 + w1) - x2
    right_overlap = (x2 + w2) - x1
    dx = jnp.stack(
        [
            jnp.where(p1_left, -left_overlap * 0.5, right_overlap * 0.5),
            jnp.where(p1_left, left_overlap * 0.5, -right_overlap * 0.5),
        ]
    )
    push = jax.vmap(Fighter.apply_position_change, in_axes=(0, 0, 0, None))
    return push(fighters, boxes, dx, p1_left | p1_right)


def _push_vs_background(fighter: Fighter, boxes: Boxes):
    """PushFighterVsBackground (BoxBase xMin/xMax, x = centre)."""
    x, w = boxes.push_rect[0], boxes.push_rect[2]
    x_min, x_max = x - w * 0.5, x + w * 0.5
    past_left = x_min < STAGE_MIN_X
    past_right = ~past_left & (x_max > STAGE_MAX_X)
    dx = jnp.where(past_left, STAGE_MIN_X - x_min, STAGE_MAX_X - x_max)
    return fighter.apply_position_change(boxes, dx, past_left | past_right)


def _check_attack(
    fighters: Fighter, boxes: Boxes, attacker: int, damaged: int
) -> Fighter:
    """CheckAttack(attacker, damaged)."""
    att, dmg = fighters.at(attacker), fighters.at(damaged)
    att_boxes, dmg_boxes = boxes.at(attacker), boxes.at(damaged)

    touching = (
        att_boxes.hit_valid
        & att.can_attack_hit(att_boxes.hit_attack)
        & att_boxes.hits(dmg_boxes)
    )
    damaging = touching & ~att_boxes.hit_proximity
    is_hit = jnp.any(damaging)
    attack = att_boxes.hit_attack[
        jnp.argmax(damaging)
    ]  # the first damaging box in asset order
    is_proximity = ~is_hit & jnp.any(touching & att_boxes.hit_proximity)

    hit, result = dmg.notify_damaged(attack)
    stun = hit_stun_frame(result, attack)
    hit = hit.set_hit_stun(stun).set_sprite_shake_frame(stun // 3)
    att_hit = att.notify_attack_hit().set_hit_stun(stun)

    new_att = att_hit.where(is_hit, att)
    new_dmg = hit.where(
        is_hit, dmg.notify_in_proximity_guard_range().where(is_proximity, dmg)
    )
    pair = (new_att, new_dmg) if attacker == 0 else (new_dmg, new_att)
    return Fighter.stack(*pair)
