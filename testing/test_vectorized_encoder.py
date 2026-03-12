"""
Unit tests for VectorizedEncoder vs FootsiesEncoder consistency.

These tests use synthetic BatchRawState protobuf messages — no server needed.
"""

import numpy as np
import pytest

from footsiesgym.footsies.encoder import (
    FootsiesEncoder,
    VectorizedEncoder,
)
from footsiesgym.footsies.game import constants
from footsiesgym.footsies.game.proto import footsies_service_pb2 as pb2

NUM_ACTIONS = 7
ACTION_ID_VALUES = list(constants.FOOTSIES_ACTION_IDS.values())


def _make_batch_raw_state(num_envs, rng=None):
    """Build a synthetic BatchRawState with random but valid fields."""
    if rng is None:
        rng = np.random.RandomState(123)

    s = pb2.BatchRawState()

    for prefix in ("p1", "p2"):
        # Float fields
        getattr(s, f"{prefix}_position_x").extend(
            rng.uniform(-3.5, 3.5, num_envs).tolist()
        )
        getattr(s, f"{prefix}_velocity_x").extend(
            rng.uniform(-4.0, 4.0, num_envs).tolist()
        )
        getattr(s, f"{prefix}_special_attack_progress").extend(
            rng.uniform(0.0, 1.5, num_envs).tolist()
        )

        # Bool fields
        getattr(s, f"{prefix}_is_dead").extend(
            rng.choice([True, False], num_envs).tolist()
        )
        getattr(s, f"{prefix}_is_action_end").extend(
            rng.choice([True, False], num_envs).tolist()
        )
        getattr(s, f"{prefix}_is_always_cancelable").extend(
            rng.choice([True, False], num_envs).tolist()
        )
        getattr(s, f"{prefix}_is_in_hit_stun").extend(
            rng.choice([True, False], num_envs).tolist()
        )
        getattr(s, f"{prefix}_is_face_right").extend(
            rng.choice([True, False], num_envs).tolist()
        )
        getattr(s, f"{prefix}_would_next_forward_input_dash").extend(
            rng.choice([True, False], num_envs).tolist()
        )
        getattr(s, f"{prefix}_would_next_backward_input_dash").extend(
            rng.choice([True, False], num_envs).tolist()
        )

        # Int fields
        getattr(s, f"{prefix}_vital_health").extend(
            rng.randint(0, 2, num_envs).tolist()
        )
        getattr(s, f"{prefix}_guard_health").extend(
            rng.randint(0, 4, num_envs).tolist()
        )
        getattr(s, f"{prefix}_current_action_id").extend(
            rng.choice(ACTION_ID_VALUES, num_envs).tolist()
        )
        getattr(s, f"{prefix}_current_action_frame").extend(
            rng.randint(0, 20, num_envs).tolist()
        )
        getattr(s, f"{prefix}_current_action_frame_count").extend(
            rng.randint(5, 30, num_envs).tolist()
        )
        getattr(s, f"{prefix}_current_action_hit_count").extend(
            rng.randint(0, 3, num_envs).tolist()
        )
        getattr(s, f"{prefix}_current_hit_stun_frame").extend(
            rng.randint(0, 8, num_envs).tolist()
        )
        getattr(s, f"{prefix}_sprite_shake_position").extend(
            rng.randint(0, 3, num_envs).tolist()
        )
        getattr(s, f"{prefix}_max_sprite_shake_frame").extend(
            rng.randint(0, 8, num_envs).tolist()
        )
        getattr(s, f"{prefix}_current_frame_advantage").extend(
            rng.randint(-5, 6, num_envs).tolist()
        )

    s.round_states.extend(rng.randint(0, 5, num_envs).tolist())
    s.dones.extend(rng.choice([True, False], num_envs).tolist())
    s.rewards.extend(rng.choice([-1, 0, 1], num_envs).tolist())
    s.frame_counts.extend(rng.randint(0, 1000, num_envs).tolist())

    return s


def _build_game_state_from_raw(raw, idx):
    """Extract a single-env GameState from BatchRawState."""
    gs = pb2.GameState()

    for prefix, player in [("p1", gs.player1), ("p2", gs.player2)]:
        player.player_position_x = getattr(raw, f"{prefix}_position_x")[idx]
        player.velocity_x = getattr(raw, f"{prefix}_velocity_x")[idx]
        player.is_dead = getattr(raw, f"{prefix}_is_dead")[idx]
        player.vital_health = getattr(raw, f"{prefix}_vital_health")[idx]
        player.guard_health = getattr(raw, f"{prefix}_guard_health")[idx]
        player.current_action_id = getattr(raw, f"{prefix}_current_action_id")[
            idx
        ]
        player.current_action_frame = getattr(
            raw, f"{prefix}_current_action_frame"
        )[idx]
        player.current_action_frame_count = getattr(
            raw, f"{prefix}_current_action_frame_count"
        )[idx]
        player.is_action_end = getattr(raw, f"{prefix}_is_action_end")[idx]
        player.is_always_cancelable = getattr(
            raw, f"{prefix}_is_always_cancelable"
        )[idx]
        player.current_action_hit_count = getattr(
            raw, f"{prefix}_current_action_hit_count"
        )[idx]
        player.current_hit_stun_frame = getattr(
            raw, f"{prefix}_current_hit_stun_frame"
        )[idx]
        player.is_in_hit_stun = getattr(raw, f"{prefix}_is_in_hit_stun")[idx]
        player.sprite_shake_position = getattr(
            raw, f"{prefix}_sprite_shake_position"
        )[idx]
        player.max_sprite_shake_frame = getattr(
            raw, f"{prefix}_max_sprite_shake_frame"
        )[idx]
        player.is_face_right = getattr(raw, f"{prefix}_is_face_right")[idx]
        player.current_frame_advantage = getattr(
            raw, f"{prefix}_current_frame_advantage"
        )[idx]
        player.would_next_forward_input_dash = getattr(
            raw, f"{prefix}_would_next_forward_input_dash"
        )[idx]
        player.would_next_backward_input_dash = getattr(
            raw, f"{prefix}_would_next_backward_input_dash"
        )[idx]
        player.special_attack_progress = getattr(
            raw, f"{prefix}_special_attack_progress"
        )[idx]

    gs.round_state = raw.round_states[idx]
    gs.frame_count = raw.frame_counts[idx]
    return gs


class TestVectorizedEncoder:
    def test_obs_size(self):
        enc = VectorizedEncoder(num_actions=7)
        assert enc.obs_size == 86
        assert VectorizedEncoder.observation_size(7) == 86
        assert VectorizedEncoder.observation_size(6) == 85

    def test_matches_scalar_encoder(self):
        """Core test: vectorized output == scalar output for every env."""
        num_envs = 32
        rng = np.random.RandomState(42)
        raw = _make_batch_raw_state(num_envs, rng)

        prev_p1 = rng.randint(0, NUM_ACTIONS, num_envs).astype(np.int64)
        prev_p2 = rng.randint(0, NUM_ACTIONS, num_envs).astype(np.int64)
        hold_p1 = rng.choice([True, False], num_envs)
        hold_p2 = rng.choice([True, False], num_envs)

        vec_enc = VectorizedEncoder(num_actions=NUM_ACTIONS)
        vec_result = vec_enc.encode(raw, prev_p1, prev_p2, hold_p1, hold_p2)

        scalar_enc = FootsiesEncoder()
        for i in range(num_envs):
            gs = _build_game_state_from_raw(raw, i)
            scalar_result = scalar_enc.encode(
                gs,
                prev_actions={
                    "p1": int(prev_p1[i]),
                    "p2": int(prev_p2[i]),
                },
                is_charging_special={
                    "p1": bool(hold_p1[i]),
                    "p2": bool(hold_p2[i]),
                },
                num_actions=NUM_ACTIONS,
            )
            scalar_enc.reset()

            np.testing.assert_allclose(
                vec_result["p1"][i],
                scalar_result["p1"],
                atol=1e-6,
                err_msg=f"p1 mismatch at env {i}",
            )
            np.testing.assert_allclose(
                vec_result["p2"][i],
                scalar_result["p2"],
                atol=1e-6,
                err_msg=f"p2 mismatch at env {i}",
            )

    def test_single_env(self):
        """Vectorized with num_envs=1 matches scalar."""
        raw = _make_batch_raw_state(1, np.random.RandomState(99))
        prev_p1 = np.array([3], dtype=np.int64)
        prev_p2 = np.array([0], dtype=np.int64)
        hold_p1 = np.array([True])
        hold_p2 = np.array([False])

        vec_enc = VectorizedEncoder(num_actions=NUM_ACTIONS)
        vec_result = vec_enc.encode(raw, prev_p1, prev_p2, hold_p1, hold_p2)

        scalar_enc = FootsiesEncoder()
        gs = _build_game_state_from_raw(raw, 0)
        scalar_result = scalar_enc.encode(
            gs,
            prev_actions={"p1": 3, "p2": 0},
            is_charging_special={"p1": True, "p2": False},
            num_actions=NUM_ACTIONS,
        )

        np.testing.assert_allclose(
            vec_result["p1"][0], scalar_result["p1"], atol=1e-6
        )
        np.testing.assert_allclose(
            vec_result["p2"][0], scalar_result["p2"], atol=1e-6
        )

    def test_output_shape(self):
        num_envs = 16
        raw = _make_batch_raw_state(num_envs)

        enc = VectorizedEncoder(num_actions=NUM_ACTIONS)
        result = enc.encode(
            raw,
            np.zeros(num_envs, dtype=np.int64),
            np.zeros(num_envs, dtype=np.int64),
            np.zeros(num_envs, dtype=bool),
            np.zeros(num_envs, dtype=bool),
        )

        assert result["p1"].shape == (num_envs, 86)
        assert result["p2"].shape == (num_envs, 86)
        assert result["p1"].dtype == np.float32
        assert result["p2"].dtype == np.float32

    def test_special_progress_clamped(self):
        """special_attack_progress > 1.0 should be clamped to 1.0."""
        raw = _make_batch_raw_state(4, np.random.RandomState(0))
        # Override with values > 1.0
        del raw.p1_special_attack_progress[:]
        raw.p1_special_attack_progress.extend([0.5, 1.0, 1.5, 2.0])

        enc = VectorizedEncoder(num_actions=NUM_ACTIONS)
        result = enc.encode(
            raw,
            np.zeros(4, dtype=np.int64),
            np.zeros(4, dtype=np.int64),
            np.zeros(4, dtype=bool),
            np.zeros(4, dtype=bool),
        )

        # special_attack_progress is at self offset 39 (37 wk + 2 dash)
        # In the full obs: col 1 + 39 = 40
        sp_col = 1 + 37 + 2  # common(1) + well_known(37) + dash(2)
        sp_vals = result["p1"][:, sp_col]
        np.testing.assert_array_less(sp_vals, 1.0 + 1e-6)
        assert sp_vals[0] == pytest.approx(0.5)
        assert sp_vals[1] == pytest.approx(1.0)
        assert sp_vals[2] == pytest.approx(1.0)
        assert sp_vals[3] == pytest.approx(1.0)

    def test_different_num_actions(self):
        """Verify obs_size changes correctly with num_actions."""
        raw = _make_batch_raw_state(4, np.random.RandomState(7))

        for na in [5, 6, 7, 8]:
            enc = VectorizedEncoder(num_actions=na)
            result = enc.encode(
                raw,
                np.zeros(4, dtype=np.int64),
                np.zeros(4, dtype=np.int64),
                np.zeros(4, dtype=bool),
                np.zeros(4, dtype=bool),
            )
            expected_size = 79 + na
            assert result["p1"].shape == (4, expected_size)
            assert result["p2"].shape == (4, expected_size)

    def test_symmetry(self):
        """p1-centric obs of env should use p1 as self, p2 as opponent."""
        raw = _make_batch_raw_state(1, np.random.RandomState(55))
        enc = VectorizedEncoder(num_actions=NUM_ACTIONS)
        result = enc.encode(
            raw,
            np.array([2], dtype=np.int64),
            np.array([4], dtype=np.int64),
            np.array([False]),
            np.array([True]),
        )

        p1_obs = result["p1"][0]
        p2_obs = result["p2"][0]

        # Common feature (distance) should be the same
        assert p1_obs[0] == pytest.approx(p2_obs[0])

        # Self block of p1 obs should differ from self block of p2 obs
        # (unless the players happen to have identical state)
        # At minimum, prev_action and holding_special differ
        assert not np.allclose(p1_obs[1:49], p2_obs[1:49])

    def test_large_batch(self):
        """Smoke test with a large batch to catch indexing issues."""
        num_envs = 512
        raw = _make_batch_raw_state(num_envs, np.random.RandomState(0))
        enc = VectorizedEncoder(num_actions=NUM_ACTIONS)
        result = enc.encode(
            raw,
            np.random.randint(0, NUM_ACTIONS, num_envs).astype(np.int64),
            np.random.randint(0, NUM_ACTIONS, num_envs).astype(np.int64),
            np.random.choice([True, False], num_envs),
            np.random.choice([True, False], num_envs),
        )
        assert result["p1"].shape == (num_envs, 86)
        assert not np.any(np.isnan(result["p1"]))
        assert not np.any(np.isnan(result["p2"]))
