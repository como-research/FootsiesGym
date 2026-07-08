"""
Tests for both action space configurations:
  - Base (6 actions): NONE, BACK, FORWARD, ATTACK, BACK_ATTACK, FORWARD_ATTACK
  - Expanded (9 actions): adds SPECIAL_CHARGE, FORWARD_SPECIAL_CHARGE,
    BACK_SPECIAL_CHARGE

Verifies action space construction, special charge resolution to base
actions, LUT boundaries, encoder compatibility, and action_to_bits
guards. No server needed.
"""

import numpy as np
import pytest

from footsiesgym.footsies import encoder
from footsiesgym.footsies.game import constants
from footsiesgym.footsies.game.footsies_game import FootsiesGame
from footsiesgym.footsies.game.proto import footsies_service_pb2 as pb2

# ── Action space construction ────────────────────────────────────


class TestActionSpaceConstruction:
    def test_base_action_space(self):
        from footsiesgym.footsies.footsies_env import FootsiesEnv

        spaces = FootsiesEnv.get_action_space(use_special_charge_action=False)
        for agent in ["p1", "p2"]:
            assert spaces[agent].n == 6

    def test_expanded_action_space(self):
        from footsiesgym.footsies.footsies_env import FootsiesEnv

        spaces = FootsiesEnv.get_action_space(use_special_charge_action=True)
        for agent in ["p1", "p2"]:
            assert spaces[agent].n == 9

    def test_base_actions_are_0_to_5(self):
        assert constants.EnvActions.NONE == 0
        assert constants.EnvActions.BACK == 1
        assert constants.EnvActions.FORWARD == 2
        assert constants.EnvActions.ATTACK == 3
        assert constants.EnvActions.BACK_ATTACK == 4
        assert constants.EnvActions.FORWARD_ATTACK == 5

    def test_special_charge_actions_are_6_to_8(self):
        assert constants.EnvActions.SPECIAL_CHARGE == 6
        assert constants.EnvActions.FORWARD_SPECIAL_CHARGE == 7
        assert constants.EnvActions.BACK_SPECIAL_CHARGE == 8


# ── LUT boundaries ──────────────────────────────────────────────


class TestLUTBoundaries:
    def test_p1_env_to_bits_covers_base_actions_only(self):
        assert len(constants.P1_ENV_TO_BITS) == 6
        # All base actions (0-5) should be valid indices
        for a in range(6):
            _ = constants.P1_ENV_TO_BITS[a]

    def test_p2_env_to_bits_covers_base_actions_only(self):
        assert len(constants.P2_ENV_TO_BITS) == 6
        for a in range(6):
            _ = constants.P2_ENV_TO_BITS[a]

    def test_charge_action_lut_covers_base_actions_only(self):
        assert len(constants.CHARGE_ACTION_LUT) == 6
        for a in range(6):
            _ = constants.CHARGE_ACTION_LUT[a]

    def test_special_charge_indices_out_of_bounds_for_luts(self):
        """Special charge actions (6-8) must not index into LUTs."""
        for action in [6, 7, 8]:
            with pytest.raises(IndexError):
                _ = constants.P1_ENV_TO_BITS[action]
            with pytest.raises(IndexError):
                _ = constants.P2_ENV_TO_BITS[action]
            with pytest.raises(IndexError):
                _ = constants.CHARGE_ACTION_LUT[action]

    def test_p1_orientation_mapping(self):
        """P1 faces right: BACK→LEFT, FORWARD→RIGHT."""
        assert (
            constants.P1_ENV_TO_BITS[constants.EnvActions.BACK]
            == constants.ActionBits.LEFT
        )
        assert (
            constants.P1_ENV_TO_BITS[constants.EnvActions.FORWARD]
            == constants.ActionBits.RIGHT
        )
        assert (
            constants.P1_ENV_TO_BITS[constants.EnvActions.BACK_ATTACK]
            == constants.ActionBits.LEFT_ATTACK
        )
        assert (
            constants.P1_ENV_TO_BITS[constants.EnvActions.FORWARD_ATTACK]
            == constants.ActionBits.RIGHT_ATTACK
        )

    def test_p2_orientation_mapping(self):
        """P2 faces left: BACK→RIGHT, FORWARD→LEFT."""
        assert (
            constants.P2_ENV_TO_BITS[constants.EnvActions.BACK]
            == constants.ActionBits.RIGHT
        )
        assert (
            constants.P2_ENV_TO_BITS[constants.EnvActions.FORWARD]
            == constants.ActionBits.LEFT
        )
        assert (
            constants.P2_ENV_TO_BITS[constants.EnvActions.BACK_ATTACK]
            == constants.ActionBits.RIGHT_ATTACK
        )
        assert (
            constants.P2_ENV_TO_BITS[constants.EnvActions.FORWARD_ATTACK]
            == constants.ActionBits.LEFT_ATTACK
        )

    def test_charge_lut_conversions(self):
        """While holding special, movement → directional attack."""
        lut = constants.CHARGE_ACTION_LUT
        assert lut[constants.EnvActions.NONE] == constants.EnvActions.ATTACK
        assert lut[constants.EnvActions.BACK] == constants.EnvActions.BACK_ATTACK
        assert lut[constants.EnvActions.FORWARD] == constants.EnvActions.FORWARD_ATTACK
        assert lut[constants.EnvActions.ATTACK] == constants.EnvActions.ATTACK
        assert lut[constants.EnvActions.BACK_ATTACK] == constants.EnvActions.BACK_ATTACK
        assert (
            lut[constants.EnvActions.FORWARD_ATTACK]
            == constants.EnvActions.FORWARD_ATTACK
        )


# ── Special charge resolution ───────────────────────────────────


class TestSpecialChargeResolution:
    """Test that special charge actions resolve to correct base actions."""

    def test_convert_special_charge_to_base_action(self):
        from footsiesgym.footsies.footsies_env import FootsiesEnv

        assert (
            FootsiesEnv._convert_special_charge_to_base_action(
                constants.EnvActions.SPECIAL_CHARGE
            )
            == constants.EnvActions.NONE
        )
        assert (
            FootsiesEnv._convert_special_charge_to_base_action(
                constants.EnvActions.FORWARD_SPECIAL_CHARGE
            )
            == constants.EnvActions.FORWARD
        )
        assert (
            FootsiesEnv._convert_special_charge_to_base_action(
                constants.EnvActions.BACK_SPECIAL_CHARGE
            )
            == constants.EnvActions.BACK
        )

    def test_convert_special_charge_rejects_non_charge(self):
        from footsiesgym.footsies.footsies_env import FootsiesEnv

        for action in range(6):
            with pytest.raises(ValueError):
                FootsiesEnv._convert_special_charge_to_base_action(action)

    def test_convert_to_charge_action(self):
        from footsiesgym.footsies.footsies_env import FootsiesEnv

        assert (
            FootsiesEnv._convert_to_charge_action(constants.EnvActions.BACK)
            == constants.EnvActions.BACK_ATTACK
        )
        assert (
            FootsiesEnv._convert_to_charge_action(constants.EnvActions.FORWARD)
            == constants.EnvActions.FORWARD_ATTACK
        )
        assert (
            FootsiesEnv._convert_to_charge_action(constants.EnvActions.NONE)
            == constants.EnvActions.ATTACK
        )
        assert (
            FootsiesEnv._convert_to_charge_action(constants.EnvActions.ATTACK)
            == constants.EnvActions.ATTACK
        )

    def test_vectorized_special_charge_toggle(self):
        """Vectorized path: special charge actions toggle holding
        state and resolve to base actions before LUT lookup."""
        N = 8
        to_execute = np.array(
            [
                constants.EnvActions.SPECIAL_CHARGE,
                constants.EnvActions.FORWARD_SPECIAL_CHARGE,
                constants.EnvActions.BACK_SPECIAL_CHARGE,
                constants.EnvActions.NONE,
                constants.EnvActions.FORWARD,
                constants.EnvActions.BACK,
                constants.EnvActions.ATTACK,
                constants.EnvActions.SPECIAL_CHARGE,
            ],
            dtype=np.int64,
        )
        # Simulate as (N, 1) for a single player column
        to_exec = to_execute.reshape(N, 1)
        holding = np.zeros((N, 1), dtype=bool)

        # Toggle holding state for special charge actions
        is_special = (
            (to_exec == constants.EnvActions.SPECIAL_CHARGE)
            | (to_exec == constants.EnvActions.FORWARD_SPECIAL_CHARGE)
            | (to_exec == constants.EnvActions.BACK_SPECIAL_CHARGE)
        )
        holding[is_special] = ~holding[is_special]

        # Convert special charge actions to base movement
        base = to_exec.copy()
        base[to_exec == constants.EnvActions.SPECIAL_CHARGE] = constants.EnvActions.NONE
        base[to_exec == constants.EnvActions.FORWARD_SPECIAL_CHARGE] = (
            constants.EnvActions.FORWARD
        )
        base[to_exec == constants.EnvActions.BACK_SPECIAL_CHARGE] = (
            constants.EnvActions.BACK
        )
        to_exec[is_special] = base[is_special]

        # Verify all actions are now base (0-5)
        assert to_exec.max() <= constants.EnvActions.FORWARD_ATTACK

        # Verify holding state toggled correctly
        expected_holding = np.array(
            [True, True, True, False, False, False, False, True]
        ).reshape(N, 1)
        np.testing.assert_array_equal(holding, expected_holding)

        # Verify base action conversions
        expected_base = np.array(
            [
                constants.EnvActions.NONE,
                constants.EnvActions.FORWARD,
                constants.EnvActions.BACK,
                constants.EnvActions.NONE,
                constants.EnvActions.FORWARD,
                constants.EnvActions.BACK,
                constants.EnvActions.ATTACK,
                constants.EnvActions.NONE,
            ]
        ).reshape(N, 1)
        np.testing.assert_array_equal(to_exec, expected_base)

        # Apply charge conversion for held envs
        held = holding
        if held.any():
            to_exec[held] = constants.CHARGE_ACTION_LUT[to_exec[held]]

        # All actions should still be base (0-5)
        assert to_exec.max() <= constants.EnvActions.FORWARD_ATTACK

        # Held envs should have charge actions:
        # env 0: NONE → ATTACK, env 1: FORWARD → FORWARD_ATTACK,
        # env 2: BACK → BACK_ATTACK, env 7: NONE → ATTACK
        assert to_exec[0, 0] == constants.EnvActions.ATTACK
        assert to_exec[1, 0] == constants.EnvActions.FORWARD_ATTACK
        assert to_exec[2, 0] == constants.EnvActions.BACK_ATTACK
        assert to_exec[7, 0] == constants.EnvActions.ATTACK


# ── action_to_bits guard ────────────────────────────────────────


class TestActionToBits:
    def test_base_actions_accepted(self):
        """All base actions (0-5) should convert without error."""
        for action in range(6):
            p1_bits = FootsiesGame.action_to_bits(action, True)
            p2_bits = FootsiesGame.action_to_bits(action, False)
            assert p1_bits in [
                constants.ActionBits.NONE,
                constants.ActionBits.LEFT,
                constants.ActionBits.RIGHT,
                constants.ActionBits.ATTACK,
                constants.ActionBits.LEFT_ATTACK,
                constants.ActionBits.RIGHT_ATTACK,
            ]
            assert p2_bits in [
                constants.ActionBits.NONE,
                constants.ActionBits.LEFT,
                constants.ActionBits.RIGHT,
                constants.ActionBits.ATTACK,
                constants.ActionBits.LEFT_ATTACK,
                constants.ActionBits.RIGHT_ATTACK,
            ]

    def test_special_charge_actions_rejected(self):
        """Special charge actions (6-8) must not reach action_to_bits."""
        for action in [6, 7, 8]:
            with pytest.raises(AssertionError):
                FootsiesGame.action_to_bits(action, True)
            with pytest.raises(AssertionError):
                FootsiesGame.action_to_bits(action, False)

    def test_numpy_scalar_accepted(self):
        """np.ndarray scalars should be handled."""
        action = np.array(3)
        bits = FootsiesGame.action_to_bits(action, True)
        assert bits == constants.ActionBits.ATTACK


# ── Encoder with both action space sizes ─────────────────────────


def _make_batch_raw_state(num_envs, rng=None):
    """Build a synthetic BatchRawState with random but valid fields."""
    if rng is None:
        rng = np.random.RandomState(123)

    action_id_values = list(constants.FOOTSIES_ACTION_IDS.values())
    s = pb2.BatchRawState()

    for prefix in ("p1", "p2"):
        getattr(s, f"{prefix}_position_x").extend(
            rng.uniform(-3.5, 3.5, num_envs).tolist()
        )
        getattr(s, f"{prefix}_velocity_x").extend(
            rng.uniform(-4.0, 4.0, num_envs).tolist()
        )
        getattr(s, f"{prefix}_special_attack_progress").extend(
            rng.uniform(0.0, 1.0, num_envs).tolist()
        )
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
        getattr(s, f"{prefix}_vital_health").extend(
            rng.randint(0, 2, num_envs).tolist()
        )
        getattr(s, f"{prefix}_guard_health").extend(
            rng.randint(0, 4, num_envs).tolist()
        )
        getattr(s, f"{prefix}_current_action_id").extend(
            rng.choice(action_id_values, num_envs).tolist()
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
        player.current_action_id = getattr(raw, f"{prefix}_current_action_id")[idx]
        player.current_action_frame = getattr(raw, f"{prefix}_current_action_frame")[
            idx
        ]
        player.current_action_frame_count = getattr(
            raw, f"{prefix}_current_action_frame_count"
        )[idx]
        player.is_action_end = getattr(raw, f"{prefix}_is_action_end")[idx]
        player.is_always_cancelable = getattr(raw, f"{prefix}_is_always_cancelable")[
            idx
        ]
        player.current_action_hit_count = getattr(
            raw, f"{prefix}_current_action_hit_count"
        )[idx]
        player.current_hit_stun_frame = getattr(
            raw, f"{prefix}_current_hit_stun_frame"
        )[idx]
        player.is_in_hit_stun = getattr(raw, f"{prefix}_is_in_hit_stun")[idx]
        player.sprite_shake_position = getattr(raw, f"{prefix}_sprite_shake_position")[
            idx
        ]
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


class TestEncoderActionSpaces:
    """Encoder produces correct obs with both 6 and 9 action configs."""

    @pytest.mark.parametrize("num_actions", [6, 9])
    def test_scalar_encoder_obs_size(self, num_actions):
        enc = encoder.FootsiesEncoder()
        raw = _make_batch_raw_state(1, np.random.RandomState(42))
        gs = _build_game_state_from_raw(raw, 0)
        result = enc.encode(
            gs,
            prev_actions={"p1": 0, "p2": 0},
            is_charging_special={"p1": False, "p2": False},
            num_actions=num_actions,
        )
        expected_size = 79 + num_actions
        assert result["p1"].shape == (expected_size,)
        assert result["p2"].shape == (expected_size,)

    @pytest.mark.parametrize("num_actions", [6, 9])
    def test_vectorized_encoder_obs_size(self, num_actions):
        N = 8
        enc = encoder.VectorizedEncoder(num_actions=num_actions)
        raw = _make_batch_raw_state(N, np.random.RandomState(42))
        result = enc.encode(
            raw,
            np.zeros(N, dtype=np.int64),
            np.zeros(N, dtype=np.int64),
            np.zeros(N, dtype=bool),
            np.zeros(N, dtype=bool),
        )
        expected_size = 79 + num_actions
        assert result["p1"].shape == (N, expected_size)
        assert result["p2"].shape == (N, expected_size)

    @pytest.mark.parametrize("num_actions", [6, 9])
    def test_scalar_vs_vectorized_match(self, num_actions):
        """Both encoders produce identical output for each config."""
        N = 16
        rng = np.random.RandomState(77)
        raw = _make_batch_raw_state(N, rng)
        prev_p1 = rng.randint(0, num_actions, N).astype(np.int64)
        prev_p2 = rng.randint(0, num_actions, N).astype(np.int64)
        hold_p1 = rng.choice([True, False], N)
        hold_p2 = rng.choice([True, False], N)

        vec_enc = encoder.VectorizedEncoder(num_actions=num_actions)
        vec_result = vec_enc.encode(raw, prev_p1, prev_p2, hold_p1, hold_p2)

        scalar_enc = encoder.FootsiesEncoder()
        for i in range(N):
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
                num_actions=num_actions,
            )
            scalar_enc.reset()

            np.testing.assert_allclose(
                vec_result["p1"][i],
                scalar_result["p1"],
                atol=1e-6,
                err_msg=(f"p1 mismatch at env {i} " f"(num_actions={num_actions})"),
            )
            np.testing.assert_allclose(
                vec_result["p2"][i],
                scalar_result["p2"],
                atol=1e-6,
                err_msg=(f"p2 mismatch at env {i} " f"(num_actions={num_actions})"),
            )

    def test_prev_action_one_hot_6_actions(self):
        """With 6 actions, prev_action one-hot has 6 slots."""
        enc = encoder.FootsiesEncoder()
        raw = _make_batch_raw_state(1, np.random.RandomState(0))
        gs = _build_game_state_from_raw(raw, 0)

        for action in range(6):
            result = enc.encode(
                gs,
                prev_actions={"p1": action, "p2": 0},
                is_charging_special={"p1": False, "p2": False},
                num_actions=6,
            )
            enc.reset()
            # obs size = 79 + 6 = 85
            assert result["p1"].shape == (85,)

    def test_prev_action_one_hot_9_actions(self):
        """With 9 actions, prev_action one-hot has 9 slots."""
        enc = encoder.FootsiesEncoder()
        raw = _make_batch_raw_state(1, np.random.RandomState(0))
        gs = _build_game_state_from_raw(raw, 0)

        for action in range(9):
            result = enc.encode(
                gs,
                prev_actions={"p1": action, "p2": 0},
                is_charging_special={"p1": False, "p2": False},
                num_actions=9,
            )
            enc.reset()
            # obs size = 79 + 9 = 88
            assert result["p1"].shape == (88,)

    def test_expanded_prev_action_activates_correct_slot(self):
        """Special charge prev_actions (6-8) activate the right
        one-hot slot in the 9-action encoder."""
        N = 1
        enc = encoder.VectorizedEncoder(num_actions=9)
        raw = _make_batch_raw_state(N, np.random.RandomState(0))

        # Privileged block starts at column 1 + 37 = 38
        # prev_action one-hot starts after dash(2) + special(1) = 41
        prev_action_start = 1 + 37 + 3  # common + wk + dash + sp

        for action in range(9):
            result = enc.encode(
                raw,
                np.array([action], dtype=np.int64),
                np.array([0], dtype=np.int64),
                np.array([False]),
                np.array([False]),
            )
            one_hot = result["p1"][0, prev_action_start : prev_action_start + 9]
            assert one_hot[action] == 1.0
            assert one_hot.sum() == 1.0


# ── Observation layout: prev_action & holding_special encoding ───
#
# Full observation layout (matching Unity's WritePlayerFull):
#   [common(1)] [well_known(37)] [fwd_dash(1), bwd_dash(1),
#    special_progress(1)] [prev_action(N)] [holding_special(1)]
#    [opponent_well_known(37)]
#
# Self-features start at offset 1.
# prev_action one-hot starts at: 1 + 37 + 3 = 41
# holding_special at: 41 + num_actions
# Opponent well-known at: 42 + num_actions
# Total obs size: 79 + num_actions


def _prev_action_offset(num_actions):
    """Offset of prev_action one-hot in the full observation."""
    return 1 + 37 + 3  # common + well_known + dash(2) + special(1)


def _holding_special_offset(num_actions):
    """Offset of holding_special scalar in the full observation."""
    return _prev_action_offset(num_actions) + num_actions


class TestPrevActionEncoding:
    """Validate that previous action passed to the encoder appears
    correctly in the output observation as a one-hot vector,
    matching the Unity C# WritePlayerFull layout."""

    @pytest.mark.parametrize("num_actions", [6, 9])
    def test_vectorized_batch_prev_action_roundtrip(self, num_actions):
        """Pass a batch of distinct prev_actions through the
        vectorized encoder and verify each env's observation has
        the correct one-hot at the right offset."""
        N = num_actions  # one env per action value
        enc = encoder.VectorizedEncoder(num_actions=num_actions)
        raw = _make_batch_raw_state(N, np.random.RandomState(11))

        # Each env gets a different prev_action for p1
        prev_p1 = np.arange(num_actions, dtype=np.int64)
        prev_p2 = np.zeros(N, dtype=np.int64)
        hold_p1 = np.zeros(N, dtype=bool)
        hold_p2 = np.zeros(N, dtype=bool)

        result = enc.encode(raw, prev_p1, prev_p2, hold_p1, hold_p2)

        pa_start = _prev_action_offset(num_actions)
        pa_end = pa_start + num_actions

        for i in range(N):
            one_hot = result["p1"][i, pa_start:pa_end]
            # Exactly one slot active
            assert one_hot.sum() == pytest.approx(
                1.0
            ), f"env {i}: one-hot sum={one_hot.sum()}"
            assert one_hot[i] == pytest.approx(1.0), (
                f"env {i}: expected slot {i} active, " f"got {one_hot}"
            )
            # All other slots zero
            for j in range(num_actions):
                if j != i:
                    assert one_hot[j] == pytest.approx(0.0)

    @pytest.mark.parametrize("num_actions", [6, 9])
    def test_scalar_batch_prev_action_roundtrip(self, num_actions):
        """Same roundtrip test using the scalar FootsiesEncoder."""
        enc = encoder.FootsiesEncoder()
        raw = _make_batch_raw_state(1, np.random.RandomState(22))
        gs = _build_game_state_from_raw(raw, 0)

        pa_start = _prev_action_offset(num_actions)
        pa_end = pa_start + num_actions

        for action in range(num_actions):
            result = enc.encode(
                gs,
                prev_actions={"p1": action, "p2": 0},
                is_charging_special={
                    "p1": False,
                    "p2": False,
                },
                num_actions=num_actions,
            )
            enc.reset()

            one_hot = result["p1"][pa_start:pa_end]
            assert one_hot.sum() == pytest.approx(1.0)
            assert one_hot[action] == pytest.approx(1.0)

    @pytest.mark.parametrize("num_actions", [6, 9])
    def test_p2_prev_action_independent(self, num_actions):
        """P2's prev_action is encoded in p2's self-block, not p1's."""
        N = 1
        enc = encoder.VectorizedEncoder(num_actions=num_actions)
        raw = _make_batch_raw_state(N, np.random.RandomState(33))

        pa_start = _prev_action_offset(num_actions)
        pa_end = pa_start + num_actions

        # p1 action=0, p2 action=3
        result = enc.encode(
            raw,
            np.array([0], dtype=np.int64),
            np.array([3], dtype=np.int64),
            np.array([False]),
            np.array([False]),
        )

        # p1's self-block should have action 0
        p1_one_hot = result["p1"][0, pa_start:pa_end]
        assert p1_one_hot[0] == pytest.approx(1.0)
        assert p1_one_hot[3] == pytest.approx(0.0)

        # p2's self-block should have action 3
        p2_one_hot = result["p2"][0, pa_start:pa_end]
        assert p2_one_hot[3] == pytest.approx(1.0)
        assert p2_one_hot[0] == pytest.approx(0.0)

    def test_prev_action_not_in_opponent_view(self):
        """Opponent's well-known block should not contain
        prev_action (it's a privileged feature)."""
        num_actions = 9
        N = 1
        enc = encoder.VectorizedEncoder(num_actions=num_actions)
        raw = _make_batch_raw_state(N, np.random.RandomState(44))

        # p2 has action 7 (FORWARD_SPECIAL_CHARGE)
        result_a = enc.encode(
            raw,
            np.array([0], dtype=np.int64),
            np.array([7], dtype=np.int64),
            np.array([False]),
            np.array([False]),
        )
        # p2 has action 2
        result_b = enc.encode(
            raw,
            np.array([0], dtype=np.int64),
            np.array([2], dtype=np.int64),
            np.array([False]),
            np.array([False]),
        )

        # p1's observation includes p2 as opponent (well-known only)
        # The opponent block is the last 37 floats
        opp_start = 1 + 37 + 4 + num_actions  # 42 + num_actions
        p1_opp_a = result_a["p1"][0, opp_start:]
        p1_opp_b = result_b["p1"][0, opp_start:]

        # Changing p2's prev_action should NOT affect p1's
        # opponent view (since prev_action is privileged)
        np.testing.assert_array_equal(
            p1_opp_a,
            p1_opp_b,
            err_msg=(
                "Opponent well-known block should not change "
                "when opponent's prev_action changes"
            ),
        )

    @pytest.mark.parametrize("num_actions", [6, 9])
    def test_special_charge_prev_actions_encoded(self, num_actions):
        """When num_actions=9, special charge actions (6-8) used
        as prev_action produce valid one-hot encoding."""
        if num_actions < 9:
            pytest.skip("Only relevant for 9-action space")

        N = 3
        enc = encoder.VectorizedEncoder(num_actions=9)
        raw = _make_batch_raw_state(N, np.random.RandomState(55))

        # All three special charge variants
        prev_p1 = np.array([6, 7, 8], dtype=np.int64)
        prev_p2 = np.zeros(N, dtype=np.int64)

        result = enc.encode(
            raw,
            prev_p1,
            prev_p2,
            np.zeros(N, dtype=bool),
            np.zeros(N, dtype=bool),
        )

        pa_start = _prev_action_offset(9)
        pa_end = pa_start + 9

        for i, expected_slot in enumerate([6, 7, 8]):
            one_hot = result["p1"][i, pa_start:pa_end]
            assert one_hot[expected_slot] == pytest.approx(1.0)
            assert one_hot.sum() == pytest.approx(1.0)


class TestHoldingSpecialEncoding:
    """Validate that holding_special is correctly encoded at the
    right offset in the observation."""

    @pytest.mark.parametrize("num_actions", [6, 9])
    def test_holding_special_offset_and_value(self, num_actions):
        """holding_special appears at the correct offset and
        reflects the input boolean."""
        N = 4
        enc = encoder.VectorizedEncoder(num_actions=num_actions)
        raw = _make_batch_raw_state(N, np.random.RandomState(66))

        hs_offset = _holding_special_offset(num_actions)

        # Mix of True/False across envs for p1
        hold_p1 = np.array([False, True, False, True])
        hold_p2 = np.array([True, False, True, False])

        result = enc.encode(
            raw,
            np.zeros(N, dtype=np.int64),
            np.zeros(N, dtype=np.int64),
            hold_p1,
            hold_p2,
        )

        # Check p1's self-block in p1-centric obs
        for i in range(N):
            expected = 1.0 if hold_p1[i] else 0.0
            assert result["p1"][i, hs_offset] == pytest.approx(expected), (
                f"p1 env {i}: expected holding_special="
                f"{expected}, got {result['p1'][i, hs_offset]}"
            )

        # Check p2's self-block in p2-centric obs
        for i in range(N):
            expected = 1.0 if hold_p2[i] else 0.0
            assert result["p2"][i, hs_offset] == pytest.approx(expected), (
                f"p2 env {i}: expected holding_special="
                f"{expected}, got {result['p2'][i, hs_offset]}"
            )

    @pytest.mark.parametrize("num_actions", [6, 9])
    def test_holding_special_not_in_opponent_view(self, num_actions):
        """holding_special is privileged — changing it should not
        affect the opponent's view of this player."""
        N = 1
        enc = encoder.VectorizedEncoder(num_actions=num_actions)
        raw = _make_batch_raw_state(N, np.random.RandomState(77))

        opp_start = 1 + 37 + 4 + num_actions

        result_off = enc.encode(
            raw,
            np.zeros(N, dtype=np.int64),
            np.zeros(N, dtype=np.int64),
            np.array([False]),  # p1 not holding
            np.array([False]),
        )
        result_on = enc.encode(
            raw,
            np.zeros(N, dtype=np.int64),
            np.zeros(N, dtype=np.int64),
            np.array([True]),  # p1 holding
            np.array([False]),
        )

        # P2's observation of p1 (opponent well-known block)
        # should be identical regardless of p1's holding_special
        p2_opp_off = result_off["p2"][0, opp_start:]
        p2_opp_on = result_on["p2"][0, opp_start:]
        np.testing.assert_array_equal(
            p2_opp_off,
            p2_opp_on,
            err_msg=(
                "Opponent well-known block should not change "
                "when player's holding_special changes"
            ),
        )

    @pytest.mark.parametrize("num_actions", [6, 9])
    def test_scalar_encoder_holding_special(self, num_actions):
        """Scalar encoder also encodes holding_special correctly."""
        enc = encoder.FootsiesEncoder()
        raw = _make_batch_raw_state(1, np.random.RandomState(88))
        gs = _build_game_state_from_raw(raw, 0)

        hs_offset = _holding_special_offset(num_actions)

        for holding in [False, True]:
            result = enc.encode(
                gs,
                prev_actions={"p1": 0, "p2": 0},
                is_charging_special={
                    "p1": holding,
                    "p2": False,
                },
                num_actions=num_actions,
            )
            enc.reset()

            expected = 1.0 if holding else 0.0
            assert result["p1"][hs_offset] == pytest.approx(expected)

    @pytest.mark.parametrize("num_actions", [6, 9])
    def test_prev_action_and_holding_jointly(self, num_actions):
        """Both prev_action and holding_special encode correctly
        when set simultaneously."""
        N = 1
        enc = encoder.VectorizedEncoder(num_actions=num_actions)
        raw = _make_batch_raw_state(N, np.random.RandomState(99))

        pa_start = _prev_action_offset(num_actions)
        pa_end = pa_start + num_actions
        hs_offset = _holding_special_offset(num_actions)

        action = min(num_actions - 1, 5)  # last base action

        result = enc.encode(
            raw,
            np.array([action], dtype=np.int64),
            np.array([0], dtype=np.int64),
            np.array([True]),
            np.array([False]),
        )

        # prev_action one-hot correct
        one_hot = result["p1"][0, pa_start:pa_end]
        assert one_hot[action] == pytest.approx(1.0)
        assert one_hot.sum() == pytest.approx(1.0)

        # holding_special correct
        assert result["p1"][0, hs_offset] == pytest.approx(1.0)

        # p2 should have action=0, holding=False
        p2_one_hot = result["p2"][0, pa_start:pa_end]
        assert p2_one_hot[0] == pytest.approx(1.0)
        assert result["p2"][0, hs_offset] == pytest.approx(0.0)
