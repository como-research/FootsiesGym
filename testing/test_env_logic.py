"""Unit tests for FootsiesEnv single-env logic using a fake game client.

These tests exercise action delay, special-charge toggling, rewards,
termination/truncation, and config validation without a running game
server: the gRPC client is replaced by a fake that returns constructed
protobuf GameStates.
"""

import numpy as np
import pytest

from footsiesgym.footsies import encoder
from footsiesgym.footsies.footsies_env import FootsiesEnv
from footsiesgym.footsies.game import constants
from footsiesgym.footsies.game.footsies_game import FootsiesGame
from footsiesgym.footsies.game.proto import footsies_service_pb2 as pb2

EnvActions = constants.EnvActions


def make_game_state(
    p1_dead=False,
    p2_dead=False,
    p1_guard=3,
    p2_guard=3,
    p1_x=-2.0,
    p2_x=2.0,
    p1_action_id=0,
    p2_action_id=0,
):
    state = pb2.GameState()
    state.player1.is_dead = p1_dead
    state.player1.guard_health = p1_guard
    state.player1.player_position_x = p1_x
    state.player1.current_action_id = p1_action_id
    state.player2.is_dead = p2_dead
    state.player2.guard_health = p2_guard
    state.player2.player_position_x = p2_x
    state.player2.current_action_id = p2_action_id
    return state


class FakeGame:
    """Stands in for FootsiesGame; records actions, replays queued states."""

    def __init__(self):
        self.state = make_game_state()
        self.pending_states = []
        self.stepped_bits = []

    def reset_game(self):
        pass

    def start_game(self):
        pass

    def get_state(self):
        return self.state

    def step_n_frames(self, p1_action, p2_action, n_frames):
        self.stepped_bits.append((p1_action, p2_action))
        if self.pending_states:
            self.state = self.pending_states.pop(0)
        return self.state

    @staticmethod
    def action_to_bits(action, is_player_1):
        return FootsiesGame.action_to_bits(action, is_player_1)


def make_env(**config_overrides):
    config = {"launch_binaries": False, "port": 50051}
    config.update(config_overrides)
    env = FootsiesEnv(config=config)
    env.game = FakeGame()
    return env


NOOP_ACTIONS = {"p1": EnvActions.NONE, "p2": EnvActions.NONE}


class TestConfigValidation:
    def test_render_mode_rejected(self):
        with pytest.raises(ValueError, match="render_mode"):
            FootsiesEnv(config={}, render_mode="human")

    def test_action_delay_must_divide_frame_skip(self):
        with pytest.raises(AssertionError, match="divisible"):
            make_env(action_delay=6, frame_skip=4)

    def test_guard_break_reward_bounded_by_win_reward(self):
        with pytest.raises(AssertionError, match="[Gg]uard break"):
            make_env(guard_break_reward=0.5, win_reward_scaling_coeff=1.0)

    def test_spaces(self):
        env = make_env()
        assert env.action_space("p1").n == 6
        assert env.observation_space("p1").shape == (
            encoder.FootsiesEncoder.observation_size,
        )

        charge_env = make_env(use_special_charge_action=True)
        assert charge_env.action_space("p1").n == 9


class TestResetAndObs:
    def test_reset_returns_obs_and_infos_for_both_agents(self):
        env = make_env()
        obs, infos = env.reset()
        assert set(obs) == {"p1", "p2"}
        assert set(infos) == {"p1", "p2"}
        for agent_obs in obs.values():
            assert agent_obs.shape == (encoder.FootsiesEncoder.observation_size,)
            assert np.all(np.isfinite(agent_obs))

    def test_fight_state_in_infos(self):
        env = make_env(return_fight_state_in_infos=True)
        _, infos = env.reset()
        assert infos["p1"]["distance_x"] == pytest.approx(4.0)
        assert infos["p1"]["is_opponent_damage"] is False
        assert "is_facing_right" in infos["p1"]


class TestActionDelay:
    def test_actions_execute_after_delay(self):
        # action_delay=8 frames at frame_skip=4 -> 2-step delay
        env = make_env(action_delay=8, frame_skip=4)
        env.reset()

        attack = {"p1": EnvActions.ATTACK, "p2": EnvActions.NONE}
        env.step(attack)
        assert env.prev_executed_actions["p1"] == EnvActions.NONE
        env.step(NOOP_ACTIONS)
        assert env.prev_executed_actions["p1"] == EnvActions.NONE
        env.step(NOOP_ACTIONS)
        assert env.prev_executed_actions["p1"] == EnvActions.ATTACK

    def test_zero_delay_executes_immediately(self):
        env = make_env(action_delay=0)
        env.reset()
        env.step({"p1": EnvActions.ATTACK, "p2": EnvActions.FORWARD})
        assert env.prev_executed_actions["p1"] == EnvActions.ATTACK
        assert env.prev_executed_actions["p2"] == EnvActions.FORWARD


class TestSpecialCharge:
    def test_toggle_holds_and_converts_movement(self):
        env = make_env(use_special_charge_action=True, action_delay=0)
        env.reset()

        env.step({"p1": EnvActions.SPECIAL_CHARGE, "p2": EnvActions.NONE})
        assert env._holding_special_charge["p1"] is True
        # While holding, the toggle itself resolves to ATTACK (held button)
        assert env.prev_executed_actions["p1"] == EnvActions.ATTACK

        env.step({"p1": EnvActions.FORWARD, "p2": EnvActions.NONE})
        assert env.prev_executed_actions["p1"] == EnvActions.FORWARD_ATTACK

        env.step({"p1": EnvActions.SPECIAL_CHARGE, "p2": EnvActions.NONE})
        assert env._holding_special_charge["p1"] is False

        env.step({"p1": EnvActions.FORWARD, "p2": EnvActions.NONE})
        assert env.prev_executed_actions["p1"] == EnvActions.FORWARD

    def test_directional_toggles_keep_movement(self):
        env = make_env(use_special_charge_action=True, action_delay=0)
        env.reset()

        env.step({"p1": EnvActions.FORWARD_SPECIAL_CHARGE, "p2": EnvActions.NONE})
        assert env._holding_special_charge["p1"] is True
        assert env.prev_executed_actions["p1"] == EnvActions.FORWARD_ATTACK


class TestRewards:
    def test_win_is_zero_sum(self):
        env = make_env(action_delay=0)
        env.reset()
        env.game.pending_states = [make_game_state(p2_dead=True)]

        _, rewards, terminateds, _, _ = env.step(NOOP_ACTIONS)
        assert rewards["p1"] == pytest.approx(1.0)
        assert rewards["p2"] == pytest.approx(-1.0)
        assert terminateds == {"p1": True, "p2": True}
        assert env.agents == []

    def test_win_reward_scaling(self):
        env = make_env(action_delay=0, win_reward_scaling_coeff=2.0)
        env.reset()
        env.game.pending_states = [make_game_state(p1_dead=True)]

        _, rewards, _, _, _ = env.step(NOOP_ACTIONS)
        assert rewards["p2"] == pytest.approx(2.0)
        assert rewards["p1"] == pytest.approx(-2.0)

    def test_guard_break_reward_additive(self):
        env = make_env(action_delay=0, guard_break_reward=0.2)
        env.reset()
        env.game.pending_states = [make_game_state(p2_guard=2)]

        _, rewards, _, _, _ = env.step(NOOP_ACTIONS)
        assert rewards["p1"] == pytest.approx(0.2)
        assert rewards["p2"] == pytest.approx(-0.2)

        # Without the budget, a later win still pays the full coefficient
        env.game.pending_states = [make_game_state(p2_guard=2, p2_dead=True)]
        _, rewards, _, _, _ = env.step(NOOP_ACTIONS)
        assert rewards["p1"] == pytest.approx(1.0)

    def test_guard_break_deducts_from_reward_budget(self):
        env = make_env(action_delay=0, guard_break_reward=0.2, use_reward_budget=True)
        env.reset()
        env.game.pending_states = [make_game_state(p2_guard=2)]
        _, rewards, _, _, _ = env.step(NOOP_ACTIONS)
        assert rewards["p1"] == pytest.approx(0.2)

        env.game.pending_states = [make_game_state(p2_guard=2, p2_dead=True)]
        _, rewards, _, _, _ = env.step(NOOP_ACTIONS)
        # Win reward reduced by the guard break already paid out
        assert rewards["p1"] == pytest.approx(0.8)

    def test_no_reward_mid_episode(self):
        env = make_env(action_delay=0)
        env.reset()
        _, rewards, terminateds, truncateds, _ = env.step(NOOP_ACTIONS)
        assert rewards == {"p1": 0.0, "p2": 0.0}
        assert not any(terminateds.values())
        assert not any(truncateds.values())


class TestTruncation:
    def test_truncates_at_max_t(self):
        env = make_env(action_delay=0, max_t=3)
        env.reset()
        for _ in range(2):
            _, _, _, truncateds, _ = env.step(NOOP_ACTIONS)
            assert not any(truncateds.values())
        _, _, _, truncateds, _ = env.step(NOOP_ACTIONS)
        assert truncateds == {"p1": True, "p2": True}
        assert env.agents == []

    def test_reset_restores_agents_and_state(self):
        env = make_env(action_delay=0, max_t=1)
        env.reset()
        env.step(NOOP_ACTIONS)
        assert env.agents == []

        env.reset()
        assert env.agents == ["p1", "p2"]
        assert env.t == 0


class TestChargeConversionHelpers:
    @pytest.mark.parametrize(
        "action,expected",
        [
            (EnvActions.NONE, EnvActions.ATTACK),
            (EnvActions.BACK, EnvActions.BACK_ATTACK),
            (EnvActions.FORWARD, EnvActions.FORWARD_ATTACK),
            (EnvActions.ATTACK, EnvActions.ATTACK),
            (EnvActions.BACK_ATTACK, EnvActions.BACK_ATTACK),
            (EnvActions.FORWARD_ATTACK, EnvActions.FORWARD_ATTACK),
        ],
    )
    def test_convert_to_charge_action(self, action, expected):
        assert FootsiesEnv._convert_to_charge_action(action) == expected

    @pytest.mark.parametrize(
        "action,expected",
        [
            (EnvActions.SPECIAL_CHARGE, EnvActions.NONE),
            (EnvActions.FORWARD_SPECIAL_CHARGE, EnvActions.FORWARD),
            (EnvActions.BACK_SPECIAL_CHARGE, EnvActions.BACK),
        ],
    )
    def test_convert_special_charge_to_base(self, action, expected):
        assert FootsiesEnv._convert_special_charge_to_base_action(action) == expected

    def test_convert_special_charge_rejects_other_actions(self):
        with pytest.raises(ValueError):
            FootsiesEnv._convert_special_charge_to_base_action(EnvActions.ATTACK)
