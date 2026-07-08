"""Tests for the footsiesgym.make() factory."""

import pytest

import footsiesgym
from footsiesgym.footsies.footsies_env import FootsiesEnv


class TestMake:
    def test_returns_pettingzoo_env(self):
        env = footsiesgym.make(launch_binaries=False)
        assert isinstance(env, FootsiesEnv)
        assert env.possible_agents == ["p1", "p2"]

    def test_config_overrides_merge(self):
        env = footsiesgym.make(
            config={"max_t": 123, "frame_skip": 2, "action_delay": 4},
            launch_binaries=False,
        )
        assert env.max_t == 123
        assert env.frame_skip == 2
        assert env.action_delay_steps == 2

    def test_platform_forwarded_to_config(self):
        env = footsiesgym.make(platform="mac", launch_binaries=False)
        assert env.config["platform"] == "mac"

    def test_auto_launch_rejects_windows(self):
        with pytest.raises(AssertionError, match="Windows"):
            footsiesgym.make(platform="windows", launch_binaries=True)

    def test_version_is_exposed(self):
        assert footsiesgym.__version__


class TestMakeRLlib:
    def test_rllib_wrapper(self):
        pytest.importorskip("ray")
        from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
        from test_env_logic import FakeGame

        from footsiesgym.wrappers import wrap_rllib

        # ParallelPettingZooEnv resets the env on construction, so swap in
        # the fake game client to avoid needing a live server.
        env = FootsiesEnv(config={"launch_binaries": False, "port": 50051})
        env.game = FakeGame()
        wrapped = wrap_rllib(env)
        assert isinstance(wrapped, ParallelPettingZooEnv)
