"""PettingZoo parallel API compatibility tests for FootsiesEnv."""

import pytest
from pettingzoo.test import parallel_api_test

import footsiesgym


@pytest.fixture
def env():
    """Create a FootsiesEnv with binaries auto-launched."""
    env = footsiesgym.make(
        config={"max_t": 200},
        launch_binaries=True,
    )
    yield env
    env.close()


@pytest.mark.slow
def test_parallel_api(env):
    """Verify FootsiesEnv passes PettingZoo's parallel_api_test."""
    parallel_api_test(env, num_cycles=50)


@pytest.mark.slow
def test_observation_spaces(env):
    """Verify observation_space(agent) returns valid spaces for all agents."""
    for agent in env.possible_agents:
        space = env.observation_space(agent)
        assert space is not None
        assert space.shape == (env._encoder.observation_size,)


@pytest.mark.slow
def test_action_spaces(env):
    """Verify action_space(agent) returns valid spaces for all agents."""
    for agent in env.possible_agents:
        space = env.action_space(agent)
        assert space is not None
        assert hasattr(space, "n")


@pytest.mark.slow
def test_step_no_all_key(env):
    """Verify step() returns dicts without '__all__' keys."""
    env.reset()
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    _, _, terminateds, truncateds, _ = env.step(actions)
    assert "__all__" not in terminateds
    assert "__all__" not in truncateds


@pytest.mark.slow
def test_reset_returns_obs_and_infos(env):
    """Verify reset() returns (obs, infos) with correct agent keys."""
    obs, infos = env.reset()
    for agent in env.agents:
        assert agent in obs
        assert agent in infos
