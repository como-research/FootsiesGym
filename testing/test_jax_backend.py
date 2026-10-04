"""Bit-exactness and API tests for the JAX backend (footsiesgym.jax).

1. The JAX simulation equals Unity's BattleSimulation frame by frame
   (recorded trace; live server under the `slow` marker).
2. FootsiesJaxEnv.vectorize(N) equals the vectorized FootsiesEnv: on the
   JAX simulation (fast) and on the Unity server (`slow`), through each
   env's first episode (Unity carries some fighter state into the next).
3. Movement reproduces Unity's double-precision float math exactly.
4. FootsiesEnv(backend="jax") follows the single-env PettingZoo API, and
   FootsiesJaxEnv the functional API Sardine uses.
"""

import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import footsiesgym
from footsiesgym.footsies.footsies_env import FootsiesEnv
from footsiesgym.jax import fighter
from footsiesgym.jax import frame_data as fd
from footsiesgym.jax import validation
from footsiesgym.jax.env import FootsiesJaxEnv

TRACE = pathlib.Path(__file__).parent / "data" / "unity_trace_9c6b36f.npz"
AGENTS = ("p1", "p2")

ENV_CONFIGS = {
    "default": {},
    "no_delay_special_charge_truncation": dict(
        action_delay=0, use_special_charge_action=True, max_t=150
    ),
    "guard_break_budget": dict(
        guard_break_reward=0.1,
        use_reward_budget=True,
        use_special_charge_action=True,
        max_t=300,
        win_reward_scaling_coeff=0.7,
    ),
    "frame_skip_2": dict(frame_skip=2, action_delay=4, guard_break_reward=0.3),
}


def _bits(x):
    x = np.asarray(x)
    return x.view(np.int32) if x.dtype == np.float32 else x


def _sticky_actions(rng, num_envs, num_steps, num_actions):
    cur = rng.integers(0, num_actions, (num_envs, 2))
    hold = rng.integers(1, 20, (num_envs, 2))
    for _ in range(num_steps):
        hold -= 1
        expired = hold <= 0
        cur[expired] = rng.integers(0, num_actions, expired.sum())
        hold[expired] = rng.choice([1, 2, 3, 5, 8, 16, 20], expired.sum())
        yield {"p1": cur[:, 0].copy(), "p2": cur[:, 1].copy()}


def _assert_matches_vectorized_env(
    reference: FootsiesEnv, config: dict, num_steps: int
):
    """Compare FootsiesJaxEnv.vectorize(N) with a vectorized FootsiesEnv.

    Each env is compared through its first episode: on later episodes Unity
    carries some fighter state over from the previous one.
    """
    num_envs = reference.num_envs
    env = FootsiesJaxEnv(config).vectorize(num_envs)
    step = jax.jit(env.step)
    key = jax.random.key(0)

    ref_obs, _ = reference.reset()
    obs, state, _ = jax.jit(env.reset)(key)
    fresh_obs = {a: np.asarray(obs[a]) for a in AGENTS}
    for a in AGENTS:
        assert obs[a].shape == (num_envs, 88) and obs[a].dtype == jnp.float32
        np.testing.assert_array_equal(
            _bits(obs[a]), _bits(ref_obs[a]), err_msg=f"reset obs {a}"
        )

    live = np.ones(num_envs, dtype=bool)
    n_actions = reference.action_space("p1").n
    for t, act in enumerate(
        _sticky_actions(np.random.default_rng(0), num_envs, num_steps, n_actions)
    ):
        ref_obs, ref_rew, ref_term, ref_trunc, _ = reference.step(act)
        obs, state, rew, term, trunc, _ = step(
            key, state, {a: jnp.asarray(act[a]) for a in AGENTS}
        )
        done = np.asarray(term["p1"]) | np.asarray(trunc["p1"])
        for a in AGENTS:
            assert rew[a].shape == term[a].shape == trunc[a].shape == (num_envs,)
            r = np.asarray(rew[a])
            np.testing.assert_array_equal(
                r[live], ref_rew[a][live].astype(r.dtype), err_msg=f"t={t} reward {a}"
            )
            np.testing.assert_array_equal(
                np.asarray(term[a])[live], ref_term[a][live], err_msg=f"t={t} term"
            )
            np.testing.assert_array_equal(
                np.asarray(trunc[a])[live], ref_trunc[a][live], err_msg=f"t={t} trunc"
            )
            np.testing.assert_array_equal(
                _bits(obs[a])[live & ~done],
                _bits(ref_obs[a])[live & ~done],
                err_msg=f"t={t} obs {a}",
            )
            # A finished game is replaced by a fresh one in the same step.
            np.testing.assert_array_equal(
                _bits(obs[a])[done], _bits(fresh_obs[a])[done], err_msg=f"t={t} reset"
            )
        assert (np.asarray(state.t)[done] == 0).all()
        live &= ~done
    assert live.mean() < 0.1, "most envs should have finished an episode"


def test_simulation_matches_recorded_unity_trace():
    data = np.load(TRACE)
    inputs = data["inputs"].astype(np.int32)
    fields = [k for k in data.files if k != "inputs"]
    frames = ({k: data[k][t] for k in fields} for t in range(len(inputs) + 1))
    assert validation.compare_stream(inputs, frames) > 100  # many full rounds


@pytest.mark.parametrize("name", ENV_CONFIGS)
def test_jax_env_matches_vectorized_env(name):
    config = ENV_CONFIGS[name]
    reference = validation.footsies_env_on_jax_simulation(dict(config, num_envs=32))
    _assert_matches_vectorized_env(
        reference, config, num_steps=2000 // config.get("frame_skip", 4)
    )


def test_single_env_api():
    env = footsiesgym.make(config={"max_t": 300}, backend="jax")
    obs, infos = env.reset(seed=1)
    assert set(obs) == set(AGENTS) and set(infos) == set(AGENTS)
    assert obs["p1"].shape == (88,) and obs["p1"].dtype == np.float32

    # Same results as the functional env, without its automatic reset.
    ref_obs, ref_state, _ = env.jax_env.reset(jax.random.key(0))
    np.testing.assert_array_equal(obs["p1"], ref_obs["p1"])
    step_env = jax.jit(env.jax_env.step_env)

    rng = np.random.default_rng(0)
    for steps in range(1, 400):
        actions = {a: int(rng.integers(6)) for a in env.agents}
        obs, rewards, terms, truncs, infos = env.step(actions)
        ref_obs, ref_state, ref_rew, ref_term, ref_trunc, _ = step_env(
            jax.random.key(0), ref_state, {a: jnp.int32(v) for a, v in actions.items()}
        )
        assert isinstance(rewards["p1"], float) and isinstance(terms["p1"], bool)
        assert rewards["p1"] == float(ref_rew["p1"])
        assert terms["p1"] == bool(ref_term["p1"])
        np.testing.assert_array_equal(obs["p2"], ref_obs["p2"])
        assert rewards["p1"] + rewards["p2"] == 0
        if terms["p1"] or truncs["p1"]:
            break
    assert env.agents == []
    env.reset()
    assert env.agents == list(AGENTS)


def test_sardine_environment_contract():
    """The interface Sardine's Anakin runner uses, checked without Sardine."""
    env = FootsiesJaxEnv(frame_skip=4, max_t=50)  # keyword config, as Sardine passes it
    assert env.possible_agents == list(AGENTS) and env.max_t == 50
    assert env.obs_dim("p1") == 88 and env.action_dim("p1") == 6
    assert env.observation_spaces["p1"].shape == (88,)

    vec = env.vectorize(8)
    assert vec.num_envs == 8 and vec.obs_dim == 88 and vec.action_dim == 6
    assert vec.possible_agents == env.possible_agents and vec.max_t == 50

    obs, state, infos = jax.jit(vec.reset)(jax.random.key(0))
    assert obs["p1"].shape == (8, 88) and set(infos) == set(AGENTS)
    assert state.t.shape == (8,) and state.action_mask is None
    assert state.active_agents is None

    def rollout(state, key):
        def body(state, key):
            actions = {a: jnp.zeros(8, jnp.int32) for a in AGENTS}
            _, state, rewards, term, trunc, _ = vec.step(key, state, actions)
            return state, (state.t, term["p1"] | trunc["p1"])

        return jax.lax.scan(body, state, jax.random.split(key, 120))

    state, (t, done) = jax.jit(rollout)(state, jax.random.key(1))
    done, t = np.asarray(done), np.asarray(t)
    assert done[49].all() and (t[49] == 0).all() and (t[48] == 49).all()  # max_t
    assert (np.asarray(state.episode_id) == 2).all()


def test_movement_is_exact():
    """Movement equals Mono's double-precision update, for every speed.

    Covers every 1009th float32 position in [-8, 8], plus the positions where
    rounding to float64 first matters. (A GPU run of every float32 position
    in [-8, 8] found no mismatches either.)
    """
    data = fd.FRAME_DATA
    speeds = {float(v) for v in data.movement_velocity.ravel() if v != 0}
    speeds |= {float(fd.FORWARD_MOVE_SPEED), -float(fd.BACKWARD_MOVE_SPEED)}
    magnitudes = np.arange(0, np.float32(8).view(np.int32), 1009, dtype=np.int32)
    positions = magnitudes.view(np.float32)
    move = jax.jit(fighter._move)
    for speed in sorted(speeds | {-v for v in speeds}):
        delta = fd.exact_delta(np.float32(speed))
        tiny = delta[2]
        p = np.concatenate(
            [positions, -positions, [tiny, -tiny], np.nextafter([tiny, -tiny], 0)]
        ).astype(np.float32)
        p = np.concatenate([p, np.nextafter(p, np.inf), np.nextafter(p, -np.inf)])
        product = np.float64(np.float32(speed)) * np.float64(fd.FIXED_DELTA_TIME)
        expected = (p.astype(np.float64) + product).astype(np.float32)
        got = np.asarray(move(jnp.asarray(p), jnp.asarray(delta), jnp.float32(1)))
        np.testing.assert_array_equal(
            got.view(np.int32), expected.view(np.int32), err_msg=f"speed {speed}"
        )


def test_jax_backend_rejects_num_envs():
    with pytest.raises(ValueError, match="vmap"):
        FootsiesEnv({"num_envs": 4}, backend="jax")
    with pytest.raises(RuntimeError):
        FootsiesEnv({"port": 0}).jax_step


@pytest.mark.slow
def test_simulation_matches_live_unity_server():
    import portpicker

    inputs = validation.sticky_random_inputs(num_envs=32, num_frames=2000, seed=123)
    port = portpicker.pick_unused_port()
    proc = validation.launch_server(validation.default_binary(), port)
    try:
        assert (
            validation.compare_stream(inputs, validation.unity_frames(port, inputs)) > 0
        )
    finally:
        proc.terminate()


@pytest.mark.slow
@pytest.mark.parametrize("name", ["default", "guard_break_budget"])
def test_jax_env_matches_unity_env(name):
    config = ENV_CONFIGS[name]
    unity = FootsiesEnv(dict(config, num_envs=32, launch_binaries=True))
    try:
        _assert_matches_vectorized_env(unity, config, num_steps=500)
    finally:
        unity.close()
