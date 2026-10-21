"""Bit-exactness tests for the C/CUDA backend (csrc/, footsiesgym.c).

The validation ladder:
1. mono_move equals Mono's double-precision movement for float32 positions
   in [-8, 8] (strided here; every position under the `long` marker).
2. The C simulation equals Unity's BattleSimulation frame by frame on the
   recorded trace: the bar for byte-identical (and on a live Unity server
   under the `slow` marker).
3. The C env equals the vectorized FootsiesEnv (observations, rewards, done
   flags) with action delay, special charge, KOs, truncation and resets:
   FootsiesEnv's own env logic on the C simulation (fast), and FootsiesEnv
   on the Unity server (`slow`). make(backend="c") equals FootsiesCEnv on
   every device and tensor_type, and so does the XLA FFI env; the
   single-game env follows the PettingZoo API.
4. Builds agree byte for byte: other compilers / -march=native on the CPU,
   and the CUDA build (when a GPU is available) against the CPU build.
"""

import importlib.util
import os
import shutil
import subprocess

import numpy as np
import pytest

import footsiesgym
from footsiesgym.c import CGames, FootsiesCEnv, build, cuda
from footsiesgym.c import frame_data as fd
from footsiesgym.c import validation as cval
from footsiesgym.c.game import BatchGames
from footsiesgym.c.lib import Library, load, ptr
from footsiesgym.footsies.footsies_env import FootsiesEnv

try:
    import jax
    import jax.numpy as jnp

    from footsiesgym.c import FootsiesFFIEnv
except ImportError:  # JAX is optional: only the FFI env needs it
    jax = None

requires_jax = pytest.mark.skipif(jax is None, reason="needs JAX")
requires_cuda = pytest.mark.skipif(
    not cuda.available(), reason="needs a CUDA GPU and nvcc"
)

AGENTS = ("p1", "p2")

# Exhaustive / large-scale checks take minutes: FOOTSIES_LONG_TESTS=1 runs them.
long_test = pytest.mark.skipif(
    not os.environ.get("FOOTSIES_LONG_TESTS"), reason="set FOOTSIES_LONG_TESTS=1"
)

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
    "frame_skip_1_long_delay": dict(
        frame_skip=1, action_delay=8, use_special_charge_action=True, max_t=1000
    ),
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
        yield cur.astype(np.int32)


# ── 0. Generated frame data ──────────────────────────────────────────────


def test_generated_frame_data_is_up_to_date():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "gen_c_frame_data", build.CSRC.parent / "scripts" / "gen_c_frame_data.py"
    )
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    # render() also checks the authored ranges against the dense tables.
    assert (build.CSRC / "frame_data.gen.h").read_text() == gen.render()


def test_fast_math_is_rejected(tmp_path):
    cc = shutil.which("cc") or shutil.which("gcc")
    if cc is None:
        pytest.skip("no C compiler")
    result = subprocess.run(
        [cc, "-ffast-math", "-fsyntax-only", str(build.CSRC / "footsies_cpu.c")],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0 and "fast-math" in result.stderr


# ── 1. Movement ──────────────────────────────────────────────────────────


def _speeds():
    speeds = {float(v) for v in fd.FRAME_DATA.movement_velocity.ravel() if v != 0}
    speeds |= {float(fd.FORWARD_MOVE_SPEED), -float(fd.BACKWARD_MOVE_SPEED)}
    return sorted(speeds | {-v for v in speeds})


def _check_move(lib, positions, speed):
    positions = np.ascontiguousarray(positions, np.float32)
    got = np.empty_like(positions)
    lib.mono_move(
        ptr(positions), ptr(got), np.float32(speed), np.float32(1), positions.size
    )
    product = np.float64(np.float32(speed)) * np.float64(fd.FIXED_DELTA_TIME)
    expected = (positions.astype(np.float64) + product).astype(np.float32)
    bad = got.view(np.int32) != expected.view(np.int32)
    if bad.any():
        i = int(np.flatnonzero(bad)[0])
        raise AssertionError(
            f"speed {speed}, position {positions[i]!r}: {got[i]!r} != {expected[i]!r}"
        )


def test_movement_is_exact():
    """Every 1009th float32 in [-8, 8], plus where rounding to float64 matters."""
    lib = load()
    magnitudes = np.arange(0, np.float32(8).view(np.int32), 1009, dtype=np.int32)
    positions = magnitudes.view(np.float32)
    for speed in _speeds():
        tiny = fd.exact_delta(np.float32(speed))[2]
        p = np.concatenate(
            [positions, -positions, [tiny, -tiny], np.nextafter([tiny, -tiny], 0)]
        ).astype(np.float32)
        p = np.concatenate([p, np.nextafter(p, np.inf), np.nextafter(p, -np.inf)])
        _check_move(lib, p, speed)


@long_test
def test_movement_is_exact_for_every_position():
    """Every float32 position in [-8, 8], for every speed."""
    lib = load()
    top = int(np.float32(8).view(np.int32))
    chunk = 1 << 24
    for speed in _speeds():
        for start in range(0, top + 1, chunk):
            magnitudes = np.arange(start, min(start + chunk, top + 1), dtype=np.int32)
            positions = magnitudes.view(np.float32)
            _check_move(lib, positions, speed)
            _check_move(lib, -positions, speed)


# ── 2. Unity simulation ──────────────────────────────────────────────────


@pytest.mark.parametrize("replay", [CGames, BatchGames], ids=["scalar", "batch"])
def test_simulation_matches_recorded_unity_trace(replay):
    assert cval.compare_trace(replay_cls=replay) > 100  # many full rounds


@pytest.mark.slow
@pytest.mark.parametrize("replay", [CGames, BatchGames], ids=["scalar", "batch"])
def test_simulation_matches_live_unity_server(replay):
    import portpicker

    inputs = cval.sticky_random_inputs(num_envs=32, num_frames=2000, seed=123)
    port = portpicker.pick_unused_port()
    proc = cval.launch_server(cval.default_binary(), port)
    try:
        frames = cval.unity_frames(port, inputs)
        assert cval.compare_stream(inputs, frames, replay) > 0
    finally:
        proc.terminate()


# ── 3. Environment ───────────────────────────────────────────────────────


def _assert_matches_vectorized_env(reference: FootsiesEnv, config, num_steps, impl):
    """Compare FootsiesCEnv with a vectorized FootsiesEnv.

    Each env is compared through its first episode: on later episodes Unity
    carries some fighter state over from the previous one.
    """
    num_envs = reference.num_envs
    env = FootsiesCEnv(num_envs, config, impl=impl)

    ref_obs, _ = reference.reset()
    obs = env.reset()
    fresh_obs = {a: obs[a].copy() for a in AGENTS}
    for a in AGENTS:
        assert obs[a].shape == (num_envs, 88) and obs[a].dtype == np.float32
        np.testing.assert_array_equal(
            _bits(obs[a]), _bits(ref_obs[a]), err_msg=f"reset obs {a}"
        )

    live = np.ones(num_envs, dtype=bool)
    n_actions = reference.action_space("p1").n
    rng = np.random.default_rng(0)
    for t, act in enumerate(_sticky_actions(rng, num_envs, num_steps, n_actions)):
        actions = {"p1": act[:, 0].copy(), "p2": act[:, 1].copy()}
        ref_obs, ref_rew, ref_term, ref_trunc, _ = reference.step(actions)
        obs, rew, term, trunc = env.step(actions)
        done = term["p1"] | trunc["p1"]
        for a in AGENTS:
            np.testing.assert_array_equal(
                rew[a][live],
                ref_rew[a][live].astype(np.float32),
                err_msg=f"t={t} reward {a}",
            )
            np.testing.assert_array_equal(
                term[a][live], ref_term[a][live], err_msg=f"t={t} term"
            )
            np.testing.assert_array_equal(
                trunc[a][live], ref_trunc[a][live], err_msg=f"t={t} trunc"
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
        assert (env.info()["t"][done] == 0).all()
        live &= ~done
    assert live.mean() < 0.1, "most envs should have finished an episode"


@pytest.mark.parametrize("impl", ["scalar", "batch"])
@pytest.mark.parametrize("name", ENV_CONFIGS)
def test_env_matches_footsies_env_on_c_simulation(name, impl):
    config = ENV_CONFIGS[name]
    reference = cval.footsies_env_on_c_simulation(dict(config, num_envs=40))
    _assert_matches_vectorized_env(
        reference, config, 2000 // config.get("frame_skip", 4), impl
    )


@pytest.mark.slow
@pytest.mark.parametrize("name", ["default", "guard_break_budget"])
def test_env_matches_unity_env(name):
    config = ENV_CONFIGS[name]
    unity = FootsiesEnv(dict(config, num_envs=32, launch_binaries=True))
    try:
        _assert_matches_vectorized_env(unity, config, 500, "batch")
    finally:
        unity.close()


def test_single_env_api():
    env = footsiesgym.make(config={"max_t": 300}, backend="c")
    obs, infos = env.reset(seed=1)
    assert set(obs) == set(AGENTS) and set(infos) == set(AGENTS)
    assert obs["p1"].shape == (88,) and obs["p1"].dtype == np.float32

    # Same results as the batched env, without its automatic reset.
    ref = FootsiesCEnv(1, max_t=300)
    ref_obs = ref.reset()
    np.testing.assert_array_equal(obs["p1"], ref_obs["p1"][0])

    rng = np.random.default_rng(0)
    for _ in range(400):
        actions = {a: int(rng.integers(6)) for a in env.agents}
        obs, rewards, terms, truncs, infos = env.step(actions)
        ref_obs, ref_rew, ref_term, _ = ref.step(
            {a: np.array([v], np.int32) for a, v in actions.items()}, autoreset=False
        )
        assert isinstance(rewards["p1"], float) and isinstance(terms["p1"], bool)
        assert rewards["p1"] == float(ref_rew["p1"][0])
        assert terms["p1"] == bool(ref_term["p1"][0])
        np.testing.assert_array_equal(obs["p2"], ref_obs["p2"][0])
        assert rewards["p1"] + rewards["p2"] == 0
        if terms["p1"] or truncs["p1"]:
            break
    assert env.agents == []
    env.reset()
    assert env.agents == list(AGENTS)


def test_backend_config_errors():
    with pytest.raises(ValueError, match="backend='c'"):
        FootsiesEnv({}, backend="jax")
    with pytest.raises(ValueError, match="device only applies"):
        FootsiesEnv({"port": 0}, device="cuda")
    with pytest.raises(ValueError, match="tensor_type"):
        footsiesgym.make(backend="c", tensor_type="tensorflow")
    with pytest.raises(ValueError, match="device"):
        footsiesgym.make(backend="c", device="gpu")


def test_make_keywords_and_config_are_equivalent():
    a = footsiesgym.make(config={"backend": "c", "num_envs": 8, "max_t": 50})
    b = footsiesgym.make(backend="c", num_envs=8, max_t=50)
    c = footsiesgym.make(config={"backend": "c"}, num_envs=8, max_t=50)
    assert a.config == b.config == c.config
    with pytest.raises(ValueError, match="'num_envs' is set twice"):
        footsiesgym.make(config={"backend": "c", "num_envs": 8}, num_envs=16)


def _requires(tensor_type, device):
    marks = []
    if tensor_type != "numpy":
        marks.append(
            pytest.mark.skipif(
                importlib.util.find_spec(tensor_type) is None,
                reason=f"needs {tensor_type}",
            )
        )
    if device == "cuda":
        marks.append(requires_cuda)
    return marks


MAKE_CASES = [
    pytest.param(d, t, marks=_requires(t, d), id=f"{d}-{t}")
    for d in ("cpu", "cuda")
    for t in ("numpy", "torch", "jax")
]


def _to_host(x):
    return np.asarray(x.cpu() if hasattr(x, "cpu") else x)


def _actions_as(act, tensor_type, device):
    """(N, 2) int actions as a {"p1", "p2"} dict of tensor_type arrays."""
    if tensor_type == "torch":
        import torch

        a = torch.from_numpy(act.astype(np.int64)).to(device)
        return {"p1": a[:, 0], "p2": a[:, 1]}
    if tensor_type == "jax":
        import jax

        dev = jax.devices("gpu" if device == "cuda" else "cpu")[0]
        a = jax.device_put(act, dev)
        return {"p1": a[:, 0], "p2": a[:, 1]}
    return {"p1": act[:, 0], "p2": act[:, 1]}


@pytest.mark.parametrize("device,tensor_type", MAKE_CASES)
def test_make_c_env_matches_c_env(device, tensor_type):
    """make(backend="c") on every device and tensor_type: the vectorized
    FootsiesEnv API, equal to FootsiesCEnv byte for byte, outputs never
    overwritten by later steps."""
    config = ENV_CONFIGS["guard_break_budget"]
    num_envs = 1000
    env = footsiesgym.make(
        config, backend="c", num_envs=num_envs, device=device, tensor_type=tensor_type
    )
    ref = FootsiesCEnv(num_envs, config)
    obs, infos = env.reset()
    ref_obs = ref.reset()
    assert set(infos) == set(AGENTS)
    for a in AGENTS:
        assert _bits(_to_host(obs[a])).tobytes() == _bits(ref_obs[a]).tobytes()

    kept, rng = None, np.random.default_rng(1)
    for t, act in enumerate(_sticky_actions(rng, num_envs, 400, 9)):
        out = env.step(_actions_as(act, tensor_type, device))
        want = [dict(x) for x in ref.step(act)]
        assert out[4] == {a: {} for a in AGENTS}
        for got, exp in zip(out[:4], want):
            for a in AGENTS:
                g = _to_host(got[a])
                assert g.shape == exp[a].shape and g.dtype == exp[a].dtype
                assert _bits(g).tobytes() == _bits(exp[a]).tobytes(), f"t={t}"
        if t == 10:
            kept = (out[:4], [{a: x[a].copy() for a in AGENTS} for x in want])
    for got, exp in zip(*kept):
        for a in AGENTS:
            assert _bits(_to_host(got[a])).tobytes() == _bits(exp[a]).tobytes()


@requires_jax
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=requires_cuda)])
def test_functional_matches_make_env(device):
    """make(functional=True) inside jit + lax.scan equals stepping make()'s env."""
    config = dict(ENV_CONFIGS["guard_break_budget"], num_envs=256)
    f = footsiesgym.make(config, backend="c", functional=True)
    ref = footsiesgym.make(config, backend="c")
    rng = np.random.default_rng(3)
    acts = np.stack(list(_sticky_actions(rng, 256, 300, 9)))  # (T, N, 2)

    dev = jax.devices("gpu" if device == "cuda" else "cpu")[0]
    with jax.default_device(dev):

        @jax.jit
        def rollout(actions):
            obs, state, _ = f.reset(None)

            def body(state, a):
                obs, state, rew, term, trunc, _ = f.step(
                    None, state, {"p1": a[:, 0], "p2": a[:, 1]}
                )
                return state, (obs["p1"], rew["p2"], term["p1"])

            return jax.lax.scan(body, state, actions)

        state, (obs, rew, term) = rollout(jnp.asarray(acts))

    ref.reset()
    for t in range(len(acts)):
        r_obs, r_rew, r_term, _, _ = ref.step(
            {"p1": acts[t, :, 0], "p2": acts[t, :, 1]}
        )
        assert _bits(np.asarray(obs[t])).tobytes() == _bits(r_obs["p1"]).tobytes()
        assert _bits(np.asarray(rew[t])).tobytes() == _bits(r_rew["p2"]).tobytes()
        np.testing.assert_array_equal(np.asarray(term[t]), r_term["p1"])
    assert np.asarray(term).sum() > 0
    assert f.num_envs == 256 and f.possible_agents == list(AGENTS)


def test_make_class():
    """make_class: instances are make()'s envs, settings merged and fixed."""
    cls = footsiesgym.make_class(backend="c", max_t=50)
    env = cls(num_envs=8)
    assert isinstance(env, FootsiesEnv)
    assert env.config == footsiesgym.make(backend="c", max_t=50, num_envs=8).config
    assert cls({"num_envs": 8}).config == env.config
    with pytest.raises(ValueError, match="'max_t' is set twice"):
        cls(max_t=60)
    with pytest.raises(ValueError, match="functional is fixed"):
        cls(functional=True)
    with pytest.raises(ValueError, match="functional=True needs backend='c'"):
        footsiesgym.make_class(functional=True)


@requires_jax
def test_make_class_functional_for_sardine():
    """Sardine builds Cls(**env_config).vectorize(num_envs)."""
    cls = footsiesgym.make_class(backend="c", functional=True, max_t=50)
    env = cls(frame_skip=4)  # env_config, as Sardine passes it
    assert env.max_t == 50 and env.config["frame_skip"] == 4
    vec = env.vectorize(8)
    want = footsiesgym.make(backend="c", functional=True, num_envs=8, max_t=50)
    obs, state, _ = jax.jit(vec.reset)(None)
    ref_obs, ref_state, _ = jax.jit(want.reset)(None)
    actions = {a: jnp.ones(8, jnp.int32) for a in AGENTS}
    for _ in range(60):
        obs, state, rew, *_ = jax.jit(vec.step)(None, state, actions)
        ref_obs, ref_state, ref_rew, *_ = jax.jit(want.step)(None, ref_state, actions)
    assert np.asarray(state.data).tobytes() == np.asarray(ref_state.data).tobytes()
    assert (np.asarray(state.episode_id) == 1).all()  # truncated at max_t=50
    with pytest.raises(ValueError, match="vectorize"):
        footsiesgym.make_class(backend="c", functional=True, num_envs=8)


def test_possible_agents_on_every_class():
    """possible_agents is a class attribute of every env class."""
    classes = [FootsiesEnv, footsiesgym.make_class(backend="c")]
    if jax is not None:
        classes.append(footsiesgym.make_class(backend="c", functional=True))
        classes.append(type(footsiesgym.make(backend="c", functional=True)))
    for cls in classes:
        assert cls.possible_agents == list(AGENTS), cls


def test_rllib_is_a_wrapper_not_a_setting():
    with pytest.raises(ValueError, match="wrap_rllib"):
        footsiesgym.make(backend="c", rllib=True)
    with pytest.raises(ValueError, match="wrap_rllib"):
        footsiesgym.make_class(backend="c", rllib=True)


@pytest.mark.skipif(importlib.util.find_spec("ray") is None, reason="needs ray")
@pytest.mark.parametrize("num_envs", [1, 4])
def test_wrap_rllib(num_envs):
    """wrap_rllib turns the PettingZoo env (single or vectorized) into an
    RLlib MultiAgentEnv with the same games."""
    from ray.rllib.env.multi_agent_env import MultiAgentEnv

    from footsiesgym.wrappers import wrap_rllib

    env = wrap_rllib(footsiesgym.make(backend="c", max_t=30, num_envs=num_envs))
    ref = footsiesgym.make(backend="c", max_t=30, num_envs=num_envs)
    assert isinstance(env, MultiAgentEnv)
    obs, _ = env.reset()
    ref_obs, _ = ref.reset()
    want_agents = (
        list(AGENTS)
        if num_envs == 1
        else [f"{p}_{i}" for i in range(num_envs) for p in AGENTS]
    )
    assert sorted(env.possible_agents) == sorted(want_agents) == sorted(obs)
    for _ in range(40):
        obs, rew, term, trunc, _ = env.step({a: 1 for a in env.agents})
        if num_envs == 1:
            r = ref.step({a: 1 for a in AGENTS})
            np.testing.assert_array_equal(obs["p1"], r[0]["p1"])
            if term["__all__"] or trunc["__all__"]:
                break
        else:
            r = ref.step({a: np.ones(num_envs, np.int64) for a in AGENTS})
            for i in range(num_envs):
                np.testing.assert_array_equal(obs[f"p1_{i}"], r[0]["p1"][i])
                assert rew[f"p2_{i}"] == float(r[1]["p2"][i])
    with pytest.raises(ValueError, match="numpy"):
        wrap_rllib(footsiesgym.make(backend="c", num_envs=4, tensor_type="torch"))


@requires_jax
def test_functional_config_errors():
    with pytest.raises(ValueError, match="backend='c'"):
        footsiesgym.make(functional=True)
    with pytest.raises(ValueError, match="tensor_type"):
        footsiesgym.make(backend="c", functional=True, tensor_type="torch")
    with pytest.raises(ValueError, match="jax.default_device"):
        footsiesgym.make(backend="c", functional=True, device="cuda")


@requires_cuda
@requires_jax
@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="needs torch")
def test_jax_env_takes_torch_gpu_actions():
    import torch

    env = footsiesgym.make(backend="c", num_envs=64, device="cuda", tensor_type="jax")
    ref = footsiesgym.make(backend="c", num_envs=64)
    env.reset(), ref.reset()
    rng = np.random.default_rng(4)
    for act in _sticky_actions(rng, 64, 100, 6):
        a = torch.from_numpy(act.astype(np.int64)).cuda()
        got = env.step({"p1": a[:, 0], "p2": a[:, 1]})
        want = ref.step({"p1": act[:, 0], "p2": act[:, 1]})
        for g, w in zip(got[:4], want[:4]):
            for p in AGENTS:
                assert _bits(np.asarray(g[p])).tobytes() == _bits(w[p]).tobytes()


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="needs torch")
def test_grpc_env_tensor_type():
    """tensor_type on the gRPC env (its env logic on the C simulation here)."""
    import torch

    config = dict(ENV_CONFIGS["default"], num_envs=16)
    plain = cval.footsies_env_on_c_simulation(config)
    typed = cval.footsies_env_on_c_simulation(dict(config, tensor_type="torch"))
    for got, want in zip(typed.reset(), plain.reset()):
        for a in AGENTS:
            if isinstance(want[a], np.ndarray):
                assert isinstance(got[a], torch.Tensor)
                np.testing.assert_array_equal(got[a].numpy(), want[a])
    rng = np.random.default_rng(2)
    for act in _sticky_actions(rng, 16, 50, 6):
        got = typed.step({"p1": torch.from_numpy(act[:, 0]), "p2": act[:, 1]})
        want = plain.step({"p1": act[:, 0], "p2": act[:, 1]})
        for g, w in zip(got[:4], want[:4]):
            for a in AGENTS:
                assert isinstance(g[a], torch.Tensor)
                np.testing.assert_array_equal(g[a].numpy(), w[a])


def _ffi_matches_c_env(config, num_envs, num_steps, autoreset):
    """FootsiesFFIEnv equals FootsiesCEnv byte for byte, t and episode_id too."""
    env = FootsiesFFIEnv(config).vectorize(num_envs)
    step = jax.jit(env.step if autoreset else env.step_env, donate_argnums=1)
    obs, state, _ = jax.jit(env.reset)(jax.random.key(0))
    ref = FootsiesCEnv(num_envs, config)
    ref_obs = ref.reset()
    for a in AGENTS:
        np.testing.assert_array_equal(_bits(obs[a]), _bits(ref_obs[a]), "reset")

    episodes = 0
    rng = np.random.default_rng(0)
    for t, act in enumerate(_sticky_actions(rng, num_envs, num_steps, env.action_dim)):
        actions = {"p1": act[:, 0].copy(), "p2": act[:, 1].copy()}
        obs, state, rew, term, trunc, _ = step(
            None, state, {a: jnp.asarray(v) for a, v in actions.items()}
        )
        ref_obs, ref_rew, ref_term, ref_trunc = ref.step(actions, autoreset=autoreset)
        for a in AGENTS:
            for name, want, got in (
                ("obs", ref_obs[a], obs[a]),
                ("reward", ref_rew[a], rew[a]),
                ("terminated", ref_term[a], term[a]),
                ("truncated", ref_trunc[a], trunc[a]),
            ):
                np.testing.assert_array_equal(
                    _bits(got), _bits(want), err_msg=f"t={t} {name} {a}"
                )
        info = ref.info()
        np.testing.assert_array_equal(np.asarray(state.t), info["t"], f"t={t} t")
        np.testing.assert_array_equal(
            np.asarray(state.episode_id), info["episode_id"], f"t={t} episode_id"
        )
        episodes += int(ref_term["p1"].sum() + ref_trunc["p1"].sum())
    assert episodes >= num_envs, "envs should cross episode boundaries"


@requires_jax
@pytest.mark.parametrize("autoreset", [True, False], ids=["step", "step_env"])
@pytest.mark.parametrize("name", ["default", "guard_break_budget"])
def test_ffi_env_matches_c_env(name, autoreset):
    with jax.default_device(jax.devices("cpu")[0]):
        _ffi_matches_c_env(ENV_CONFIGS[name], 32, 500, autoreset)


@requires_jax
def test_ffi_env_sardine_contract():
    """The interface Sardine's Anakin runner uses."""
    env = FootsiesFFIEnv(frame_skip=4, max_t=50)
    assert env.possible_agents == list(AGENTS) and env.max_t == 50
    assert env.obs_dim("p1") == 88 and env.action_dim("p1") == 6
    vec = env.vectorize(8)
    assert vec.num_envs == 8 and vec.obs_dim == 88 and vec.action_dim == 6

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


def test_env_rejects_bad_config():
    with pytest.raises(ValueError, match="divisible"):
        FootsiesCEnv(4, frame_skip=4, action_delay=6)
    with pytest.raises(ValueError, match="MAX_DELAY_STEPS"):
        FootsiesCEnv(4, frame_skip=1, action_delay=40)


# ── 4. Builds agree byte for byte ────────────────────────────────────────


def _rollout_bytes(
    lib, num_envs=256, num_steps=600, impl="batch", name="guard_break_budget"
):
    """Env state, obs and rewards after each step, from one library build."""
    config = ENV_CONFIGS[name]
    env = FootsiesCEnv(num_envs, config, lib=lib, impl=impl)
    env.reset()
    rng = np.random.default_rng(5)
    for act in _sticky_actions(rng, num_envs, num_steps, 9):
        obs, rew, term, trunc = env.step(act)
        yield b"".join(
            x.tobytes()
            for x in (
                env.state,
                obs["p1"],
                obs["p2"],
                rew["p1"],
                rew["p2"],
                term["p1"],
                trunc["p1"],
            )
        )


def _other_builds():
    builds = {
        "gcc -march=native": dict(extra_flags=("-march=native",)),
        # GNU mode on AVX-512 FP16 CPUs sets FLT_EVAL_METHOD to 16
        "gcc -march=native -std=gnu11": dict(
            extra_flags=("-march=native", "-std=gnu11")
        ),
    }
    if shutil.which("clang"):
        builds["clang"] = dict(compiler=shutil.which("clang"))
        builds["clang -march=native"] = dict(
            compiler=shutil.which("clang"), extra_flags=("-march=native",)
        )
    return builds


@pytest.mark.parametrize("name", list(_other_builds()))
def test_cpu_builds_agree(name):
    """Other compilers and -march=native (which enables FMA) change nothing."""
    other = Library(build.build_cpu(**_other_builds()[name]))
    for impl in ("scalar", "batch"):
        default = _rollout_bytes(load(), impl=impl)
        for t, (a, b) in enumerate(zip(default, _rollout_bytes(other, impl=impl))):
            assert a == b, f"{name} ({impl}) differs at step {t}"


BATCH_BUILDS = {
    "default": {},  # AVX-512 when the CPU has it
    "portable lane code": dict(extra_flags=("-DFOOTSIES_NO_AVX512",)),
    "x86-64-v3 (AVX2)": dict(extra_flags=("-march=x86-64-v3",)),
}


@pytest.mark.parametrize("build_name", list(BATCH_BUILDS))
@pytest.mark.parametrize("name", ENV_CONFIGS)
def test_batch_matches_scalar_bytes(name, build_name):
    """The SIMD batches equal footsies.h byte for byte: state, obs, rewards."""
    lib = Library(build.build_cpu(**BATCH_BUILDS[build_name]))
    scalar = _rollout_bytes(lib, num_envs=1000, num_steps=800, impl="scalar", name=name)
    batch = _rollout_bytes(lib, num_envs=1000, num_steps=800, impl="batch", name=name)
    for t, (a, b) in enumerate(zip(scalar, batch)):
        assert a == b, f"step {t}"


@requires_cuda
@pytest.mark.parametrize(
    "name", ["default", "guard_break_budget", "frame_skip_1_long_delay"]
)
def test_cuda_env_matches_cpu_env_bytes(name):
    config = ENV_CONFIGS[name]
    num_envs = 4096
    cpu = FootsiesCEnv(num_envs, config)
    gpu = cuda.CudaEnv(num_envs, config)
    cpu.reset()
    gpu.reset()
    assert gpu.state.download().tobytes() == cpu.state.tobytes()
    rng = np.random.default_rng(6)
    for t, act in enumerate(_sticky_actions(rng, num_envs, 1000, 9)):
        obs, rew, term, trunc = cpu.step(act)
        gpu.step(act)
        out = gpu.outputs()
        assert (
            out["state"].tobytes() == cpu.state.tobytes()
        ), f"state differs at step {t}"
        np.testing.assert_array_equal(
            out["obs"].view(np.int32), cpu._obs.view(np.int32)
        )
        np.testing.assert_array_equal(
            out["rew"].view(np.int32), cpu._rew.view(np.int32)
        )
        np.testing.assert_array_equal(out["term"], cpu._term)
        np.testing.assert_array_equal(out["trunc"], cpu._trunc)


@requires_cuda
def test_cuda_frames_match_cpu_frames_bytes():
    num_games, num_frames = 4096, 20000
    inputs = cval.sticky_random_inputs(num_games, num_frames, seed=7)
    cpu, gpu = CGames(num_games), cuda.CudaGames(num_games)
    for t in range(num_frames):
        cpu.step(inputs[t])
        gpu.step(inputs[t])
        if t % 10 == 0 or t == num_frames - 1:
            assert gpu.state_bytes().tobytes() == cpu.state.tobytes(), f"frame {t}"


@requires_cuda
@requires_jax
def test_cuda_ffi_env_matches_c_env():
    with jax.default_device(jax.devices("gpu")[0]):
        _ffi_matches_c_env(ENV_CONFIGS["guard_break_budget"], 64, 500, True)
