"""Bit-exactness tests for the C/CUDA backend (csrc/, footsiesgym.c).

The validation ladder:
1. mono_move equals Mono's double-precision movement for float32 positions
   in [-8, 8] (strided here; every position under the `long` marker).
2. The C simulation equals Unity's BattleSimulation frame by frame on the
   recorded trace: the bar for byte-identical.
3. It equals the JAX port on every BatchRawState field after every frame,
   and the C env equals FootsiesJaxEnv (observations, rewards, done flags,
   t, episode_id) with action delay, special charge, KOs, truncation and
   resets; directly (numpy) and as an XLA FFI call.
4. Builds agree byte for byte: other compilers / -march=native on the CPU,
   and the CUDA build (when a GPU is available) against the CPU build.
"""

import os
import shutil
import subprocess

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from footsiesgym.c import CGames, FootsiesCEnv, FootsiesFFIEnv, build, cuda
from footsiesgym.c import validation as cval
from footsiesgym.c.game import BatchGames
from footsiesgym.c.lib import Library, load, ptr
from footsiesgym.jax import frame_data as fd
from footsiesgym.jax.env import FootsiesJaxEnv

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
    # render() also checks the authored ranges against the dense JAX tables.
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


# ── 2. Unity trace ───────────────────────────────────────────────────────


@pytest.mark.parametrize("replay", [CGames, BatchGames], ids=["scalar", "batch"])
def test_simulation_matches_recorded_unity_trace(replay):
    assert cval.compare_trace(replay_cls=replay) > 100  # many full rounds


# ── 3. JAX port: frames, env, FFI ────────────────────────────────────────


@pytest.mark.parametrize("replay", [CGames, BatchGames], ids=["scalar", "batch"])
def test_simulation_matches_jax_frame_by_frame(replay):
    kos = cval.compare_with_jax(
        num_envs=500, num_frames=4000, seed=3, replay_cls=replay
    )
    assert kos > 1000


@long_test
def test_simulation_matches_jax_at_scale():
    kos = cval.compare_with_jax(4096, 50000, seed=4, replay_cls=BatchGames)
    assert kos > 10**5


def _assert_env_matches_jax(config, num_envs, num_steps, autoreset, make_c_env):
    jenv = FootsiesJaxEnv(config).vectorize(num_envs)
    jstep = jax.jit(jenv.step if autoreset else jenv.step_env)
    key = jax.random.key(0)
    jobs, jstate, _ = jax.jit(jenv.reset)(key)
    c_reset, c_step, c_info = make_c_env(config, num_envs, autoreset)
    cobs = c_reset()
    for a in AGENTS:
        np.testing.assert_array_equal(_bits(cobs[a]), _bits(jobs[a]), err_msg="reset")

    episodes = 0
    rng = np.random.default_rng(0)
    for t, act in enumerate(_sticky_actions(rng, num_envs, num_steps, jenv.action_dim)):
        actions = {"p1": act[:, 0], "p2": act[:, 1]}
        jobs, jstate, jrew, jterm, jtrunc, _ = jstep(
            key, jstate, {a: jnp.asarray(v) for a, v in actions.items()}
        )
        cobs, crew, cterm, ctrunc = c_step(actions)
        for a in AGENTS:
            for name, want, got in (
                ("obs", jobs[a], cobs[a]),
                ("reward", jrew[a], crew[a]),
                ("terminated", jterm[a], cterm[a]),
                ("truncated", jtrunc[a], ctrunc[a]),
            ):
                np.testing.assert_array_equal(
                    _bits(got), _bits(want), err_msg=f"t={t} {name} {a}"
                )
        t_c, episode_c = c_info()
        np.testing.assert_array_equal(t_c, np.asarray(jstate.t), err_msg=f"t={t} t")
        np.testing.assert_array_equal(
            episode_c, np.asarray(jstate.episode_id), err_msg=f"t={t} episode_id"
        )
        episodes += int(np.asarray(jterm["p1"] | jtrunc["p1"]).sum())
    assert episodes >= num_envs, "envs should cross episode boundaries"


def _numpy_env(config, num_envs, autoreset, impl="batch"):
    env = FootsiesCEnv(num_envs, config, impl=impl)

    def step(actions):
        obs, rew, term, trunc = env.step(actions, autoreset=autoreset)
        return obs, rew, term, trunc

    def info():
        i = env.info()
        return i["t"], i["episode_id"]

    return env.reset, step, info


def _ffi_env(config, num_envs, autoreset):
    env = FootsiesFFIEnv(config).vectorize(num_envs)
    step_fn = jax.jit(env.step if autoreset else env.step_env, donate_argnums=1)
    holder = {}

    def reset():
        obs, holder["state"], _ = jax.jit(env.reset)(jax.random.key(0))
        return obs

    def step(actions):
        obs, holder["state"], rew, term, trunc, _ = step_fn(
            None, holder["state"], {a: jnp.asarray(v) for a, v in actions.items()}
        )
        return obs, rew, term, trunc

    def info():
        return np.asarray(holder["state"].t), np.asarray(holder["state"].episode_id)

    return reset, step, info


@pytest.mark.parametrize("autoreset", [True, False], ids=["step", "step_env"])
@pytest.mark.parametrize("name", ENV_CONFIGS)
def test_env_matches_jax_env(name, autoreset):
    config = ENV_CONFIGS[name]
    steps = 2000 // config.get("frame_skip", 4)
    _assert_env_matches_jax(
        config, 40, steps, autoreset, _numpy_env
    )  # 40: a partial block


@pytest.mark.parametrize("name", ["default", "guard_break_budget"])
def test_scalar_env_matches_jax_env(name):
    make = lambda c, n, a: _numpy_env(c, n, a, impl="scalar")
    _assert_env_matches_jax(ENV_CONFIGS[name], 32, 500, True, make)


@pytest.mark.parametrize("autoreset", [True, False], ids=["step", "step_env"])
@pytest.mark.parametrize("name", ["default", "guard_break_budget"])
def test_ffi_env_matches_jax_env(name, autoreset):
    _assert_env_matches_jax(ENV_CONFIGS[name], 32, 500, autoreset, _ffi_env)


def test_ffi_env_sardine_contract():
    """The interface Sardine's Anakin runner uses (as for FootsiesJaxEnv)."""
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


requires_cuda = pytest.mark.skipif(
    not cuda.available(), reason="needs a CUDA GPU and nvcc"
)


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
    from footsiesgym.jax.validation import sticky_random_inputs

    num_games, num_frames = 4096, 20000
    inputs = sticky_random_inputs(num_games, num_frames, seed=7)
    cpu, gpu = CGames(num_games), cuda.CudaGames(num_games)
    for t in range(num_frames):
        cpu.step(inputs[t])
        gpu.step(inputs[t])
        if t % 10 == 0 or t == num_frames - 1:
            assert gpu.state_bytes().tobytes() == cpu.state.tobytes(), f"frame {t}"


@requires_cuda
def test_cuda_ffi_env_matches_jax_env():
    devices = jax.devices("gpu")
    with jax.default_device(devices[0]):
        _assert_env_matches_jax(
            ENV_CONFIGS["guard_break_budget"], 64, 500, True, _ffi_env
        )
