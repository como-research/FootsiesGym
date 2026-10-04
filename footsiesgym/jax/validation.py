"""Bit-exact validation of the JAX backend against the Unity game server.

Drives N parallel ``BattleSimulation``s on the Unity server one frame at a
time and compares every ``BatchRawState`` field with the JAX port, or
records such a trace so it can be replayed without the binary. Also
provides ``footsies_env_on_jax_simulation`` for env-level tests.

    python -m footsiesgym.jax.validation --num-envs 256 --frames 20000
    python -m footsiesgym.jax.validation --record testing/data/unity_trace.npz
"""

import argparse
import os
import subprocess
import time
import types

import jax
import jax.numpy as jnp
import numpy as np

from .game import FootsiesGame

# Input bit patterns: none, L, R, A, L+A, R+A, L+R, L+R+A
INPUT_BITS = np.array([0, 1, 2, 4, 5, 6, 3, 7], dtype=np.int32)
# Hold durations chosen to exercise taps, dashes and 60-frame special charges.
HOLD_FRAMES = np.array([1, 2, 3, 5, 10, 20, 61, 70])
FLOAT_FIELDS = ("position_x", "velocity_x", "special_attack_progress")


def sticky_random_inputs(num_envs: int, num_frames: int, seed: int) -> np.ndarray:
    """(num_frames, num_envs, 2) input bits, each held for a random duration."""
    rng = np.random.default_rng(seed)
    current = rng.integers(0, len(INPUT_BITS), (num_envs, 2))
    hold = rng.integers(1, 80, (num_envs, 2))
    out = np.empty((num_frames, num_envs, 2), dtype=np.int32)
    for t in range(num_frames):
        hold -= 1
        expired = hold <= 0
        current[expired] = rng.integers(0, len(INPUT_BITS), expired.sum())
        hold[expired] = rng.choice(HOLD_FRAMES, expired.sum())
        out[t] = INPUT_BITS[current]
    return out


def raw_fields(raw) -> dict[str, np.ndarray]:
    """Comparable per-env fields of a BatchRawState (proto or namespace)."""
    names = list(jax.eval_shape(lambda: FootsiesGame.new().raw_state()))
    return {k: np.asarray(getattr(raw, k)) for k in names}


def mismatches(expected: dict, actual: dict) -> list[str]:
    """Fields that differ, comparing floats by bit pattern."""
    errors = []
    for k, want in expected.items():
        got = np.asarray(actual[k])
        if k.endswith(FLOAT_FIELDS):
            same = want.astype(np.float32).view(np.int32) == got.astype(
                np.float32
            ).view(np.int32)
        else:
            same = want.astype(np.int64) == got.astype(np.int64)
        if not same.all():
            i = int(np.flatnonzero(~same)[0])
            errors.append(f"{k}: env {i} expected {want[i]!r}, got {got[i]!r}")
    return errors


class JaxReplay:
    """Steps a batch of JAX simulations frame by frame."""

    def __init__(self, num_envs: int):
        self.games = jax.vmap(lambda _: FootsiesGame.new())(jnp.arange(num_envs))
        self._step = jax.jit(jax.vmap(FootsiesGame.step))
        self._raw = jax.jit(jax.vmap(FootsiesGame.raw_state))

    def step(self, inputs: np.ndarray):
        self.games = self._step(
            self.games, jnp.asarray(inputs[:, 0]), jnp.asarray(inputs[:, 1])
        )

    def fields(self) -> dict[str, np.ndarray]:
        return {
            k: np.asarray(v) for k, v in jax.device_get(self._raw(self.games)).items()
        }


class JaxSimulationServer:
    """Stand-in for the gRPC ``VectorizedFootsiesGame`` that runs the JAX
    simulation, mirroring ``VectorizedEnvironmentManager`` (including its
    in-step auto-reset). Lets ``FootsiesEnv``'s own Python env logic run on
    the JAX game, to check ``FootsiesJaxEnv`` against it without the binary.
    """

    def __init__(self, num_envs: int):
        self.num_envs = num_envs
        self.games = None
        self._dones = np.zeros(num_envs, dtype=bool)
        self._rewards = np.zeros(num_envs, dtype=np.int32)
        self._step_n = jax.jit(
            jax.vmap(FootsiesGame.step_n, in_axes=(0, 0, 0, None)), static_argnums=3
        )
        self._reset = jax.jit(
            jax.vmap(lambda game, reset: game.reset().where(reset, game))
        )
        self._raw = jax.jit(jax.vmap(FootsiesGame.raw_state))

    def start_and_init(self, timeout: float = 10.0):
        self.games = jax.vmap(lambda _: FootsiesGame.new())(jnp.arange(self.num_envs))

    def batch_step(self, p1_actions, p2_actions, n_frames: int):
        self.games = self._step_n(
            self.games,
            jnp.asarray(p1_actions, jnp.int32),
            jnp.asarray(p2_actions, jnp.int32),
            int(n_frames),
        )
        self._dones = np.asarray(self.games.done)
        self._rewards = np.asarray(self.games.reward)
        return self._response()

    def batch_reset(self, mask):
        mask = np.asarray(mask, dtype=bool)
        self.games = self._reset(self.games, jnp.asarray(mask))
        self._dones = np.where(mask, False, self._dones)
        self._rewards = np.where(mask, 0, self._rewards)
        return self._response()

    def batch_reset_all(self):
        return self.batch_reset(np.ones(self.num_envs, dtype=bool))

    def _response(self):
        fields = {
            k: np.asarray(v) for k, v in jax.device_get(self._raw(self.games)).items()
        }
        fields["dones"], fields["rewards"] = self._dones, self._rewards
        return types.SimpleNamespace(**fields)


def footsies_env_on_jax_simulation(config: dict):
    """A vectorized gRPC ``FootsiesEnv`` whose game server is the JAX simulation."""
    from footsiesgym.footsies.footsies_env import FootsiesEnv

    env = FootsiesEnv(dict(config, port=0))  # the gRPC channel is never used
    env.vec_game = JaxSimulationServer(env.num_envs)
    return env


def launch_server(binary_path: str, port: int) -> subprocess.Popen:
    proc = subprocess.Popen(
        [binary_path, "-batchmode", "--grpc", "--port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(5)
    return proc


def unity_frames(port: int, inputs: np.ndarray):
    """Yield BatchRawState fields after reset and after every frame."""
    from footsiesgym.footsies.game.footsies_game import VectorizedFootsiesGame

    vec = VectorizedFootsiesGame(port=port, num_envs=inputs.shape[1])
    vec.start_and_init()
    yield raw_fields(vec.batch_reset_all())
    for frame in inputs:
        yield raw_fields(
            vec.batch_step(
                frame[:, 0].astype(np.int64), frame[:, 1].astype(np.int64), 1
            )
        )


def compare_stream(inputs: np.ndarray, expected_frames) -> int:
    """Replay inputs in JAX against expected per-frame fields; returns #KOs."""
    replay = JaxReplay(inputs.shape[1])
    kos = 0
    for t, expected in enumerate(expected_frames):
        if t > 0:
            replay.step(inputs[t - 1])
            kos += int(expected["dones"].sum())
        errors = mismatches(expected, replay.fields())
        if errors:
            raise AssertionError(f"frame {t - 1}: " + "; ".join(errors))
    return kos


def default_binary() -> str:
    from footsiesgym.binary_manager import get_binary_manager

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rel = get_binary_manager().executable_relpath("linux", True)
    return os.path.join(root, "binaries", "footsies_binaries_headless", rel)


def main():
    import portpicker

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--frames", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--binary", default=None)
    parser.add_argument(
        "--record", default=None, help="write the Unity trace to this .npz"
    )
    args = parser.parse_args()

    inputs = sticky_random_inputs(args.num_envs, args.frames, args.seed)
    port = portpicker.pick_unused_port()
    proc = launch_server(args.binary or default_binary(), port)
    try:
        frames = unity_frames(port, inputs)
        if args.record:
            frames = list(frames)
            stacked = {k: np.stack([f[k] for f in frames]) for k in frames[0]}
            np.savez_compressed(args.record, inputs=inputs.astype(np.int8), **stacked)
            print(
                f"Recorded {args.frames} frames x {args.num_envs} envs to {args.record}"
            )
        kos = compare_stream(inputs, frames)
        print(
            f"OK: {args.num_envs} envs x {args.frames} frames bit-identical ({kos} KOs)"
        )
    finally:
        proc.terminate()


if __name__ == "__main__":
    main()
