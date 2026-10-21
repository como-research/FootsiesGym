"""Bit-exact validation of the C backend against the Unity game.

  * against the recorded Unity trace (the bar for byte-identical):
        python -m footsiesgym.c.validation trace
  * against a live Unity game server, every BatchRawState field after every
    frame, with sticky random inputs (needs the headless binary):
        python -m footsiesgym.c.validation unity --num-envs 256 --frames 20000
  * record a new trace from the live server:
        python -m footsiesgym.c.validation unity --record testing/data/unity_trace.npz

Also provides ``footsies_env_on_c_simulation``: the vectorized ``FootsiesEnv``
(its own Python env logic) with the C simulation as its game server, so the
C env can be checked against it without the binary.
"""

import argparse
import os
import pathlib
import subprocess
import time
import types

import numpy as np

from .game import BatchGames, CGames
from .lib import RAW_STATE, raw_to_fields

TRACE = (
    pathlib.Path(__file__).resolve().parents[2]
    / "testing"
    / "data"
    / "unity_trace_9c6b36f.npz"
)

# Input bit patterns: none, L, R, A, L+A, R+A, L+R, L+R+A
INPUT_BITS = np.array([0, 1, 2, 4, 5, 6, 3, 7], dtype=np.int32)
# Hold durations chosen to exercise taps, dashes and 60-frame special charges.
HOLD_FRAMES = np.array([1, 2, 3, 5, 10, 20, 61, 70])
FLOAT_FIELDS = ("position_x", "velocity_x", "special_attack_progress")
FIELDS = tuple(raw_to_fields(np.zeros(1, RAW_STATE)))


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
    return {k: np.asarray(getattr(raw, k)) for k in FIELDS}


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


def compare_stream(inputs: np.ndarray, expected_frames, replay_cls=None) -> int:
    """Replay inputs in C against expected per-frame fields; returns #KOs."""
    replay = (replay_cls or CGames)(inputs.shape[1])
    kos = 0
    for t, expected in enumerate(expected_frames):
        if t > 0:
            replay.step(inputs[t - 1])
            kos += int(expected["dones"].sum())
        errors = mismatches(expected, replay.fields())
        if errors:
            raise AssertionError(f"frame {t - 1}: " + "; ".join(errors))
    return kos


def compare_trace(path=TRACE, replay_cls=None) -> int:
    """Replay a recorded Unity trace; returns the number of KOs."""
    data = np.load(path)
    inputs = data["inputs"].astype(np.int32)
    fields = [k for k in data.files if k != "inputs"]
    frames = ({k: data[k][t] for k in fields} for t in range(len(inputs) + 1))
    return compare_stream(inputs, frames, replay_cls)


class CSimulationServer:
    """Stand-in for the gRPC ``VectorizedFootsiesGame`` that runs the C
    simulation, mirroring ``VectorizedEnvironmentManager`` (including its
    in-step auto-reset). Lets ``FootsiesEnv``'s own Python env logic run on
    the C game, to check ``FootsiesCEnv`` against it without the binary.
    """

    def __init__(self, num_envs: int):
        self.num_envs = num_envs
        self.games = None
        self._dones = np.zeros(num_envs, dtype=bool)
        self._rewards = np.zeros(num_envs, dtype=np.int32)

    def start_and_init(self, timeout: float = 10.0):
        self.games = CGames(self.num_envs)

    def batch_step(self, p1_actions, p2_actions, n_frames: int):
        inputs = np.stack([p1_actions, p2_actions], axis=1).astype(np.int32)
        self.games.step(inputs, int(n_frames))
        fields = self.games.fields()
        self._dones = fields["dones"].astype(bool)
        self._rewards = fields["rewards"].astype(np.int32)
        return self._response(fields)

    def batch_reset(self, mask):
        mask = np.asarray(mask, dtype=bool)
        self.games.reset(mask)
        self._dones = np.where(mask, False, self._dones)
        self._rewards = np.where(mask, 0, self._rewards)
        return self._response(self.games.fields())

    def batch_reset_all(self):
        return self.batch_reset(np.ones(self.num_envs, dtype=bool))

    def _response(self, fields):
        fields = dict(fields, dones=self._dones, rewards=self._rewards)
        return types.SimpleNamespace(**fields)


def footsies_env_on_c_simulation(config: dict):
    """A vectorized gRPC ``FootsiesEnv`` whose game server is the C simulation."""
    from footsiesgym.footsies.footsies_env import FootsiesEnv

    env = FootsiesEnv(dict(config, port=0))  # the gRPC channel is never used
    env.vec_game = CSimulationServer(env.num_envs)
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


def default_binary() -> str:
    """The headless Linux server, downloaded and extracted if needed."""
    from footsiesgym.binary_manager import get_binary_manager

    manager = get_binary_manager()
    binaries = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "binaries"
    )
    if not manager.ensure_binaries_extracted("linux", binaries, headless=True):
        raise FileNotFoundError("could not download and extract the footsies binary")
    rel = manager.executable_relpath("linux", True)
    return os.path.join(binaries, "footsies_binaries_headless", rel)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    parser.add_argument(
        "--impl", choices=("scalar", "batch"), default="batch", help="C implementation"
    )
    trace = sub.add_parser("trace", help="replay the recorded Unity trace")
    trace.add_argument("--path", default=str(TRACE))
    unity = sub.add_parser("unity", help="compare with a live Unity game server")
    unity.add_argument("--num-envs", type=int, default=64)
    unity.add_argument("--frames", type=int, default=5000)
    unity.add_argument("--seed", type=int, default=0)
    unity.add_argument("--binary", default=None)
    unity.add_argument(
        "--record", default=None, help="write the Unity trace to this .npz"
    )
    args = parser.parse_args()

    replay_cls = BatchGames if args.impl == "batch" else CGames
    if args.cmd == "trace":
        kos = compare_trace(args.path, replay_cls)
        print(f"OK: Unity trace bit-identical ({kos} KOs)")
        return

    import portpicker

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
        kos = compare_stream(inputs, frames, replay_cls)
        print(
            f"OK: {args.num_envs} envs x {args.frames} frames bit-identical "
            f"to Unity ({kos} KOs)"
        )
    finally:
        proc.terminate()


if __name__ == "__main__":
    main()
