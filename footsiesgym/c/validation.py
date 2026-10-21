"""Bit-exact validation of the C backend.

  * against the recorded Unity trace (the bar for byte-identical):
        python -m footsiesgym.c.validation trace
  * against the JAX port, every BatchRawState field after every frame, with
    sticky random inputs (``footsiesgym.jax.validation.sticky_random_inputs``):
        python -m footsiesgym.c.validation jax --num-envs 4096 --frames 200000

``CReplay`` has the ``step``/``fields`` interface of ``JaxReplay``, so the
JAX port's ``mismatches`` (floats compared by bit pattern) is reused as is.
"""

import argparse
import pathlib
import time

import numpy as np

from .game import BatchGames, CGames

CReplay = CGames

TRACE = (
    pathlib.Path(__file__).resolve().parents[2]
    / "testing"
    / "data"
    / "unity_trace_9c6b36f.npz"
)


def compare_trace(path=TRACE, replay_cls=None) -> int:
    """Replay a recorded Unity trace; returns the number of KOs."""
    from footsiesgym.jax.validation import mismatches

    data = np.load(path)
    inputs = data["inputs"].astype(np.int32)
    fields = [k for k in data.files if k != "inputs"]
    replay = (replay_cls or CReplay)(inputs.shape[1])
    kos = 0
    for t in range(len(inputs) + 1):
        if t > 0:
            replay.step(inputs[t - 1])
            kos += int(data["dones"][t].sum())
        expected = {k: data[k][t] for k in fields}
        errors = mismatches(expected, replay.fields())
        if errors:
            raise AssertionError(f"frame {t - 1}: " + "; ".join(errors))
    return kos


def _jax_chunk_fn():
    """jit(scan) over frames: (games, inputs[T, N, 2]) -> (games, raw[T])."""
    import jax

    from footsiesgym.jax.game import FootsiesGame

    step = jax.vmap(FootsiesGame.step)
    raw = jax.vmap(FootsiesGame.raw_state)

    def chunk(games, inputs):
        def body(games, frame):
            games = step(games, frame[:, 0], frame[:, 1])
            return games, raw(games)

        return jax.lax.scan(body, games, inputs)

    return jax.jit(chunk)


def compare_with_jax(
    num_envs: int,
    num_frames: int,
    seed: int = 0,
    chunk: int = 256,
    log_every: int = 0,
    replay_cls=None,
) -> int:
    """Step the C and JAX simulations on the same inputs and compare every
    BatchRawState field after every frame; returns the number of KOs."""
    import jax
    import jax.numpy as jnp

    from footsiesgym.jax.game import FootsiesGame
    from footsiesgym.jax.validation import mismatches, sticky_random_inputs

    inputs = sticky_random_inputs(num_envs, num_frames, seed)
    games = jax.vmap(lambda _: FootsiesGame.new())(jnp.arange(num_envs))
    run = _jax_chunk_fn()
    replay = (replay_cls or CReplay)(num_envs)
    names = list(replay.fields())
    kos, start = 0, time.time()
    for t0 in range(0, num_frames, chunk):
        frames = inputs[t0 : t0 + chunk]
        games, raw = run(games, jnp.asarray(frames))
        raw = {k: np.asarray(v) for k, v in jax.device_get(raw).items()}
        for i, frame in enumerate(frames):
            replay.step(frame)
            got = replay.fields()
            expected = {k: raw[k][i] for k in names}
            errors = mismatches(expected, got)
            if errors:
                raise AssertionError(f"frame {t0 + i}: " + "; ".join(errors))
            kos += int(expected["dones"].sum())
        if log_every and (t0 // chunk) % log_every == 0:
            done = t0 + len(frames)
            print(
                f"  {done}/{num_frames} frames, {kos} KOs, {time.time() - start:.0f}s",
                flush=True,
            )
    return kos


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    parser.add_argument(
        "--impl", choices=("scalar", "batch"), default="batch", help="C implementation"
    )
    trace = sub.add_parser("trace", help="replay the recorded Unity trace")
    trace.add_argument("--path", default=str(TRACE))
    vs_jax = sub.add_parser("jax", help="differential test against the JAX port")
    vs_jax.add_argument("--num-envs", type=int, default=1024)
    vs_jax.add_argument("--frames", type=int, default=20000)
    vs_jax.add_argument("--seed", type=int, default=0)
    vs_jax.add_argument("--chunk", type=int, default=256)
    args = parser.parse_args()

    replay_cls = BatchGames if args.impl == "batch" else CGames
    if args.cmd == "trace":
        kos = compare_trace(args.path, replay_cls)
        print(f"OK: Unity trace bit-identical ({kos} KOs)")
    else:
        kos = compare_with_jax(
            args.num_envs, args.frames, args.seed, args.chunk, 20, replay_cls
        )
        print(
            f"OK: {args.num_envs} envs x {args.frames} frames bit-identical "
            f"to JAX ({kos} KOs)"
        )


if __name__ == "__main__":
    main()
