"""Throughput benchmark for the JAX FOOTSIES backend.

Steps ``FootsiesJaxEnv(...).vectorize(num_envs)`` the way a JAX training
loop (such as Sardine's) does: random actions sampled on device, finished
games reset inside ``step``, and ``--scan-steps`` steps per jitted
``lax.scan`` call. Rows go to the same CSV as ``bench_throughput.py`` (label
``jax_<device>``), so ``plot_throughput.py --label <label>`` can plot them.

Examples:

    python benchmarking/bench_jax_throughput.py --num-envs 1,16,256,4096,65536
    JAX_PLATFORMS=cpu python benchmarking/bench_jax_throughput.py --num-envs 1,256,4096
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import jax
import numpy as np
from bench_throughput import (
    DEFAULT_CSV,
    BenchResult,
    append_result,
    ci95_half_width,
    parse_n_list,
    print_result,
)

from footsiesgym.jax import FootsiesJaxEnv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num-envs", default="1,16,256,1024,4096,16384,65536")
    p.add_argument("--duration", type=float, default=10.0)
    p.add_argument("--bucket", type=float, default=1.0)
    p.add_argument(
        "--scan-steps", type=int, default=128, help="env steps per jitted call"
    )
    p.add_argument("--frame-skip", type=int, default=4)
    p.add_argument("--action-delay", type=int, default=0)
    p.add_argument("--max-t", type=int, default=10_000)
    p.add_argument("--label", default=None, help="defaults to jax_<device>")
    p.add_argument("--csv", default=str(DEFAULT_CSV))
    args = p.parse_args()
    if args.label is None:
        args.label = f"jax_{jax.default_backend()}"
    return args


def make_rollout(env, scan_steps: int):
    n_actions = env.action_dim

    @jax.jit
    def rollout(state, key):
        def body(carry, _):
            state, key = carry
            key, k1, k2, k_step = jax.random.split(key, 4)
            actions = {
                "p1": jax.random.randint(k1, (env.num_envs,), 0, n_actions),
                "p2": jax.random.randint(k2, (env.num_envs,), 0, n_actions),
            }
            _, state, rewards, _, _, _ = env.step(k_step, state, actions)
            return (state, key), rewards["p1"].sum()

        (state, key), rew = jax.lax.scan(body, (state, key), None, length=scan_steps)
        return state, key, rew.sum()

    return rollout


def bench(args, num_envs: int) -> tuple[list[int], list[float]]:
    config = {
        "frame_skip": args.frame_skip,
        "action_delay": args.action_delay,
        "max_t": args.max_t,
    }
    env = FootsiesJaxEnv(config).vectorize(num_envs)
    rollout = make_rollout(env, args.scan_steps)
    key = jax.random.key(0)
    _, state, _ = jax.jit(env.reset)(key)
    state, key, total = rollout(state, key)  # compile
    total.block_until_ready()

    bucket_calls, bucket_walls = [], []
    t0 = bucket_start = time.perf_counter()
    calls = 0
    while True:
        state, key, total = rollout(state, key)
        total.block_until_ready()
        calls += args.scan_steps
        now = time.perf_counter()
        if now - bucket_start >= args.bucket:
            bucket_calls.append(calls)
            bucket_walls.append(now - bucket_start)
            bucket_start, calls = now, 0
        if now - t0 >= args.duration:
            break
    return bucket_calls, bucket_walls


def summarize(args, num_envs: int, bucket_calls, bucket_walls) -> BenchResult:
    env_sps = [num_envs * c / w for c, w in zip(bucket_calls, bucket_walls)]
    call_sps = [c / w for c, w in zip(bucket_calls, bucket_walls)]
    total_calls = int(sum(bucket_calls))
    return BenchResult(
        label=args.label,
        config=f"{args.label}_N{num_envs}",
        num_envs=num_envs,
        num_processes=1,
        frame_skip=args.frame_skip,
        action_delay=args.action_delay,
        step_calls=total_calls,
        env_steps=num_envs * total_calls,
        wall_seconds=float(sum(bucket_walls)),
        trials=len(bucket_calls),
        env_steps_per_sec=float(np.mean(env_sps)),
        env_steps_per_sec_ci95=ci95_half_width(env_sps),
        step_calls_per_sec=float(np.mean(call_sps)),
        step_calls_per_sec_ci95=ci95_half_width(call_sps),
    )


def main() -> None:
    args = parse_args()
    print(f"JAX {jax.__version__} on {jax.devices()}")
    for n in parse_n_list(args.num_envs):
        result = summarize(args, n, *bench(args, n))
        append_result(result, Path(args.csv))
        print_result(result)


if __name__ == "__main__":
    main()
