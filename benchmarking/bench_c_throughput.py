"""Throughput benchmark for the C/CUDA FOOTSIES backend (footsiesgym.c).

Four ways to step ``num_envs`` games with uniformly random actions,
resetting finished games inside ``step``:

  * ``make``: ``footsiesgym.make(backend="c", device=..., tensor_type=...)``,
    what users get: new output arrays every step, actions given as
    ``--tensor-type`` arrays on ``--device``.
  * ``cpu``:  ``FootsiesCEnv`` (OpenMP over preallocated numpy buffers;
    ``--impl batch`` steps 16 games per SIMD instruction, ``--impl scalar``
    one game at a time, ``--impl auto`` picks like ``FootsiesCEnv``).
    Actions come from a pre-generated pool so the RNG is not measured.
  * ``ffi``:  ``FootsiesFFIEnv`` inside a jitted ``lax.scan`` (actions
    sampled on device), the way a JAX training loop drives it. Runs on JAX's
    default backend.
  * ``cuda``: the CUDA kernel alone, through its C API; actions come from a
    pool already on the device.

Rows go to the same CSV as ``bench_throughput.py`` (label ``c_<backend>``).

Examples:

    python benchmarking/bench_c_throughput.py --backend make --device cuda --tensor-type torch
    python benchmarking/bench_c_throughput.py --backend cpu --num-envs 1,256,4096,65536
    python benchmarking/bench_c_throughput.py --backend ffi --num-envs 1024,16384,262144
    python benchmarking/bench_c_throughput.py --backend cuda --num-envs 16384,65536,262144
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from bench_throughput import (
    DEFAULT_CSV,
    BenchResult,
    append_result,
    ci95_half_width,
    parse_n_list,
    print_result,
)

ACTION_POOL = 64  # distinct random action batches, cycled


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=("make", "cpu", "ffi", "cuda"), default="cpu")
    p.add_argument("--device", default="cpu", help="make: cpu, cuda or cuda:N")
    p.add_argument(
        "--tensor-type", choices=("numpy", "torch", "jax"), default="numpy", help="make"
    )
    p.add_argument("--num-envs", default="1,16,256,1024,4096,16384,65536")
    p.add_argument("--duration", type=float, default=10.0)
    p.add_argument("--bucket", type=float, default=1.0)
    p.add_argument(
        "--scan-steps", type=int, default=128, help="ffi: steps per jitted call"
    )
    p.add_argument("--threads", type=int, default=None, help="cpu: OpenMP threads")
    p.add_argument(
        "--impl", choices=("auto", "batch", "scalar"), default="auto", help="cpu"
    )
    p.add_argument("--frame-skip", type=int, default=4)
    p.add_argument("--action-delay", type=int, default=0)
    p.add_argument("--max-t", type=int, default=10_000)
    p.add_argument("--label", default=None, help="defaults to c_<backend>")
    p.add_argument("--csv", default=str(DEFAULT_CSV))
    args = p.parse_args()
    if args.backend == "cpu" and args.impl == "auto":
        from footsiesgym.c.lib import load

        args.impl = "batch" if load().batch_avx512() else "scalar"
    if args.label is None:
        args.label = f"c_{args.backend}"
        if args.backend == "ffi":
            import jax

            args.label += f"_{jax.default_backend()}"
        if args.backend == "make":
            args.label += f"_{args.device.replace(':', '')}_{args.tensor_type}"
        if args.backend in ("cpu", "make") and args.threads:
            args.label += f"_T{args.threads}"
        if args.backend == "cpu":
            args.label += f"_{args.impl}"
    return args


def config(args) -> dict:
    return {
        "frame_skip": args.frame_skip,
        "action_delay": args.action_delay,
        "max_t": args.max_t,
    }


def timed(step, steps_per_call: int, args) -> tuple[list[int], list[float]]:
    """Call ``step()`` (which must block) for ``--duration`` seconds."""
    step()  # warm up (and compile)
    bucket_calls, bucket_walls = [], []
    t0 = bucket_start = time.perf_counter()
    calls = 0
    while True:
        step()
        calls += steps_per_call
        now = time.perf_counter()
        if now - bucket_start >= args.bucket:
            bucket_calls.append(calls)
            bucket_walls.append(now - bucket_start)
            bucket_start, calls = now, 0
        if now - t0 >= args.duration:
            break
    return bucket_calls, bucket_walls


def bench_make(args, num_envs: int):
    import footsiesgym

    env = footsiesgym.make(
        config(args),
        backend="c",
        num_envs=num_envs,
        device=args.device,
        tensor_type=args.tensor_type,
        num_threads=args.threads,
    )
    env.reset()
    rng = np.random.default_rng(0)
    host = rng.integers(0, 6, (ACTION_POOL, 2, num_envs))
    if args.tensor_type == "torch":
        import torch

        dev = "cpu" if args.device == "cpu" else args.device
        pool = torch.from_numpy(host).to(dev)
        sync = torch.cuda.synchronize if dev != "cpu" else (lambda: None)
    elif args.tensor_type == "jax":
        import jax

        dev = jax.devices("cpu" if args.device == "cpu" else "gpu")[0]
        pool = jax.device_put(host, dev)
        sync = None  # block on each step's outputs (JAX dispatches async)
    else:
        pool, sync = host, lambda: None
    actions = [{"p1": pool[k, 0], "p2": pool[k, 1]} for k in range(ACTION_POOL)]
    if num_envs == 1:  # the single-game API: scalar actions, no automatic reset
        actions = [{a: int(v[a][0]) for a in v} for v in actions]
    i = [0]

    def step():
        out = env.step(actions[i[0] % ACTION_POOL])
        if num_envs == 1:
            if out[2]["p1"] or out[3]["p1"]:
                env.reset()
        elif sync is None:
            jax.block_until_ready(out[:4])
        else:
            sync()
        i[0] += 1

    return timed(step, 1, args)


def bench_cpu(args, num_envs: int):
    from footsiesgym.c import FootsiesCEnv

    env = FootsiesCEnv(num_envs, config(args), num_threads=args.threads, impl=args.impl)
    env.reset()
    rng = np.random.default_rng(0)
    pool = rng.integers(0, env.num_actions, (ACTION_POOL, num_envs, 2), dtype=np.int32)
    i = [0]

    def step():
        env.step(pool[i[0] % ACTION_POOL])
        i[0] += 1

    return timed(step, 1, args)


def bench_ffi(args, num_envs: int):
    import jax

    from footsiesgym.c import FootsiesFFIEnv

    env = FootsiesFFIEnv(config(args)).vectorize(num_envs)
    n_actions = env.action_dim

    def rollout(state, key):
        def body(carry, _):
            state, key = carry
            key, k1, k2, k_step = jax.random.split(key, 4)
            actions = {
                "p1": jax.random.randint(k1, (num_envs,), 0, n_actions),
                "p2": jax.random.randint(k2, (num_envs,), 0, n_actions),
            }
            _, state, rewards, _, _, _ = env.step(k_step, state, actions)
            return (state, key), rewards["p1"].sum()

        (state, key), rew = jax.lax.scan(
            body, (state, key), None, length=args.scan_steps
        )
        return state, key, rew.sum()

    rollout = jax.jit(rollout, donate_argnums=0)
    _, state, _ = jax.jit(env.reset)(jax.random.key(0))
    carry = [state, jax.random.key(0)]

    def step():
        carry[0], carry[1], total = rollout(carry[0], carry[1])
        total.block_until_ready()

    return timed(step, args.scan_steps, args)


def bench_cuda(args, num_envs: int):
    from footsiesgym.c import cuda
    from footsiesgym.c.lib import ptr

    env = cuda.CudaEnv(num_envs, config(args))
    env.reset()
    rng = np.random.default_rng(0)
    num_actions = 6
    pool = cuda.DeviceArray(env.lib, np.zeros((ACTION_POOL, num_envs, 2), np.int32))
    pool.upload(
        rng.integers(0, num_actions, (ACTION_POOL, num_envs, 2), dtype=np.int32)
    )
    stride = num_envs * 2 * 4
    steps_per_call = 64
    i = [0]

    def step():
        for _ in range(steps_per_call):
            env.lib.check(
                env.lib.env_step(
                    ptr(env._config),
                    env.state.ptr,
                    pool.ptr + (i[0] % ACTION_POOL) * stride,
                    env.obs.ptr,
                    env.rew.ptr,
                    env.term.ptr,
                    env.trunc.ptr,
                    1,
                    num_envs,
                )
            )
            i[0] += 1
        env.synchronize()

    return timed(step, steps_per_call, args)


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
    run = {"make": bench_make, "cpu": bench_cpu, "ffi": bench_ffi, "cuda": bench_cuda}[
        args.backend
    ]
    for n in parse_n_list(args.num_envs):
        result = summarize(args, n, *run(args, n))
        append_result(result, Path(args.csv))
        print_result(result)


if __name__ == "__main__":
    main()
