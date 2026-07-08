"""Throughput benchmark for FootsiesEnv (headless, single-env and vectorized).

Drives ``footsiesgym.make(...)``: the env downloads, launches, and tears
down the headless Unity binary on its own. The script just sweeps
``num_envs``, times ``env.step()`` in a tight loop, and appends rows to a
shared CSV.

Single-env vs vectorized service selection is automatic inside FootsiesEnv
based on ``num_envs`` (single-env at N=1, VectorizedFootsiesService at N>1).

Examples (Linux box with `footsies-paper` conda env active):

    # Headless single-env baseline:
    python scripts/bench_throughput.py --num-envs 1 --label headless_singleenv

    # Vectorized scaling sweep (same headless binary):
    python scripts/bench_throughput.py \\
        --num-envs 2,4,16,64,256,1024 --label vectorized_headless

    # Multi-process sweep: 1..4 independent game servers, each with N envs,
    # stepped in parallel from Python threads. Reports aggregate throughput.
    python scripts/bench_throughput.py \\
        --num-envs 64,128 --num-processes 4 --label multiproc
"""

from __future__ import annotations

import argparse
import csv
import math
import platform
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

import footsiesgym

# Two-sided 95% t critical values for df = 1..30. df > 30 falls back to z=1.96.
_T_CRIT_95 = [
    12.706,
    4.303,
    3.182,
    2.776,
    2.571,
    2.447,
    2.365,
    2.306,
    2.262,
    2.228,
    2.201,
    2.179,
    2.160,
    2.145,
    2.131,
    2.120,
    2.110,
    2.101,
    2.093,
    2.086,
    2.080,
    2.074,
    2.069,
    2.064,
    2.060,
    2.056,
    2.052,
    2.048,
    2.045,
    2.042,
]


def t_crit_95(df: int) -> float:
    if df <= 0:
        return float("nan")
    if df <= len(_T_CRIT_95):
        return _T_CRIT_95[df - 1]
    return 1.96


def ci95_half_width(samples: list[float]) -> float:
    n = len(samples)
    if n < 2:
        return float("nan")
    arr = np.asarray(samples, dtype=float)
    sem = arr.std(ddof=1) / math.sqrt(n)
    return t_crit_95(n - 1) * sem


BENCH_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = BENCH_DIR / "figures" / "data" / "throughput.csv"

CSV_FIELDS = [
    "label",
    "config",
    "num_envs",
    "num_processes",
    "frame_skip",
    "action_delay",
    "step_calls",
    "env_steps",
    "wall_seconds",
    "trials",
    "env_steps_per_sec",
    "env_steps_per_sec_ci95",
    "step_calls_per_sec",
    "step_calls_per_sec_ci95",
]


@dataclass
class BenchResult:
    label: str
    config: str
    num_envs: int
    num_processes: int
    frame_skip: int
    action_delay: int
    step_calls: int
    env_steps: int
    wall_seconds: float
    trials: int
    env_steps_per_sec: float
    env_steps_per_sec_ci95: float
    step_calls_per_sec: float
    step_calls_per_sec_ci95: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--num-envs",
        default="1,2,4,8,16,32,64,128",
        help="Single int or comma-separated list, e.g. '1' or '2,4,16,32,64,128,256'.",
    )
    p.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Total wall-clock seconds of timed stepping per --num-envs value.",
    )
    p.add_argument(
        "--bucket",
        type=float,
        default=1.0,
        help="Bucket size (s) for splitting the timed window into samples for the 95%% CI.",
    )
    p.add_argument(
        "--num-processes",
        default="1,2,3,4",
        help="Single int or comma-separated list of process counts. Each value runs that many independent game-server processes in parallel, each with --num-envs envs; aggregate throughput is reported.",
    )
    p.add_argument("--frame-skip", type=int, default=4)
    p.add_argument(
        "--action-delay",
        type=int,
        default=0,
        help="Frames of action delay (must divide frame_skip).",
    )
    p.add_argument(
        "--warmup",
        type=int,
        default=20,
        help="Untimed step calls before the timing loop.",
    )
    p.add_argument(
        "--max-t",
        type=int,
        default=10_000,
        help="Max steps per episode before truncation (kept large so resets don't dominate the bench).",
    )
    p.add_argument(
        "--label",
        default="bench",
        help="CSV tag for this run. Row config is '<label>' for N=1, else '<label>_N{N}'.",
    )
    p.add_argument(
        "--platform",
        default="mac" if platform.system() == "Darwin" else "linux",
        choices=["linux", "mac"],
        help="Forwarded to footsiesgym.make(). Defaults to the current OS.",
    )
    p.add_argument(
        "--no-launch-binaries",
        dest="launch_binaries",
        action="store_false",
        help="Don't have the env launch the Unity binary; assume a manually-started server on --port.",
    )
    p.set_defaults(launch_binaries=True)
    p.add_argument(
        "--port",
        type=int,
        default=None,
        help="Server port. Only meaningful with --no-launch-binaries (otherwise the env auto-picks).",
    )
    p.add_argument(
        "--host",
        default="localhost",
        help="Server host. Only meaningful with --no-launch-binaries.",
    )
    p.add_argument(
        "--csv",
        default=str(DEFAULT_CSV),
        help="CSV to append results to.",
    )
    return p.parse_args()


def parse_n_list(spec: str) -> list[int]:
    out = [int(tok) for tok in spec.split(",") if tok.strip()]
    if not out:
        raise SystemExit("--num-envs must contain at least one integer")
    return out


def sample_actions(env, num_envs: int, rng: np.random.Generator) -> dict[str, Any]:
    """Build the action dict expected by FootsiesEnv for either mode."""
    n_actions = env.action_space("p1").n
    if num_envs == 1:
        return {
            "p1": int(rng.integers(0, n_actions)),
            "p2": int(rng.integers(0, n_actions)),
        }
    return {
        "p1": rng.integers(0, n_actions, size=num_envs, dtype=np.int64),
        "p2": rng.integers(0, n_actions, size=num_envs, dtype=np.int64),
    }


def build_config(args: argparse.Namespace, num_envs: int) -> dict[str, Any]:
    config: dict[str, Any] = {
        "num_envs": num_envs,
        "headless": True,
        "frame_skip": args.frame_skip,
        "action_delay": args.action_delay,
        "max_t": args.max_t,
    }
    # The env auto-picks an unused port when none is supplied; only pass
    # explicit host/port when the user opted out of the launch path.
    if not args.launch_binaries:
        config["host"] = args.host
        if args.port is not None:
            config["port"] = args.port
    return config


def _step_loop_one(
    env,
    num_envs: int,
    args: argparse.Namespace,
    rng: np.random.Generator,
    barrier: threading.Barrier | None,
) -> tuple[list[int], list[float]]:
    """Run warmup + a time-bucketed step loop on a single env handle.

    Returns per-bucket (step_calls, wall_seconds). If a barrier is given,
    all worker threads wait on it before starting the timed loop so
    bucket boundaries align across processes.
    """
    duration = max(args.bucket, float(args.duration))
    bucket_size = max(1e-3, float(args.bucket))

    env.reset()

    for _ in range(args.warmup):
        actions = sample_actions(env, num_envs, rng)
        _, _, term, trunc, _ = env.step(actions)
        if num_envs == 1 and (term["p1"] or trunc["p1"]):
            env.reset()

    if barrier is not None:
        barrier.wait()

    bucket_calls: list[int] = []
    bucket_walls: list[float] = []
    loop_t0 = time.perf_counter()
    bucket_start = loop_t0
    calls_in_bucket = 0
    while True:
        actions = sample_actions(env, num_envs, rng)
        _, _, term, trunc, _ = env.step(actions)
        calls_in_bucket += 1
        if num_envs == 1 and (term["p1"] or trunc["p1"]):
            env.reset()

        now = time.perf_counter()
        if now - bucket_start >= bucket_size:
            bucket_calls.append(calls_in_bucket)
            bucket_walls.append(now - bucket_start)
            bucket_start = now
            calls_in_bucket = 0
        if now - loop_t0 >= duration:
            break
    return bucket_calls, bucket_walls


def bench_one(
    args: argparse.Namespace, num_envs: int, num_processes: int
) -> BenchResult:
    num_processes = max(1, int(num_processes))
    config = build_config(args, num_envs)

    envs = [
        footsiesgym.make(
            config=config,
            platform=args.platform,
            launch_binaries=args.launch_binaries,
        )
        for _ in range(num_processes)
    ]
    rngs = [
        np.random.default_rng(0xF005 + num_envs + 1000 * i)
        for i in range(num_processes)
    ]

    try:
        if num_processes == 1:
            results = [_step_loop_one(envs[0], num_envs, args, rngs[0], None)]
        else:
            barrier = threading.Barrier(num_processes)
            with ThreadPoolExecutor(max_workers=num_processes) as ex:
                futures = [
                    ex.submit(
                        _step_loop_one,
                        envs[i],
                        num_envs,
                        args,
                        rngs[i],
                        barrier,
                    )
                    for i in range(num_processes)
                ]
                results = [f.result() for f in futures]
    finally:
        for env in envs:
            try:
                env.close()
            except Exception:
                pass

    # Per-bucket aggregate: sum throughputs across processes within each
    # bucket. Truncate to the shortest worker if they disagree slightly.
    min_buckets = min(len(r[0]) for r in results)
    env_sps_per_bucket: list[float] = []
    call_sps_per_bucket: list[float] = []
    for b in range(min_buckets):
        env_sps = 0.0
        call_sps = 0.0
        for bucket_calls, bucket_walls in results:
            c, w = bucket_calls[b], bucket_walls[b]
            if w > 0:
                env_sps += (num_envs * c) / w
                call_sps += c / w
        env_sps_per_bucket.append(env_sps)
        call_sps_per_bucket.append(call_sps)

    total_calls = int(sum(sum(r[0]) for r in results))
    env_steps = num_envs * total_calls
    wall = max(float(sum(r[1])) for r in results)
    trials = min_buckets

    env_sps_mean = float(np.mean(env_sps_per_bucket)) if env_sps_per_bucket else 0.0
    call_sps_mean = float(np.mean(call_sps_per_bucket)) if call_sps_per_bucket else 0.0
    env_sps_ci = ci95_half_width(env_sps_per_bucket)
    call_sps_ci = ci95_half_width(call_sps_per_bucket)

    if num_envs == 1 and num_processes == 1:
        config_name = args.label
    elif num_processes == 1:
        config_name = f"{args.label}_N{num_envs}"
    else:
        config_name = f"{args.label}_N{num_envs}_P{num_processes}"
    return BenchResult(
        label=args.label,
        config=config_name,
        num_envs=num_envs,
        num_processes=num_processes,
        frame_skip=args.frame_skip,
        action_delay=args.action_delay,
        step_calls=total_calls,
        env_steps=env_steps,
        wall_seconds=wall,
        trials=trials,
        env_steps_per_sec=env_sps_mean,
        env_steps_per_sec_ci95=env_sps_ci,
        step_calls_per_sec=call_sps_mean,
        step_calls_per_sec_ci95=call_sps_ci,
    )


def append_result(result: BenchResult, csv_path: Path) -> None:
    """Append a row, dedup'd by (label, config, num_envs)."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict[str, str]] = []
    if csv_path.exists():
        with csv_path.open("r", newline="") as fh:
            existing = list(csv.DictReader(fh))

    key = (
        result.label,
        result.config,
        str(result.num_envs),
        str(result.num_processes),
    )
    existing = [
        row
        for row in existing
        if (
            row.get("label"),
            row.get("config"),
            row.get("num_envs"),
            row.get("num_processes"),
        )
        != key
    ]
    existing.append({k: str(v) for k, v in asdict(result).items()})

    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(existing)


def print_result(result: BenchResult) -> None:
    print(
        f"[{result.config}] N={result.num_envs} P={result.num_processes} "
        f"step_calls={result.step_calls} frame_skip={result.frame_skip} "
        f"trials={result.trials} wall={result.wall_seconds:.2f}s "
        f"env-steps/s={result.env_steps_per_sec:,.0f}"
        f"±{result.env_steps_per_sec_ci95:,.0f} "
        f"calls/s={result.step_calls_per_sec:,.1f}"
        f"±{result.step_calls_per_sec_ci95:,.1f}"
    )


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    n_values = parse_n_list(args.num_envs)
    p_values = parse_n_list(args.num_processes)

    mode = (
        "launching binaries"
        if args.launch_binaries
        else (f"connecting to {args.host}:{args.port}")
    )
    print(f"FootsiesEnv ({mode}); N sweep = {n_values}; " f"P sweep = {p_values}")
    for p in p_values:
        for n in n_values:
            result = bench_one(args, n, p)
            append_result(result, csv_path)
            print_result(result)


if __name__ == "__main__":
    main()
