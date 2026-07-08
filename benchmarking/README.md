# Benchmark scripts (§4 scaling study)

Throughput numbers and figure for §4 of the paper. We drive
`footsiesgym.make(config={"num_envs": N, ...})`, which internally selects
single-env or vectorized mode based on `num_envs`. The env handles
binary download/launch/teardown — no manual server invocations.

## Layout

```
scripts/
├── README.md
├── bench_throughput.py        # sweeps num_envs ∈ {1, 2, 4, ...}
└── plot_throughput.py         # reads CSV, writes benchmarking/figures/throughput_scaling_<platform>.pdf
```

CSV lives at `benchmarking/figures/data/throughput.csv` and is appended to on every run
(rows dedup'd by `(label, config, num_envs)`).

## Environment

Use the `footsies-paper` conda env (`footsies-gym` already installed):

```bash
conda activate footsies-paper
pip install matplotlib pandas   # if not already present
```

Run on Linux — `footsiesgym.make(..., launch_binaries=True)` is Linux-only,
which is what we want for the published throughput numbers anyway.

## Reproducing the §4 scaling study

```bash
# Single-env point (optional, anchors N=1 on the curve):
python scripts/bench_throughput.py --num-envs 1 --label headless_singleenv

# Sweep N for the scaling curve:
python scripts/bench_throughput.py \
    --num-envs 2,4,16,64,256,1024 \
    --label vectorized_headless \
    --step-calls 200

# Produce the figure:
python scripts/plot_throughput.py
```

Each invocation appends rows to `benchmarking/figures/data/throughput.csv`. Re-running
with the same `(label, num_envs)` overwrites that row.

## What the figure shows

- `benchmarking/figures/throughput_scaling_<platform>.pdf` — env-steps/sec vs N, log-log, with a
  linear-ideal dashed reference. The knee of the curve is the
  figure-worthy observation.

## Useful flags

- `--step-calls`  timed iterations per N value (env-steps = N × step_calls).
  Lower this for large N if a single run takes too long.
- `--frame-skip`  defaults to 4 to match the env wrapper. Set to 1 to
  report raw simulation steps/sec instead of env steps/sec.
- `--action-delay 0` (default) keeps the bench loop minimal. Action delay is
  interesting for §3.3 but doesn't affect raw throughput.
- `--platform linux` (default) — forwarded to `footsiesgym.make()`.
- `--no-launch-binaries --port P` — escape hatch for pointing at a manually
  launched server (e.g., for one-off profiling). Default path lets the env
  manage the binary lifecycle.

## Notes

- `footsiesgym.make()` falls back to the single-env `FootsiesGameService`
  whenever `num_envs == 1`, so there is no "vectorized N=1" row reachable
  through `make()`. The ablation skips it; the design contribution of the
  vectorized service is visible starting at N=2.
