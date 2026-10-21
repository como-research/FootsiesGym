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

## C/CUDA backend

`bench_c_throughput.py` measures `footsiesgym.c` with random actions and
finished games reset inside `step`. Rows go to the same CSV with the label
`c_<backend>`:

- `--backend make`: `footsiesgym.make(backend="c", ...)` as users run it
  (`--device`, `--tensor-type`, `--threads`): new output arrays every step.
- `--backend cpu`: `FootsiesCEnv` (numpy, OpenMP; `--threads`, `--impl`).
- `--backend ffi`: `FootsiesFFIEnv` inside a jitted `lax.scan`, the way a
  JAX training loop drives it (needs JAX).
- `--backend cuda`: the CUDA kernel alone, through its C API.

```bash
python benchmarking/bench_c_throughput.py --backend make --device cuda --tensor-type torch --num-envs 16384,262144
python benchmarking/bench_c_throughput.py --backend cpu --threads 24 --num-envs 16384,65536
python benchmarking/bench_c_throughput.py --backend ffi --num-envs 4096,65536,262144
python benchmarking/bench_c_throughput.py --backend cuda --num-envs 65536,1048576
```

Measured with frame_skip=4, action_delay=0, on an RTX 3090 and a Threadripper
3960X (24 cores, AVX2, so `impl="scalar"`) (env-steps per second):

| N | CUDA kernel | FFI (GPU) | CPU, 1 thread | CPU, 24 threads |
|---|---|---|---|---|
| 4,096 | 79M | 60M | | |
| 16,384 | 107M | 98M | 2.2M | 40M |
| 65,536 | 136M | 128M | 2.3M | 43M |
| 262,144 | 154M | 149M | | |
| 1,048,576 | 165M | 156M | | |

Through `make()` (new arrays every step, waiting for each step's outputs):
`device="cuda"`, torch 66M at 16,384 envs and 114M at 262,144; JAX 34M and
105M; `device="cpu"` with 24 threads, about 33M. A jitted JAX loop over
`make(functional=True)` runs at the FFI column's rates.

For comparison, the best Unity-server row is about 48k env-steps/s (N=128, P=4).

### Figure

`assets/throughput_c.png` (in the main README) and `figures/throughput_c.pdf`
come from `figures/data/throughput_c.csv`. To make the figure again, run the
three benchmarks to that CSV, then the plot script:

```bash
CSV=benchmarking/figures/data/throughput_c.csv
N=1,16,256,1024,4096,16384,65536,262144,1048576
python benchmarking/bench_c_throughput.py --backend ffi --num-envs $N --duration 5 --csv $CSV
python benchmarking/bench_c_throughput.py --backend make --device cuda --tensor-type torch --num-envs $N --duration 5 --csv $CSV
python benchmarking/bench_c_throughput.py --backend make --device cpu --threads 24 --num-envs 1,16,256,1024,4096,16384,65536,262144 --duration 5 --csv $CSV
python benchmarking/plot_throughput.py
```

The figure shows the Unity server with P=4 (from `figures/data/throughput.csv`)
as a reference. Close other programs that use the GPU before you run the GPU
benchmarks.
