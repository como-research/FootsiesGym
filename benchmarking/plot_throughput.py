"""Plot the throughput figures from the benchmark CSVs.

Inputs:  benchmarking/figures/data/throughput.csv    (bench_throughput.py, Unity)
         benchmarking/figures/data/throughput_c.csv  (bench_c_throughput.py, C/CUDA)
Outputs: benchmarking/figures/throughput_scaling_<platform>.pdf  (Unity, §4)
         benchmarking/figures/throughput_c.pdf and assets/throughput_c.png
"""

from __future__ import annotations

import argparse
import platform
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

BENCH_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = BENCH_DIR / "figures" / "data" / "throughput.csv"
DEFAULT_C_CSV = BENCH_DIR / "figures" / "data" / "throughput_c.csv"
FIG_DIR = BENCH_DIR / "figures"
ASSETS_DIR = BENCH_DIR.parent / "assets"

# The C figure's lines: (CSV label, legend text, color). Categorical slots
# 1-3 of the dataviz reference palette, validated all-pairs on white.
C_SERIES = [
    ("c_ffi_gpu", "GPU, JAX (jitted)", "#2a78d6"),
    ("c_make_cuda_torch", "GPU, PyTorch", "#eb6834"),
    ("c_make_cpu_numpy_T24", "CPU, 24 threads", "#1baf7a"),
]
UNITY_REFERENCE = ("bench", 4, "#8a8984")  # label, num_processes, neutral gray

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    }
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", default=str(DEFAULT_CSV))
    p.add_argument("--c-csv", default=str(DEFAULT_C_CSV))
    p.add_argument("--out-dir", default=str(FIG_DIR))
    p.add_argument(
        "--label",
        default="bench",
        help="Label whose rows define the scaling curve (matches --label in bench_throughput.py).",
    )
    p.add_argument(
        "--platform",
        default="mac" if platform.system() == "Darwin" else "linux",
        choices=["linux", "mac"],
        help="Suffix for the output filename (defaults to the current OS).",
    )
    return p.parse_args()


def load(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}\nRun bench_throughput.py first.")
    df = pd.read_csv(csv_path)
    df["num_envs"] = df["num_envs"].astype(int)
    df["env_steps_per_sec"] = df["env_steps_per_sec"].astype(float)
    if "env_steps_per_sec_ci95" in df.columns:
        df["env_steps_per_sec_ci95"] = pd.to_numeric(
            df["env_steps_per_sec_ci95"], errors="coerce"
        )
    else:
        df["env_steps_per_sec_ci95"] = float("nan")
    if "num_processes" in df.columns:
        df["num_processes"] = (
            pd.to_numeric(df["num_processes"], errors="coerce").fillna(1).astype(int)
        )
    else:
        df["num_processes"] = 1
    return df


def plot_scaling(
    df: pd.DataFrame,
    out_path: Path,
    label: str,
) -> None:
    df = df[df["label"] == label].copy()
    if df.empty:
        print(f"[scaling] no rows with label {label!r}")
        return

    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    p_values = sorted(df["num_processes"].unique())
    cmap = plt.colormaps.get_cmap("viridis")
    color_anchors = (
        np.linspace(0.15, 0.85, len(p_values)) if len(p_values) > 1 else np.array([0.5])
    )
    colors = [cmap(a) for a in color_anchors]

    for color, p in zip(colors, p_values):
        rows = df[df["num_processes"] == p].sort_values("num_envs").reset_index()
        x = rows["num_envs"].to_numpy()
        y = rows["env_steps_per_sec"].to_numpy()
        ci = rows["env_steps_per_sec_ci95"].fillna(0).to_numpy()
        ax.fill_between(x, y - ci, y + ci, color=color, alpha=0.2, linewidth=0)
        ax.plot(
            x,
            y,
            color=color,
            linewidth=2.0,
            marker="o",
            markersize=4,
            markerfacecolor=color,
            markeredgecolor=color,
            label=f"$P={p}$",
        )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ns = sorted(df["num_envs"].unique())
    ax.set_xticks(ns)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, *_: f"{int(x)}"))
    ax.xaxis.set_minor_formatter(FuncFormatter(lambda *_: ""))

    y_ticks = [500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]
    ax.set_yticks(y_ticks)
    ax.set_yticks([], minor=True)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, *_: f"{int(y):,}"))

    ax.set_xlabel("Parallel Environments")
    ax.set_ylabel("Steps per Second (SPS)")
    ax.grid(True, which="major", alpha=0.3, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="in", length=4)
    if len(p_values) > 1:
        ax.legend(
            loc="upper left",
            frameon=False,
            fontsize=10,
            handlelength=1.6,
        )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    print(f"[scaling] wrote {out_path}")


def _series(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Rows of ``label`` (the bench label is a prefix of the per-N config)."""
    rows = df[df["config"].str.startswith(label + "_N") | (df["label"] == label)]
    return rows.drop_duplicates("num_envs", keep="last").sort_values("num_envs")


def _si(value: float) -> str:
    """1e6 -> "1M" (decimal: steps per second)."""
    for scale, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if value >= scale:
            return f"{value / scale:g}{suffix}"
    return f"{value:g}"


def _binary(value: float) -> str:
    """4096 -> "4K", 2**20 -> "1M" (env counts are powers of two)."""
    for scale, suffix in ((2**20, "M"), (2**10, "K")):
        if value >= scale:
            return f"{value / scale:g}{suffix}"
    return f"{int(value)}"


def plot_c(df_c: pd.DataFrame, df_unity: pd.DataFrame, out_paths: list[Path]) -> None:
    """C/CUDA backend throughput against num_envs, with Unity as a reference."""
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ends = []
    for label, name, color in C_SERIES:
        rows = _series(df_c, label)
        if rows.empty:
            print(f"[c] no rows for {label!r}")
            continue
        x = rows["num_envs"].to_numpy()
        y = rows["env_steps_per_sec"].to_numpy()
        ci = rows["env_steps_per_sec_ci95"].fillna(0).to_numpy()
        ax.fill_between(x, y - ci, y + ci, color=color, alpha=0.2, linewidth=0)
        ax.plot(x, y, color=color, linewidth=2.0, marker="o", markersize=4, label=name)
        ends.append((x[-1], y[-1], name))

    label, procs, gray = UNITY_REFERENCE
    unity = df_unity[
        (df_unity["label"] == label) & (df_unity["num_processes"] == procs)
    ]
    unity = unity.sort_values("num_envs")
    if not unity.empty:
        ax.plot(
            unity["num_envs"],
            unity["env_steps_per_sec"],
            color=gray,
            linewidth=1.5,
            linestyle="--",
            label=f"Unity server ($P={procs}$)",
        )
        ends.append(
            (
                unity["num_envs"].iloc[-1],
                unity["env_steps_per_sec"].iloc[-1],
                f"Unity ($P={procs}$)",
            )
        )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    x_ticks = [1, 16, 256, 4096, 65536, 1048576]
    ax.set_xticks(x_ticks)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, *_: _binary(x)))
    ax.xaxis.set_minor_formatter(FuncFormatter(lambda *_: ""))
    y_ticks = [1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9]
    ax.set_yticks(y_ticks)
    ax.set_yticks([], minor=True)
    ax.set_ylim(5e2, 1e9)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, *_: _si(y)))

    # Direct labels at each line's end, nudged apart where lines end close
    # together (identity never by color alone).
    gap = 0.4  # decades between label centers that share a column
    placed: list[tuple[float, float]] = []
    for x, y, name in sorted(ends, key=lambda e: -e[1]):
        ly = y
        for px, py in placed:
            if abs(np.log2(px) - np.log2(x)) < 6 and abs(np.log10(ly / py)) < gap:
                ly = py / 10**gap
        placed.append((x, ly))
        ax.annotate(
            name,
            (x, y),
            xytext=(x * 1.5, ly),
            textcoords="data",
            fontsize=8.5,
            color="#0b0b0b",
            va="center",
        )
    ax.set_xlim(0.6, 2**20 * 2**8)

    ax.set_xlabel("Parallel Environments")
    ax.set_ylabel("Steps per Second (SPS)")
    ax.grid(True, which="major", alpha=0.3, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="in", length=4)
    ax.legend(loc="upper left", frameon=False, fontsize=9, handlelength=1.6)
    fig.tight_layout()
    for out_path in out_paths:
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05, dpi=200)
        print(f"[c] wrote {out_path}")


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load(csv_path)
    plot_scaling(
        df,
        out_dir / f"throughput_scaling_{args.platform}.pdf",
        args.label,
    )
    if Path(args.c_csv).exists():
        plot_c(
            load(Path(args.c_csv)),
            df,
            [out_dir / "throughput_c.pdf", ASSETS_DIR / "throughput_c.png"],
        )


if __name__ == "__main__":
    main()
