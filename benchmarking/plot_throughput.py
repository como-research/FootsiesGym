"""Plot the §4 scaling figure from the throughput CSV.

Inputs:  benchmarking/figures/data/throughput.csv  (written by bench_throughput.py)
Outputs: benchmarking/figures/throughput_scaling_<platform>.pdf
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
FIG_DIR = BENCH_DIR / "figures"

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


if __name__ == "__main__":
    main()
