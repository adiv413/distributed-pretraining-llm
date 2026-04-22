#!/usr/bin/env python3
"""
Produce the 4 key figures from results/summary.csv:
    figures/throughput.png          - tokens/sec/GPU per config
    figures/memory.png              - peak memory per config
    figures/scaling_efficiency.png  - scaling eff vs single-GPU baseline
    figures/mfu.png                 - Model FLOP Utilization per config
"""

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = REPO_ROOT / "figures"
SUMMARY_PATH = REPO_ROOT / "results" / "summary.csv"

# Consistent ordering for plots
CONFIG_ORDER = [
    "single_gpu",
    "ddp_8",
    "fsdp_8",
    "tp_2",
    "tp_8",
    "fsdp_4_tp_2",
    "fsdp_2_tp_4",
]

# Prettier labels
LABELS = {
    "single_gpu":    "1 GPU",
    "ddp_8":         "DDP=8",
    "fsdp_8":        "FSDP=8",
    "tp_2":          "TP=2",
    "tp_8":          "TP=8",
    "fsdp_4_tp_2":   "FSDP=4 × TP=2",
    "fsdp_2_tp_4":   "FSDP=2 × TP=4",
}


def load():
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(f"{SUMMARY_PATH} missing. Run parse_results.py first.")
    df = pd.read_csv(SUMMARY_PATH)
    # Order and label
    df["order"] = df["config"].map({name: i for i, name in enumerate(CONFIG_ORDER)})
    df = df.sort_values("order").reset_index(drop=True)
    df["label"] = df["config"].map(LABELS).fillna(df["config"])
    return df


def plot_throughput(df):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(df["label"], df["tokens_per_sec_per_gpu"], color="steelblue")
    ax.set_ylabel("Tokens / sec / GPU")
    ax.set_title("Per-GPU throughput by parallelism strategy")
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["label"], rotation=30, ha="right")
    for i, v in enumerate(df["tokens_per_sec_per_gpu"]):
        ax.text(i, v, f"{int(v):,}", ha="center", va="bottom", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "throughput.png", dpi=150)
    plt.close()
    print(f"Wrote {FIGURES_DIR / 'throughput.png'}")


def plot_memory(df):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(df["label"], df["peak_mem_gb"], color="darkorange")
    ax.set_ylabel("Peak GPU memory (GB)")
    ax.set_title("Peak memory per GPU by parallelism strategy")
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["label"], rotation=30, ha="right")
    for i, v in enumerate(df["peak_mem_gb"]):
        ax.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "memory.png", dpi=150)
    plt.close()
    print(f"Wrote {FIGURES_DIR / 'memory.png'}")


def plot_scaling(df):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(df["label"], df["scaling_efficiency"], color="seagreen")
    ax.axhline(1.0, color="black", linestyle="--", alpha=0.5, label="perfect scaling")
    ax.set_ylabel("Scaling efficiency")
    ax.set_title("Scaling efficiency vs single-GPU baseline")
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["label"], rotation=30, ha="right")
    for i, v in enumerate(df["scaling_efficiency"]):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, max(1.1, df["scaling_efficiency"].max() * 1.15))
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "scaling_efficiency.png", dpi=150)
    plt.close()
    print(f"Wrote {FIGURES_DIR / 'scaling_efficiency.png'}")


def plot_mfu(df):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(df["label"], df["mfu_pct"], color="mediumpurple")
    ax.set_ylabel("MFU (%)")
    ax.set_title("Model FLOP Utilization by parallelism strategy")
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["label"], rotation=30, ha="right")
    for i, v in enumerate(df["mfu_pct"]):
        ax.text(i, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, max(50, df["mfu_pct"].max() * 1.15))
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "mfu.png", dpi=150)
    plt.close()
    print(f"Wrote {FIGURES_DIR / 'mfu.png'}")


def main():
    FIGURES_DIR.mkdir(exist_ok=True)
    df = load()
    plot_throughput(df)
    plot_memory(df)
    plot_scaling(df)
    plot_mfu(df)


if __name__ == "__main__":
    main()