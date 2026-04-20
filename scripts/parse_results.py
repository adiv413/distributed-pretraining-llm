#!/usr/bin/env python3
"""
Parse per-config CSV logs (one row per training step) into a summary table.

Expected input: results/<config_name>.csv with columns:
    step, step_time_s, tokens_this_step, peak_mem_gb, loss

Output: results/summary.csv with one row per config, including derived metrics.
"""

import csv
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"

# Map config name -> number of GPUs
CONFIG_GPUS = {
    "single_gpu":    1,
    "ddp_8":         8,
    "fsdp_8":        8,
    "tp_2":          2,
    "tp_8":          8,
    "fsdp_4_tp_2":   8,
    "fsdp_2_tp_4":   8,
}

# Model constants — adjust if you change the 1B flavor
MODEL_PARAMS = 1_000_000_000  # approximate
PEAK_FLOPS_A100_BF16 = 312e12  # spec sheet, fp16/bf16 TFLOPS
WARMUP_STEPS = 50


def parse_one(name: str, num_gpus: int) -> dict | None:
    csv_path = RESULTS_DIR / f"{name}.csv"
    if not csv_path.exists():
        print(f"[SKIP] {name}: {csv_path} not found")
        return None

    df = pd.read_csv(csv_path)
    if len(df) <= WARMUP_STEPS:
        print(f"[WARN] {name}: only {len(df)} steps, fewer than warmup={WARMUP_STEPS}")
        return None

    # Discard warmup
    df = df[df["step"] >= WARMUP_STEPS].copy()

    total_time = df["step_time_s"].sum()
    total_tokens = df["tokens_this_step"].sum()
    tokens_per_sec = total_tokens / total_time

    peak_mem = df["peak_mem_gb"].max()
    mean_step_time = df["step_time_s"].mean()

    # MFU calculation
    flops_per_sec = 6 * MODEL_PARAMS * tokens_per_sec
    mfu = flops_per_sec / (num_gpus * PEAK_FLOPS_A100_BF16)

    # Tokens/sec per GPU (for scaling efficiency)
    tokens_per_sec_per_gpu = tokens_per_sec / num_gpus

    return {
        "config": name,
        "num_gpus": num_gpus,
        "mean_step_time_s": mean_step_time,
        "tokens_per_sec": tokens_per_sec,
        "tokens_per_sec_per_gpu": tokens_per_sec_per_gpu,
        "peak_mem_gb": peak_mem,
        "mfu_pct": mfu * 100,
        "num_steps_measured": len(df),
    }


def main():
    rows = []
    for name, num_gpus in CONFIG_GPUS.items():
        row = parse_one(name, num_gpus)
        if row is not None:
            rows.append(row)

    if not rows:
        print("No results parsed. Did any configs complete?")
        return

    df = pd.DataFrame(rows)

    # Compute scaling efficiency vs single_gpu baseline
    if "single_gpu" in df["config"].values:
        baseline_tps_per_gpu = df[df["config"] == "single_gpu"]["tokens_per_sec_per_gpu"].iloc[0]
        df["scaling_efficiency"] = df["tokens_per_sec_per_gpu"] / baseline_tps_per_gpu
    else:
        df["scaling_efficiency"] = float("nan")
        print("[WARN] no single_gpu baseline; scaling_efficiency will be NaN")

    # Round for readability
    df_display = df.copy()
    df_display["tokens_per_sec"] = df_display["tokens_per_sec"].round(0).astype(int)
    df_display["tokens_per_sec_per_gpu"] = df_display["tokens_per_sec_per_gpu"].round(0).astype(int)
    df_display["peak_mem_gb"] = df_display["peak_mem_gb"].round(2)
    df_display["mfu_pct"] = df_display["mfu_pct"].round(1)
    df_display["scaling_efficiency"] = df_display["scaling_efficiency"].round(3)
    df_display["mean_step_time_s"] = df_display["mean_step_time_s"].round(3)

    summary_path = RESULTS_DIR / "summary.csv"
    df_display.to_csv(summary_path, index=False)
    print(f"\nWrote {summary_path}\n")
    print(df_display.to_string(index=False))


if __name__ == "__main__":
    main()