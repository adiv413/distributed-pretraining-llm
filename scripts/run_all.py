#!/usr/bin/env python3
"""
Launch all parallelism configs sequentially.
Each config writes its CSV to results/<config_name>.csv via the CSV_OUTPUT env var.
Runs independent of each other — if one fails, others still run.

Usage:
    python scripts/run_all.py
    python scripts/run_all.py --only fsdp_8,tp_8    # run subset
    python scripts/run_all.py --steps 100           # override training steps
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

# Map config name -> number of GPUs needed
CONFIGS = {
    "single_gpu":    1,
    "ddp_8":         8,
    "fsdp_8":        8,
    "tp_2":          2,
    "tp_8":          8,
    "fsdp_4_tp_2":   8,
    "fsdp_2_tp_4":   8,
}

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "configs"
RESULTS_DIR = REPO_ROOT / "results"
TORCHTITAN_DIR = REPO_ROOT / "torchtitan"


def run_config(name: str, nproc: int, steps_override: int | None) -> bool:
    config_path = CONFIG_DIR / f"{name}.toml"
    csv_path = RESULTS_DIR / f"{name}.csv"
    log_path = RESULTS_DIR / f"{name}.log"

    if not config_path.exists():
        print(f"[SKIP] {name}: config not found at {config_path}")
        return False

    env = os.environ.copy()
    env["CSV_OUTPUT"] = str(csv_path)
    env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    # Build torchrun command
    cmd = [
        "torchrun",
        f"--nproc_per_node={nproc}",
        "--rdzv_backend", "c10d",
        "--rdzv_endpoint", "localhost:0",
        "--local-ranks-filter", "0",
        "--role", "rank",
        "--tee", "3",
        "-m", "torchtitan.train",
        "--job.config_file", str(config_path),
    ]
    if steps_override is not None:
        cmd.extend(["--training.steps", str(steps_override)])

    print(f"\n{'=' * 60}")
    print(f"[RUN] {name} (nproc={nproc})")
    print(f"  config: {config_path}")
    print(f"  csv:    {csv_path}")
    print(f"  log:    {log_path}")
    print(f"{'=' * 60}")

    start = time.time()
    with open(log_path, "w") as logf:
        result = subprocess.run(
            cmd,
            cwd=str(TORCHTITAN_DIR),
            env=env,
            stdout=logf,
            stderr=subprocess.STDOUT,
        )
    elapsed = time.time() - start

    if result.returncode == 0:
        print(f"[OK]   {name} finished in {elapsed:.1f}s")
        return True
    else:
        print(f"[FAIL] {name} exited {result.returncode} after {elapsed:.1f}s")
        print(f"       see {log_path}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=str, default=None,
                        help="comma-separated config names to run (default: all)")
    parser.add_argument("--steps", type=int, default=None,
                        help="override training.steps (for quick smoke tests)")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)

    if args.only:
        requested = [x.strip() for x in args.only.split(",")]
        configs = {k: v for k, v in CONFIGS.items() if k in requested}
        missing = set(requested) - set(configs)
        if missing:
            print(f"[WARN] unknown configs ignored: {missing}")
    else:
        configs = CONFIGS

    print(f"Running {len(configs)} configs: {list(configs.keys())}")

    results = {}
    total_start = time.time()
    for name, nproc in configs.items():
        results[name] = run_config(name, nproc, args.steps)

    total_elapsed = time.time() - total_start

    # Summary
    print(f"\n{'=' * 60}")
    print(f"SUMMARY ({total_elapsed:.1f}s total)")
    print(f"{'=' * 60}")
    for name, ok in results.items():
        status = "OK" if ok else "FAIL"
        print(f"  [{status}] {name}")

    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()