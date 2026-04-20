"""Generate fake per-step CSVs to test parse_results and plot_results."""
import csv
import random
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Made-up plausible throughput numbers (tokens/sec total across all GPUs)
FAKE = {
    "single_gpu":    (5000,  1, 18.0),
    "ddp_8":         (38000, 8, 22.0),
    "fsdp_8":        (33000, 8, 14.5),
    "tp_2":          (8800,  2, 12.0),
    "tp_8":          (26000, 8, 8.5),
    "fsdp_4_tp_2":   (30000, 8, 11.0),
    "fsdp_2_tp_4":   (27000, 8, 10.0),
}

for name, (total_tps, num_gpus, mem_gb) in FAKE.items():
    step_time = (num_gpus * 2048 * 16 / num_gpus) / total_tps  # rough
    tokens_per_step = total_tps * step_time
    rows = []
    for step in range(550):
        jitter = 1.0 + random.uniform(-0.05, 0.05)
        rows.append({
            "step": step,
            "step_time_s": round(step_time * jitter, 4),
            "tokens_this_step": int(tokens_per_step),
            "peak_mem_gb": round(mem_gb + random.uniform(-0.2, 0.2), 3),
            "loss": round(random.uniform(2.5, 4.0), 3),
        })
    path = RESULTS_DIR / f"{name}.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "step_time_s", "tokens_this_step", "peak_mem_gb", "loss"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")
