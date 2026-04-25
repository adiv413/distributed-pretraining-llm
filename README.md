# Distributed Pretraining LLM: Parallelism Strategies Benchmark

This repository benchmarks various distributed training strategies (DDP, FSDP, tensor parallelism) for pretraining a small Llama3-style model (torchtitan flavor `1B` from the patch; scale is in the “~1B” class) on up to 8 GPUs (`single_gpu` and `tp_2` use 1 and 2 GPUs; see `scripts/run_all.py`).

## Experiment Setup

- **Hardware:** 8x A100 80GB SXM (NVLink), RunPod
- **Software:** The training stack is whatever `setup.sh` installs from the pinned [torchtitan](https://github.com/pytorch/torchtitan) commit; reported PyTorch / CUDA / torchtitan package versions (e.g. 2.11 / 12.8 / v0.2.2) describe the environment used for the published numbers, not a checked-in lockfile.
- **Model:** Pinned patch adds a `1B` `TransformerModelArgs` entry: dim=2048, 16 layers, 16 heads, GQA with 8 KV heads (`n_kv_heads=8`); see `patches/csv_logging_and_1b_flavor.patch`. Vocabulary size and other defaults follow the upstream torchtitan Llama3 model definition.
- **Training:** seq_len=2048, AdamW, global batch=16 (32,768 tokens/step, fixed) via `local_batch_size` × `seq_len` × data-parallel size; 550 steps, compile off (`[compile].enable = false` in TOML). Precision follows torchtitan defaults for that model (typically bf16; not set explicitly in our TOMLs).
- **Data:** `c4_test` in torchtitan (small set intended for quick runs / smoke tests; exact size is defined upstream).
- **Configs:** 7 TOML files under `configs/`. The parallel layout and `training.local_batch_size` are tuned so the global token count per step matches the single-GPU case; `job.description` also differs per file.
- **Metrics:** tokens/sec per GPU, peak memory, model FLOP utilization (MFU), scaling efficiency vs `single_gpu`. **Warmup:** `parse_results.py` keeps rows with `step >= 50` (steps **1–49** treated as warm-up; **501** rows from **50** through **550** when a full 550-step run finishes). MFU in `parse_results.py` uses **1e9** parameters and A100 BF16 spec FLOPS as rough constants, so treat MFU as an approximate, comparable score—not a match to a new exact param count.

## Repository layout and code

Training is not implemented in this repo from scratch. **This repo configures [torchtitan](https://github.com/pytorch/torchtitan), patches it for CSV logging and a small Llama3 flavor, runs several TOML jobs, and analyzes the logs.**

| Path                                      | What it does                                                                                                                                                                                                                                                                                                                        |
| ----------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `setup.sh`                                | Creates a Python 3.11 `venv/`, clones `torchtitan/` at a **pinned commit** (`TORCHTITAN_COMMIT` in the script), applies `patches/csv_logging_and_1b_flavor.patch`, installs torchtitan (editable) and this repo’s `requirements.txt`. Run once on a fresh machine.                                                                  |
| `patches/csv_logging_and_1b_flavor.patch` | Adds a **1B-scale** Llama3 `TransformerModelArgs` entry and **per-step CSV logging** in torchtitan’s trainer. When **`CSV_OUTPUT`** is set, rank 0 writes a header once, then one row per step: `step`, `step_time_s`, `tokens_this_step`, `peak_mem_gb`, `loss` (the `loss` column is left **empty** in this patch so parsing does not depend on it). |
| `configs/*.toml`                          | One file per strategy. **Parallelism** and **`training.local_batch_size`** differ (plus **`job.description`** strings); everything else in the shared template is aligned. `local_batch_size` is set so the **global token count per step** matches the 1-GPU baseline (32,768 tokens from batch × `seq_len` × data-parallel world).          |
| `scripts/run_all.py`                      | For each name, runs `torchrun` (rendezvous, `--local-ranks-filter 0`, `--tee 3`, etc.) for `-m torchtitan.train` with **`cwd` = `torchtitan/`**, sets **`CSV_OUTPUT=results/<name>.csv`** and **`PYTORCH_ALLOC_CONF=expandable_segments:True`**, and **redirects combined stdout+stderr** to **`results/<name>.log`** (log file only, not a shell “tee” to your terminal). Flags: `--only a,b`, `--steps N` for partial/smoke runs. |
| `scripts/parse_results.py`                | Reads `results/<config>.csv`, keeps rows with **`step` ≥ 50** (drops steps **1–49** as warm-up), writes **`results/summary.csv`**. **Scaling efficiency** is per-GPU throughput vs the `single_gpu` run. **MFU** uses fixed constants in the script; see Experiment Setup.                                                                          |
| `scripts/plot_results.py`                 | Reads `results/summary.csv` and writes **`figures/throughput.png`**, **`figures/memory.png`**, **`figures/scaling_efficiency.png`**, and **`figures/mfu.png`**.                                                                                                                                                                                 |
| `scripts/make_fake_data.py`               | Generates synthetic per-step CSVs under `results/` so you can run parse/plot without GPUs or torchtitan.                                                                                                                                                                                                                            |
| `requirements.txt` (repo root)            | Analysis dependencies only (`pandas`, `matplotlib`, `numpy`). Training stack comes from torchtitan after `setup.sh`.                                                                                                                                                                                                                |
| `test.py`                                 | Pins the torchtitan commit hash for reference; not an automated test runner.                                                                                                                                                                                                                                                        |

**Typical workflow**

1. `bash setup.sh` then `source venv/bin/activate`
2. `python scripts/run_all.py` (needs the right GPU count per config; use `--only` / `--steps` for partial runs)
3. `python scripts/parse_results.py`
4. `python scripts/plot_results.py`

The **`torchtitan/`** directory is created by `setup.sh` and is not part of this git tree; training always launches from there so imports and relative paths (e.g. tokenizer assets) match upstream torchtitan.

## Results Summary

| Config      | Tok/s/GPU  | Mem/GPU    | MFU       | Scaling Eff |
| :---------- | :--------- | :--------- | :-------- | :---------- |
| single_gpu  | 21,210     | 22.9 GB    | 40.8%     | 1.00        |
| **fsdp_8**  | **18,002** | **2.9 GB** | **34.6%** | **0.85**    |
| fsdp_4_tp_2 | 14,762     | 2.9 GB     | 28.4%     | 0.70        |
| ddp_8       | 14,610     | 28.6 GB    | 28.1%     | 0.69        |
| fsdp_2_tp_4 | 14,005     | 2.9 GB     | 26.9%     | 0.66        |
| tp_2        | 2,114      | 12.0 GB    | 4.1%      | 0.10        |
| tp_8        | 1,765      | 3.8 GB     | 3.4%      | 0.08        |

Numbers are rounded to match `results/summary.csv` produced from the checked-in per-step `results/<config>.csv` logs (re-run `parse_results.py` after new experiments).

---

## Key Findings

**1. FSDP beat DDP at the same GPU count.** This is surprising and warrants investigation. FSDP: 18,002 tok/s/GPU vs DDP: 14,610. FSDP uses 1/10th the memory (2.9 vs 28.6 GB) AND is faster. In theory DDP should be faster because it has less comm. My guess: FSDP's overlap of weight gather with compute is so effective that it actually hides better than DDP's end-of-backward gradient all-reduce. Also, DDP's full-replica gradients may be stressing memory bandwidth more.

**2. FSDP memory savings are massive.** 22.9 GB (single GPU) -> 2.9 GB (FSDP=8) is an 8x reduction. Exactly what the theory predicts - P/G/O state sharded across 8 GPUs.

**3. TP is catastrophically slow.** TP=2 gets 10% scaling efficiency; TP=8 gets 8%. The model is small (flavor `1B`), and with TP=8 the hidden size is split across 8 ranks (e.g. ~2048 → ~256 per rank for matrix work), so each per-rank GEMM is tiny compared to what tensor cores like. Collectives on the forward/backward path dominate; MFU in the 3–4% range is consistent with a communication-bound step.

**4. Combined FSDP+TP is worse than pure FSDP.** Both fsdp_4_tp_2 (0.70) and fsdp_2_tp_4 (0.66) underperform pure fsdp_8 (0.85). This confirms: TP is only worth adding when you need it (model too big, activation memory too large), not for speedup. In your 1B regime, pure FSDP is the winner.

**5. Best config is FSDP=8.** Lowest memory (2.9 GB), highest scaling (0.85), second highest throughput per GPU. At this scale, FSDP is the right tool.

---

## Metric Breakdown

### Why single_gpu has the highest MFU

Single GPU has 40.8% MFU - higher than any distributed config. This is because:

- No comm overhead at all
- Full batch (16 sequences) keeps matmuls big
- Tensor cores well-utilized

This is actually a common finding in systems papers: distributed training always has some efficiency loss compared to single-device. The question is how much. FSDP loses 15% (40.8 -> 34.6). TP loses 90%+.

### Memory

This is the clearest "wow, sharding works" metric:

- DDP=8 is actually worse than single-GPU (28.6 vs 22.9) - gradient buffers for all-reduce add overhead.
- Every FSDP-containing config collapses to about **2.9 GB** (roughly **8×** less than the ~23 GB single-GPU peak, matching shard count over 8 GPUs).
- TP=2 uses 12 GB (only half the sharding of TP=8's 3.8 GB because fewer GPUs).

### Throughput

Tells the speed story:

- Single GPU is fastest per-GPU (no comm overhead).
- FSDP=8 is second - comm overlap works beautifully.
- DDP and combined configs cluster at ~14-15K.
- TP is in the gutter (~2K).

### Scaling Efficiency

This is the money shot:

- FSDP=8 at 0.85 is excellent - near the 0.9 practitioners consider "good".
- DDP=8 at 0.69 is mediocre, exactly what small-batch scaling looks like.
- TP at 0.08-0.10 is catastrophic - the punchline.

---

## Deep Dive: The Surprise of FSDP Beating DDP

This is the most unexpected finding and worth investigating. In theory, DDP should be faster. Possible explanations:

1. **Memory bandwidth effects.** DDP holds 28.6 GB per GPU in duplicated state. Every matmul has to sweep through this memory. FSDP holds 2.9 GB "hot" (its shard) plus gathers layers on demand. The gathered weights are immediately consumed and thrown away, keeping cache pressure low.
2. **Gradient all-reduce timing.** DDP's single all-reduce at the end of backward is actually hard to overlap well, because all gradients need to be ready at once. FSDP's reduce-scatter happens layer-by-layer during backward, overlapping more naturally with the next layer's compute.
3. **NCCL algorithm differences.** DDP's single massive all-reduce may hit a less-optimized code path than FSDP's many smaller all-gathers and reduce-scatters.

Any of these could dominate. For a class project, "FSDP's fine-grained overlap pattern beats DDP's coarse single-reduce at small batch sizes" is a great takeaway to discuss.

---

## Deep Dive: The Mechanics of FSDP Memory Savings

Another interesting finding is the sheer magnitude of memory reduction when using FSDP, which drops the per-GPU memory from 22.9 GB down to 2.9 GB (an almost exact 8x reduction). This comes down to how different strategies handle the "Model State" (which includes the model's Parameters, computed Gradients, and Optimizer States):

1. **DDP Replication:** DDP duplicates the entire Model State across every GPU. If the total state requires 22.9 GB, every single GPU holds that full 22.9 GB footprint, resulting in massive redundancy.
2. **FSDP Sharding:** FSDP slices the Model State into equal pieces across the active GPUs. With 8 GPUs, the 22.9 GB state is divided by 8, resulting in ~2.86 GB per GPU (matching the observed 2.9 GB).

To compute the forward and backward passes while only holding a fraction of the model, FSDP performs "just-in-time" gathering. It fetches the required shards for a specific layer from the other GPUs, computes the layer, and then immediately discards the gathered weights. This mechanism trades a small amount of communication overhead for the massive memory savings shown in the benchmark.

---

## Deep Dive: Why Tensor Parallelism Was So Slow Here

TP shards **layers** across GPUs, so every block pays **many small collectives** on the critical path (harder to overlap than DDP/FSDP-style patterns). With dim 2048 and TP=8, per-rank GEMMs are **tiny** (~256-wide), so Tensor Cores stay underfed and the step goes **comm-bound**—low MFU matches “mostly waiting.” TP is mainly for **memory** when layers do not fit; here FSDP already suffices, so TP adds traffic without helping this small model. **Caveat:** wider models, bigger microbatches, or different stacks can amortize TP better; this is not “TP is always bad.”
