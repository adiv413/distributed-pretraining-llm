# Distributed Pretraining LLM: Parallelism Strategies Benchmark

This repository benchmarks various distributed training strategies (DDP, FSDP, Tensor Parallelism) for pretraining a small LLM (Llama 3 1.4B) across an 8-GPU node.
Final Presentation Link: https://docs.google.com/presentation/d/1mGfGcpNcU-2aKmKW2dCXlYU1cD006ZgYewZtOqn120c/edit?usp=sharing

## Experiment Setup

- **Hardware:** 8x A100 80GB SXM (NVLink), RunPod
- **Software:** PyTorch 2.11 / CUDA 12.8 / torchtitan v0.2.2
- **Model:** Llama3 (1.43B params): dim=2048, 16 layers, 16 heads (GQA 8), vocab=128K
- **Training:** seq_len=2048, bf16, AdamW, global batch=16 (32K tokens/step, fixed), 550 steps, no compile
- **Data:** c4_test (2K samples) - systems bench only
- **Configs:** 7 parallelism strategies, identical except for [parallelism] section and local_batch_size (adjusted so global batch stays at 16)
- **Metrics:** Tokens/sec/GPU, peak memory, Model Flop Utilization (MFU), scaling efficiency. (Warmup: 50 steps discarded; measured over steps 50-550)

## Repository layout and code

Training is not implemented in this repo from scratch. **This repo configures [torchtitan](https://github.com/pytorch/torchtitan), patches it for CSV logging and a small Llama3 flavor, runs several TOML jobs, and analyzes the logs.**

| Path                                      | What it does                                                                                                                                                                                                                                                                                                                        |
| ----------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `setup.sh`                                | Creates a Python 3.11 `venv/`, clones `torchtitan/` at a **pinned commit** (`TORCHTITAN_COMMIT` in the script), applies `patches/csv_logging_and_1b_flavor.patch`, installs torchtitan (editable) and this repo’s `requirements.txt`. Run once on a fresh machine.                                                                  |
| `patches/csv_logging_and_1b_flavor.patch` | Adds a **1B-scale** Llama3 `TransformerModelArgs` entry and **per-step CSV logging** in torchtitan’s trainer. When the env var **`CSV_OUTPUT`** is set, rank 0 writes one row per step: `step`, `step_time_s`, `tokens_this_step`, `peak_mem_gb`, `loss`.                                                                           |
| `configs/*.toml`                          | One file per parallelism strategy. Sections match torchtitan’s job config: `[model]`, `[training]`, `[parallelism]`, etc. Only **`[parallelism]`** and **`training.local_batch_size`** differ between runs; local batch is chosen so **global batch stays 16** (32K tokens/step) across configs.                                    |
| `scripts/run_all.py`                      | For each config name, runs `torchrun --nproc_per_node=<gpus> -m torchtitan.train --job.config_file ...` with **`cwd` = `torchtitan/`**, sets **`CSV_OUTPUT=results/<name>.csv`**, and tees stdout/stderr to **`results/<name>.log`**. Flags: `--only a,b` to run a subset, `--steps N` to override training length for smoke tests. |
| `scripts/parse_results.py`                | Reads `results/<config>.csv`, drops the first **50** steps as warmup, aggregates into **`results/summary.csv`** (throughput, peak memory, MFU, **scaling efficiency vs `single_gpu`**). Expects pandas; MFU uses a fixed parameter count and A100 BF16 peak FLOPS in-script (approximate).                                          |
| `scripts/plot_results.py`                 | Reads `results/summary.csv` and writes **`figures/throughput.png`**, **`figures/memory.png`**, **`figures/scaling_efficiency.png`**.                                                                                                                                                                                                |
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

---

## Key Findings

**1. FSDP beat DDP at the same GPU count.** This is surprising and warrants investigation. FSDP: 18,002 tok/s/GPU vs DDP: 14,610. FSDP uses 1/10th the memory (2.9 vs 28.6 GB) AND is faster. In theory DDP should be faster because it has less comm. My guess: FSDP's overlap of weight gather with compute is so effective that it actually hides better than DDP's end-of-backward gradient all-reduce. Also, DDP's full-replica gradients may be stressing memory bandwidth more.

**2. FSDP memory savings are massive.** 22.9 GB (single GPU) -> 2.9 GB (FSDP=8) is an 8x reduction. Exactly what the theory predicts - P/G/O state sharded across 8 GPUs.

**3. TP is catastrophically slow.** TP=2 gets 10% scaling efficiency; TP=8 gets 8%. This is the single most interesting finding. What's happening: your model is tiny (1B) and your per-GPU batch was only 4 sequences. The matmuls being sharded across TP ranks become microscopic (dim=2048 -> dim=256 per GPU for TP=8), way below the size where tensor cores are efficient. Every matmul is sync-comm-bound with trivial compute. MFU of 3-4% means GPUs are idle 96% of the time waiting for comm.

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
- Every FSDP-containing config collapses to 2.9 GB (10x reduction).
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
