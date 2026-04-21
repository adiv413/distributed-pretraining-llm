# distributed-pretraining-llm

# Distributed Pretraining LLM: Parallelism Strategies Benchmark

This repository benchmarks various distributed training strategies (DDP, FSDP, Tensor Parallelism) for pretraining a small LLM (Llama 3 1.4B) across an 8-GPU node.

## Experiment Setup

- **Hardware:** 8x A100 80GB SXM (NVLink), RunPod
- **Software:** PyTorch 2.11 / CUDA 12.8 / torchtitan v0.2.2
- **Model:** Llama3 (1.43B params): dim=2048, 16 layers, 16 heads (GQA 8), vocab=128K
- **Training:** seq_len=2048, bf16, AdamW, global batch=16 (32K tokens/step, fixed), 550 steps, no compile
- **Data:** c4_test (2K samples) - systems bench only
- **Configs:** 7 parallelism strategies, identical except for [parallelism] section and local_batch_size (adjusted so global batch stays at 16)
- **Metrics:** Tokens/sec/GPU, peak memory, Model Flop Utilization (MFU), scaling efficiency. (Warmup: 50 steps discarded; measured over steps 50-550)

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
