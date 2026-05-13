# DSv4-Flash SM12x Batch-Push Milestone (v5)

Captured 2026-05-14 from a sequence of `tokenspeed-serve` configuration
sweeps on the SM120 workstation. The point of this note is to record that
the SM12x DSv4-Flash decode path scales cleanly past the bs=1 ceiling we
previously hit and to lock in the configuration that unlocks it.

## Result

| Configuration | bench c | output tok/s | success | TPOT (ms) | TTFT (ms) |
|---|---:|---:|---:|---:|---:|
| Pre-V2 baseline (V2 Stage 1, baseline config) | 1 | 22.04 | 2/2 | 43.03 | 2383 |
| v5 batch-push, c=4 | 4 | 40.93 | 8/8 | — | 11325 |
| **v5 batch-push, c=8 (steady)** | **8** | **75.56** | **8/8** | **—** | 10436 |
| v5 batch-push, c=8 + 16 queued prompts | 8 | 74.94 | 16/16 | — | 11867 |

Headline: **+243%** output throughput at c=8 vs c=1 baseline (3.4×).
All 16 prompts succeed in the queue test — no hangs, no SocketTimeoutError
spillover (the bench fix at commit `e376d01` is the safety net that
catches any stragglers).

## What was needed

The pre-V2 baseline used `--max-cudagraph-capture-size 2 --max-num-seqs 4
--max-total-tokens 4096 --gpu-memory-utilization 0.80`. That left only
~8 GB on each GPU for KV cache + CUDA graph private pool + activation
buffer once the 74 GiB DSv4-Flash weight slab had loaded. Pushing any
of (capture-size, num-seqs, total-tokens) higher OOM'd immediately.

The working v5 configuration:

```bash
tokenspeed-serve \
  --model deepseek-ai/DeepSeek-V4-Flash \
  --trust-remote-code --tokenizer-mode deepseek_v4 \
  --kv-cache-dtype fp8 \
  --attn-tp-size 2 --ep-size 2 --enable-expert-parallel \
  --moe-backend sm12x_mxfp4 \
  --max-total-tokens 16384 \
  --chunked-prefill-size 1024 \
  --max-cudagraph-capture-size 4 \
  --disable-cuda-graph-padding \
  --disable-prefill-graph \
  --gpu-memory-utilization 0.98 \
  --disable-kvstore \
  --max-num-seqs 8 \
  --host 127.0.0.1 --port 8000
```

Key knobs and rationale:

1. **`--gpu-memory-utilization 0.98`** (vs 0.80). vLLM on the same
   hardware safely uses 0.985; PyTorch's cap is a soft target. With 0.98
   we get ~91 GiB of usable budget. Weights take 74 GiB, leaving ~17 GiB
   for KV + graphs + activations — comfortably matches vLLM's measured
   layout (KV 7.4 GiB, CUDA graph pool 1.5 GiB + 2.1 GiB safety reserve,
   ~4 GiB activations, ~2 GiB NCCL/Triton/allocator overhead).
2. **`--max-total-tokens 16384`** (vs 4096). Decode 8 seqs × (1024
   input + 1024 output) needs 16k tokens of KV pool simultaneously.
   The default 4k could only fit 2 seqs at our prompt size, so requests
   queued and the per-stream `sock_read=120` from the bench fix
   eventually fired.
3. **`--max-num-seqs 8`** (vs 4). Caps the server's actual concurrency
   to 8 — matches what fits in the KV pool.
4. **`--max-cudagraph-capture-size 4`** (vs 2). Captures graphs for
   bs=1/2/3/4 so the dispatcher doesn't have to re-capture on shape
   changes mid-bench.
5. **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** in the bench
   script. Reduces fragmentation when the CUDA graph private pool grows
   during the first traffic wave.

## Memory layout under the working config

Measured snapshots (each GPU, TP=2):

| Stage | GPU mem used | Notes |
|---|---:|---|
| Pre-launch | 2 MiB | Clean |
| Weights loaded | 74.17 GiB | `Load weight end. avail mem=20.23 GB` |
| Endpoint up, idle | 86.18 GiB | + 12 GiB for KV pool + initial graph pool |
| After first traffic wave | 95.44 GiB | + 9 GiB CUDA graph private pool grew |
| Post-stop | 2 MiB | All released |

The 95.44 GiB peak corresponds to 99% of the 96 GiB physical capacity.
Higher util percentages would push into the driver / runtime reserve
and risk OOM at the moment the CUDA graph pool expands. 0.98 is the
ceiling we found stable.

## Why this matters for our optimisation track

- The V2 Stage 2 (pre-projection multi-stream fan-out) we parked
  earlier was neutral at bs=1 because the small GEMMs were
  bandwidth-bound, not compute-bound. At c=4-8 the GEMMs are
  4× to 8× bigger (per-iter rows go from 1 to 4-8), pushing them
  closer to compute-bound territory. Stage 2 may now be worth
  re-evaluating on the v5 config.
- The TTFT at c=4-8 is high (10-12 s) because prefill batches
  sequentially under the current scheduler — when 8 concurrent
  requests all want prefill, the 8th request waits for the prior 7.
  Upstream PR `lightseekorg/tokenspeed#122` (mixed prefill/decode
  batching) is what lands the scheduler-level fix; pick it up after
  the HTTP-server-stack reconciliation noted below.

## Known parking lot

* Cherry-picking upstream PR #122 onto this fork is non-trivial
  because the fork's rebase commit `fa8fd4a` already removed the
  legacy HTTP server stack (`python/tokenspeed/api_server.py`,
  `python/tokenspeed/runtime/entrypoints/http_server.py`) in favour of
  the gRPC `serve_smg` path, which needs `smg_grpc_servicer`. Future
  attempts should either install `smg_grpc_servicer` on the bench host
  or first cherry-pick a "restore HTTP server stack" patch from a
  pre-rebase commit, then layer PR #122 on top.
* The fork's previous v5 bench (used the workstation's stale legacy
  HTTP stack) is captured in
  `/tmp/tk-batch-push-v5/bench_*.log` on the bench workstation. The
  workstation Python env was over-merged during the PR #122 attempt
  and would need a clean re-install (or a fresh worktree) before the
  next bench round.

## Bench script

The v5 bench driver lives at `/tmp/tk_batch_push_v5.sh` on the bench
workstation; relevant invocation pattern is captured above. For an
independent re-run, the script body fits in ~80 lines and can be
re-created from the configuration table.
