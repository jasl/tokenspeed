# SM12x Architecture Backlog (vLLM + ds4 inspired)

Captured 2026-05-13 from a side-by-side reading of:

- `~/Workspaces/vllm` branch `ds4-sm120-preview-dev` — production-shape SM12x
  DSv4 path on top of the upstream vLLM v1 engine.
- `~/Workspaces/ds4` — a ~ten-kfile single-binary DSv4 Flash decode server
  hand-tuned for GB10 / Mac (also SM12x).

After T1/T2/T3 + M-aware MoE + Path B SWA-only specialisation, every remaining
per-kernel lever has been mid-single-digit %. To reach the vLLM ~100 tok/s
baseline (we are at 20.29 tok/s on SM120 1024×1024 c1) we need an
architectural breakthrough, not another kernel tune. This document lists the
patterns worth borrowing, ranked by expected ROI.

DSv4-Flash layer mix (44 layers total): **SWA-only 3, HCA 20, CSA 21**. Any
fused / cached attention pattern saves ~41 launches per decode step.

## Source 1 — vLLM SM12x branch

### V1. BatchDescriptor-keyed cudagraph dispatcher (highest infra impact)

* `vllm/v1/cudagraph_dispatcher.py`, `vllm/forward_context.py:30`
* Pattern: graph cache keyed by `(batch_size, has_prefill, has_decode,
  capture_mode)`; **all graphs share one mempool**. Forward path picks the
  matching graph and replays.
* Expected gain: +15-30% e2e because we never recapture or fall back to eager
  on shape changes.
* Our state: single-shape capture in
  `python/tokenspeed/runtime/execution/model_executor.py`; shape change
  re-captures.
* Cost: ~600 LOC, medium-high risk (mempool sharing across replays needs
  careful event handling; tested in vLLM).
* Depends on: a clean per-step "begin/end commands" boundary (see D3 below).

### V2. aux-stream `execute_in_parallel` overlap (highest localised ROI)

* `vllm/utils/multi_stream_utils.py:61`,
  `vllm/model_executor/layers/deepseek_v4_attention.py:437`
* Pattern: env-gated `VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD` triggers wrapping
  Q/KV-proj + indexer-compressor in an aux CUDA stream with cross-stream
  events.
* Expected gain: +10-20% e2e. Directly targets the projection cluster — the
  same region T3-α nsys flagged as the largest dense block on the timeline.
* **Status (2026-05-14)**: Stage 1 shipped on `codex/ds4-sm12x-poc`; Stage 2
  parked on branch `v2-stage1-stage2-experimental` (commit `050b99f`).
  * **Stage 1** (post-projection only, single aux stream, ~40 LOC):
    `run_indexer || (insert_swa + compressor)` on HCA layers,
    `compressor || insert_swa` on CSA. SM120 1024×1024 c1 graph mode:
    21.70 → 22.07 tok/s (**+1.7%**, TPOT 43.80 → 43.03 ms).
  * **Stage 2** (pre-projection 3-way fan-out vLLM-style, ~250 LOC):
    `fused_wqa_wkv || compressor.kv_score || indexer.weights_proj ||
    indexer.compressor.kv_score`. Verified path engages
    (`multi_stream_enabled=True` across all 44 layers). Bench: +0.05 tok/s
    over Stage 1 — **bandwidth-bound at bs=1** (num_tokens=1 per attention
    layer makes every input GEMM a GEMV on shared HBM). Pick back up when
    decode batch grows past ~16 or when prefill chunk size dominates TPOT.

### V3. Per-type planner-state caching

* `vllm/v1/attention/backends/mla/sparse_swa.py:185-191`
* Pattern: `(kind, batch_size_bucket, total_seq_len_bucket) -> planner_state`
  cache; same-kind layers re-use the planner state.
* Expected gain: +5-10% e2e (cheap CPU win when 20+ HCA layers share state).
* Our state: per-layer planning.
* Cost: ~150 LOC, low risk (additive cache).

### V4. Async output-copy thread

* `vllm/v1/worker/gpu_worker.py` (separate worker thread copies logits/outputs
  D2H while the next forward runs).
* Expected gain: +3-8%; smooths TPOT by hiding D2H.
* Our state: synchronous output copy.
* Cost: ~250 LOC, low risk (clean thread boundary).

### V5. Centralised backend oracle

* `vllm/v1/attention/backends/oracle.py`-style registry.
* Pattern: one function picks backend (SWA/HCA/CSA/MLA) at startup; logged once.
* Expected gain: 0% perf, but removes scattered if/else and prevents
  miscalibration bugs (we have lived through several — see
  `feedback_verify_fast_path_engagement.md`).
* Cost: ~150 LOC, low risk (pure refactor).

## Source 2 — ds4 reference

### D1. bs=1 + top_k=6 super-specialised MoE kernels

* `ds4_cuda.cu:8488` `moe_down_sum6_qwarp32_kernel`
* `ds4_cuda.cu:7669` `moe_gate_up_mid_decode_lut_qwarp32_kernel`
* Pattern: kernels hard-coded for top_k=6 single-token decode; LUT for expert
  routing.
* Expected gain: +20-40% on the hottest decode shape, which is exactly
  ours at bs=1 top_k=6.
* Our state: M-aware warp kernel handles M<16 but is still generic.
* Cost: ~400 LOC, medium risk; needs a shape gate that falls back to the
  generic warp path for any other shape.

### D2. Single fused sparse-attention kernel across SWA + CSA + HCA

* `ds4_cuda.cu:3247` `attention_indexed_mixed_heads8_online_kernel`
* Pattern: one kernel handles all three mask kinds via constant-folded
  template + per-call mask selector; 8 heads/block.
* Expected gain: +10-15% (one launch per layer; no per-kind dispatch).
* Our state: per-kind launches (Path B SWA-only just landed; could be merged).
* Cost: ~500 LOC, medium risk (template explosion).

### D3. `begin_commands` / `end_commands` decode-step boundary

* `ds4_runtime.cc` exposes an explicit capture-boundary primitive at the
  decode-step level; commands are recorded, then replayed.
* Expected gain: structural — this is the kind of primitive that V1
  (BatchDescriptor graph dispatcher) needs to plug into cleanly.
* Cost: ~150 LOC, low risk (additive primitive).

## Priority queue

| Order | Lever | LOC | Risk | Gain | Why this order |
|---|---|---|---|---|---|
| 1 | V2 aux-stream | ~200 | L-M | 10-20% | smallest, highest-leverage, matches the known bottleneck |
| 2 | D1 bs=1+top_k=6 MoE | ~400 | M | 20-40% on decode | orthogonal to V2; same hot shape |
| 3 | V1 BatchDescriptor graph | ~600 | M-H | 15-30% | needs D3 first; biggest infra change |
| 3a | D3 command boundary | ~150 | L | structural | prerequisite for clean V1 |
| 4 | V3 planner cache | ~150 | L | 5-10% | cheap, independent |
| 5 | D2 fused SWA/CSA/HCA | ~500 | M | 10-15% | after V2 (so we can profile what is left) |
| 6 | V4 async output | ~250 | L | 3-8% | tail polish |
| 7 | V5 backend oracle | ~150 | L | 0% | refactor last |

Stacking 1+2+3 conservatively: ~1.3× to ~1.7× on top of current 20.29 tok/s,
which lands in the 27-35 tok/s band. To get to the ~100 tok/s vLLM target we
still need 5,6,7 plus a bs=1 ds4-style monolithic kernel rethink — that is
outside this backlog and will be its own track.

## Hygiene notes

* vLLM writes most kernels in Triton; we are CUDA-native by project policy
  (see `feedback_cuda_over_triton.md`). Do not "save LOC" by Tritonising —
  that policy is intentional.
* ds4 ships per-shape kernels (top_k=6, heads_per_block=8). For a serving
  target with bs > 1 these need gates, not as defaults.
* When implementing V1/V2, instrument with NVTX from the first commit
  (see `feedback_verify_fast_path_engagement.md`).
