# DeepSeek V4 SM12x Experiment Archive (T1 / T2 / T3)

This note distills the SM12x DSv4-Flash decode-throughput work from
2026-05-09 to 2026-05-13 into one place. The rolling plan still lives in
`docs/plans/2026-05-09-ds4-sm12x-native-route.md`; this archive captures
hypothesis, evidence, and verdict for each experiment so they don't
have to be re-discovered next time someone wonders why we kept (or
rejected) a particular code path.

## Result Summary

SM120 `1024x1024 c1` graph-mode bench, 2 prompts, `--max-cudagraph-capture-size 2`:

| Checkpoint | tok/s | TPOT | TTFT | Δ tok/s |
|---|---|---|---|---|
| Pre-T1-α (reference-heavy attn + miscalibrated gates) | 14.54 | -- | -- | baseline |
| Post-T1-α + gate fixes (warp baseline, k=1) | 17.13 | 52.84 ms | -- | +17.8% |
| T2-α-1 K-split default = 8 | 19.66 | 45.35 ms | -- | +35.2% |
| M-aware MoE auto-dispatch | 20.14 | 45.31 ms | 4477 ms | +38.5% |
| Path B SWA-only specialised kernel | **20.29** | **45.27 ms** | **4166 ms** | **+39.5%** |

Total improvement vs the pre-T1-α reference path: **+39.5% tok/s** on the
same SM120 hardware, plus a meaningful first-token latency win that
landed in the M-aware MoE + Path B phase. vLLM ~100 tok/s baseline
remains the longer-term target.

## Live Knobs (ship surface)

* `TOKENSPEED_SM12X_MXFP4_MOE_IMPL` — default `auto` (M-aware per-call
  dispatch). Explicit modes: `warp`, `scalar` (deprecated, see Dead
  Code), `persistent`. `=warp` is the kill switch back to legacy
  global-warp behaviour.
* `TOKENSPEED_SM12X_MXFP4_MOE_M_THRESHOLD` — auto-dispatch threshold;
  default `16` (Blackwell MMA tile width). `M = num_tokens * top_k`.
* `TOKENSPEED_DEEPSEEK_V4_CUDA_SPARSE_MLA_CACHE_K_SPLIT` — sparse-MLA
  K-axis split count; default `8`, `=1` is the legacy single-block
  kill switch.
* `TOKENSPEED_DEEPSEEK_V4_CUDA_SPARSE_MLA_CACHE_K_SPLIT_{SWA,HCA,CSA}`
  — per-kind overrides; defaults all `8` after the T2-α-2 negative
  sweep (kept as benchmarking levers).
* `TOKENSPEED_NVTX` (or `--enable-nvtx`) — arm the NVTX ranges used by
  the `nsys --capture-range=nvtx` workflow.

## T1-α: Graph Capture + Gate Fix (LANDED)

* **Hypothesis**: eager-mode decode is launch-overhead-bound; CUDA
  graphs should elide most per-layer launch cost.
* **Pre-condition fixes**:
  * `read_deepseek_v4_indexer_fp8_cache` was doing `.tolist()` (CPU↔CUDA
    sync); rewrote as a vectorized `torch.gather`.
  * `_prefill_workspace` was reading tensor values during capture; uses
    config-derived upper bounds when `torch.cuda.is_current_stream_capturing()`.
  * The sparse-MLA fast-path's default `max_topk=1024` gate was *tighter*
    than the DSv4-Flash CSA topk_tokens (8192), making every decode
    silently fall through to the BF16 reference path. Removed the
    default; env knob retained as a kill switch.
* **Result**: warp baseline 14.54 → 17.13 tok/s (+17.8%) at the same
  shape. Verified via `--max-cudagraph-capture-size 2
  --disable-cuda-graph-padding --disable-prefill-graph`.
* **Status**: LANDED. The graph-capture path is the default.

## T2-α-1: Sparse-MLA K-axis Split (LANDED)

* **Hypothesis**: the sparse-MLA fp8-cache decode kernel runs one block
  per (token, head). At `bs=1 * 64` heads that's only 64 blocks on 140
  SMs — parallelism-starved at the worst possible decode shape.
* **Implementation**: split the candidate axis into `k_split` chunks
  per (token, head); each chunk writes a `(max, denom, acc[head_dim])`
  partial to scratch and a small reduce kernel merges them with the
  attention sink as a phantom partial.
* **Microbench** at the DSv4-Flash CSA decode shape (head_dim=512, 64
  heads, swa=128, extra=8192): `~3.7x` per-call speedup at `k=4`,
  saturating around `k=8`.
* **E2E bench** at `1024x1024 c1`:

  | k_split | tok/s | TPOT |
  |---|---|---|
  | 1 (kill switch) | 17.13 | 52.84 ms |
  | 4 | 19.26 | 46.40 ms |
  | **8** | **19.66** | **45.35 ms** |
  | 16 | 19.67 | 45.30 ms |

* **Greedy oracle** (1024-token prompt, temperature=0): first ~8
  tokens bit-identical between `k=1` and `k=8`, then expected fp
  accumulation-order drift. Within `atol=5e-3` per-call, unit-tested.
* **Status**: LANDED, default `k_split=8`. Net win at the new default
  vs the kill switch: **+14.8% tok/s / -14.2% TPOT**.

## T2-α-2: Per-kind K-split Tuning (REJECTED, infrastructure kept)

* **Hypothesis**: short-`total_len` layers (`swa` ~128, `hca` ~136 at
  1024 ctx) over-split at the global `k=8` and would benefit from
  smaller per-kind chunk counts.
* **Sweep**:

  | Variant | tok/s | TPOT |
  |---|---|---|
  | global_k8 (all kinds k=8) | **19.69** | **45.25 ms** |
  | per_kind_default (swa=1,hca=2,csa=8) | 19.31 | 46.23 ms |
  | hca_k1 (swa=1,hca=1,csa=8) | 19.01 | 47.05 ms |
  | hca_k4 (swa=1,hca=4,csa=8) | 19.50 | 45.74 ms |

* **Root cause**: at bs=1 * 64 heads, the kernel is parallelism-bound
  on the 140-SM SM120. `k=8` gives 512 blocks (`~3.66` blocks/SM,
  near-perfect occupancy); `k=2` only 128 blocks (`~46%` utilisation).
  The extra block-level parallelism wins even when each chunk only
  walks ~17 candidates. Per-block arithmetic intensity does not
  dominate at this shape.
* **Outcome**: per-kind defaults all set back to `k=8` so end-to-end
  matches the T2-α-1 ship default. Per-kind dispatch + env knobs
  (`..._K_SPLIT_SWA/HCA/CSA`) kept as benchmarking levers; commit
  `3ce60ee`.
* **Lesson**: when grid is small relative to SM count, always check
  parallelism first (`grid_blocks / num_SMs`) before reasoning about
  per-block arithmetic intensity. Same lesson as
  `feedback_blackwell_parallelism.md`.

## T3-α: Persistent MoE Regression Root-Caused (NVTX nsys)

* **Background**: T3-α tensorcore MoE forward had a microbench-vs-runtime
  inversion — `1.20-1.27x` faster than warp on isolated calls but a
  `~6%` end-to-end regression on the bench. Several earlier hypotheses
  (graph-pool fragmentation via A1 scratch cache, scheduling glitches)
  were tested and ruled out.
* **Instrumentation**: `_run_target_forward` got a nested
  `nvtx_range("decode_step")` (zero-overhead when
  `--enable-nvtx`/`TOKENSPEED_NVTX=1` is off). Capture wrapper at
  `/tmp/tk-serve/nsys_decode_step.sh` (workstation) uses
  `nsys profile --capture-range=nvtx
  --nvtx-capture='decode_step@tokenspeed'
  --capture-range-end=repeat-shutdown:N` to skip the slow
  prefill/warmup/capture phases.
* **NVTX summary** (warp vs persistent, 5 decode_step ranges, 5 child
  traces):

  | Range | warp ns/step | persistent ns/step | Δ |
  |---|---|---|---|
  | `decode_step` | 2247 ms | 2307 ms | +2.6% |
  | `ffn_total` | 186 ms | 231 ms | +24% |
  | `moe_sm12x_experts` | 51 ms | 95 ms | **+85%** |
  | `attn_total` | 1337 ms | 1351 ms | +1% |
  | `NCCL:ncclAllReduce` | 15.0 ms | 11.3 ms | -25% |

* **Kernel-level inside `moe_sm12x_experts`** (same 344 layer-calls
  on both paths):

  | Path | Kernels | GPU time |
  |---|---|---|
  | warp | gate_activation_warp_ds4_decode + down_warp_ds4_decode | **55.3 ms** |
  | persistent | mxfp4_moe_w13_tensorcore + mxfp4_moe_w2_tensorcore + ancillary | **102.1 ms** |

* **Root cause**: at bs=1 * top_k=6 routes, effective `M=6`. Blackwell
  SM120 block-scaled MXFP4 × FP8 MMA tiles are `16x16` minimum so the
  persistent tensorcore path wastes `~10/16` of its M-axis per tile.
  The warp kernel runs one route per warp with no tile padding.
* **Verdict**: structural to small-M tensorcore, not fixable by
  tweaking the persistent orchestrator. The fix is per-call
  kernel-family selection (M-aware dispatch, landed in commit
  `1aa425c`).

## M-aware MoE Dispatch (LANDED, ship default)

* **Hypothesis**: tensorcore wins at large M (prefill), warp wins at
  small M (decode); dispatch per-call.
* **Implementation**: `TOKENSPEED_SM12X_MXFP4_MOE_IMPL=auto` (new
  default) picks warp for `M < threshold (=16)`, tensorcore for
  `M >= threshold`. Threshold matches the MMA tile width.
* **Bench**:

  | Variant | tok/s | TTFT (ms) | TPOT (ms) |
  |---|---|---|---|
  | baseline_unset (pre-flip warp default) | 19.64 | 5703 | 45.39 |
  | **auto_default** (new SM120 default) | **20.14** | **4477** | **45.31** |
  | auto_threshold1 (force tensorcore everywhere) | 18.30 | 4470 | 50.33 |

* **Headline**: decode TPOT unchanged (M=6 routes to warp, identical
  kernel path); prefill TTFT `-21.5%` (`M=6144` routes to tensorcore
  where it wins). `auto_threshold1` reproduces the T3-α regression
  when forced, confirming the dispatch actually flips.
* **W2 bias fuse** (commit `3e7c51f`): the persistent tensorcore W2
  kernel originally silently dropped `w2_bias`. The fuse adds
  `bias[local_expert_id * hidden + row]` at the kernel's final write of
  `per_pair_out`. This let `auto` drop its `w2_bias is None` precondition
  and broadens the path's reach to checkpoints with real non-zero
  bias. DSv4-Flash itself has zero w2_bias, so the kernel-level
  speedup is just the freed gating constraint; other models benefit
  more.

## T2-α-2 Path B: SWA-only Specialised K-split Kernel (LANDED)

* **Hypothesis**: SWA-only layers (`extra_width == 0`) pay a per-
  candidate `use_extra` branch + an unused `extra_lens[token]` gmem
  read inside the K-split chunk kernel; templating on
  `bool kSwaOnly` lets the compiler eliminate both.
* **Implementation**: `sparse_mla_fp8_cache_online_softmax_k_split_kernel`
  renamed to `_impl` and templated. The launcher dispatches on
  `extra_width == 0`. 3/44 layers are SWA-only on DSv4-Flash.
* **Bench**:

  | Variant | tok/s | TTFT (ms) | TPOT (ms) |
  |---|---|---|---|
  | pre-Path-B (auto_default) | 20.14 | 4477 | 45.31 |
  | **pathb_default** (auto + K-split=8 + SwaOnly template) | **20.29** | **4166** | **45.27** |
  | pathb_ksplit1 (K-split kill switch) | 17.52 | 4182 | 53.04 |

* **Headline**: TPOT essentially unchanged (45.27 vs 45.31), but TTFT
  `-7%` (`4166` vs `4477`). The TTFT win is the SwaOnly template
  helping SWA-only layers during chunked prefill (where they still
  hit the K-split path).
* **Status**: LANDED. Correctness-neutral, perf-neutral on TPOT,
  perf-positive on TTFT.

## NCCL Overlap (DEFERRED)

* **Observation** (from the T3-α trace): NCCL all-reduce per-decode-step
  total was `15.01 ms` on warp vs `11.25 ms` on persistent (`-25%`),
  same `440` instances per process.
* **Diagnosis**: a follow-up 3-decode-step nsys trace at the current
  `auto` default showed NCCL kernel times are heavily long-tailed:
  p50 `~6 us`, avg `37-88 us`, max can hit `2.37 ms`. The "persistent
  -25%" measurement was an average pulled down by fewer tail outliers
  on the persistent path, not a per-call kernel speedup.
* **Mechanism**: tail variance is from cross-rank NCCL ring-LL barriers
  waiting on the slowest peer. Persistent's longer CPU-side launch
  sequence happened to give NCCL more time to settle before barriers
  hit. Not a programmable optimization on warp -- we don't want to
  artificially slow down warp launches.
* **Real overlap-engineering** (moving NCCL to a side stream so it
  overlaps with MoE) would touch `python/tokenspeed/runtime/distributed/comm_manager.py`
  `post_mlp_comm`, validate across TP/EP topologies, and is bounded
  by EP=2 having limited overlap room. Substantial project, deferred
  to a future overlap-engineering pass.
* **Status**: DEFERRED. Observation documented; no actionable code
  change in this phase.

## Threshold Tuning at Batched Decode

* **Hypothesis**: at higher concurrency, decode `M = bs * top_k`
  crosses the auto threshold and routes to tensorcore. Microbench at
  `M~12` said tensorcore is `1.20-1.27x` faster than warp; runtime
  hadn't been validated above bs=1.
* **Sweep** (partial, see "Bench tooling bug" below):

  | Variant | tok/s | TTFT (ms) | TPOT (ms) |
  |---|---|---|---|
  | bs1_th16 (default, M=6 -> warp) | 20.23 | 4177 | 45.41 |
  | bs2_th16 (concurrency=2, M=12 -> warp) | 27.11 | 7413 | 63.73 |
  | v2_bs2_th8 (concurrency=2, M=12 -> tensorcore) | 26.88 | 7423 | **58.8** |
  | v2_bs4_th16 (concurrency=4, M=24 -> tensorcore) | TIMEOUT | -- | -- |
  | v2_bs4_warp (concurrency=4, M=24 forced warp) | TIMEOUT | -- | -- |

* **Findings**:
  * bs=1: warp is the correct route, decode unchanged from baseline.
  * bs=2: tok/s essentially identical between warp and forced-
    tensorcore (`27.11` vs `26.88`, within noise), but TPOT is
    `-7.7%` on tensorcore (`58.8` vs `63.73 ms`). A single data
    point hint that the threshold of `16` may be slightly high;
    bs=2 (M=12) might benefit from tensorcore. Needs an A/B with
    the same bench tool to confirm, since the two numbers came from
    different bench harnesses (the original `tokenspeed bench serve`
    vs a curl-based replacement).
  * bs=4: not closeable -- both the original bench tool
    (``tokenspeed bench serve``) and the curl-based replacement
    timed out at the test's 10-min budget on this workload size
    (8 prompts x 1024 in + 1024 out at concurrency=4). Server-side
    capacity is the bottleneck, not a programmable issue.

* **Bench tooling bug**: ``tokenspeed bench serve`` at
  ``--max-concurrency 4`` against a server with
  ``--max-num-seqs 4 --max-total-tokens 4096`` hung for 4 hours
  on cleanup despite the server completing all requests. Root cause
  is in ``python/tokenspeed/bench.py``:
  ``aiohttp.ClientSession`` created without ``async with`` (never
  explicitly closed) + a single ``asyncio.gather`` on all task
  futures with a 6-hour per-request timeout. One leaked stream-
  response future blocks the entire gather. Workaround
  ``/tmp/tk-serve/curl_bench.py`` (a 30-line Python script using
  ``concurrent.futures.ThreadPoolExecutor`` + ``urllib`` streaming
  reads) avoids the issue. Filing a real fix is out of scope for
  this experiment phase.

* **Verdict**: keep the default threshold at `16`. The principled
  reason (matches Blackwell MMA tile width) stands; the bs=2 hint
  is suggestive but not conclusive. Revisit if we actively support
  high-concurrency decode.

## Cumulative Code Footprint vs `upstream/main`

`78 files changed, +22,417 -368` as of commit `31b01d1`. An audit
spawned 2026-05-13 found:

* ~650 lines of dead code (the `scalar` MoE backend's whole chain, a
  retired C++ FFI alias, an old `inv_rope_grouped` CUDA path
  superseded by the output-projection island, and four
  `register_kernel` wrappers in `ops/attention/deepseek_v4/cuda.py`
  that are no longer imported by the runtime).
* ~30 lines of LIVE-but-INERT code (the `_SM12X_MOE_SCRATCH_CACHE`
  module-level cache from the A1 hypothesis; A/B showed zero perf
  delta).
* ~1500 lines of intentional probe infrastructure (the MXFP4×FP8 MMA
  tile / dense probes documented in this archive; per-kind K-split
  dispatch knobs).

Cleanup is queued behind this archive landing.

## Capture Infrastructure (kept)

* `_run_target_forward` NVTX `decode_step` range — zero-overhead
  off-path (`TOKENSPEED_NVTX=1` / `--enable-nvtx` to arm).
* `/tmp/tk-serve/nsys_decode_step.sh` capture wrapper on the SM120
  workstation. Re-usable for future T3-x diagnoses.
* `/tmp/tk-serve/diff_traces.py` + `diff_kernels.py` -- NVTX summary
  + CUDA kernel summary side-by-side diff tools for nsys trace
  comparison.
