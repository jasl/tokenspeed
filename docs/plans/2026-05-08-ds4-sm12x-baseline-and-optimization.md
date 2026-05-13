# DeepSeek V4 SM12x Baseline And Optimization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Establish current official vLLM B300 baselines, then use those baselines to drive TokenSpeed SM120 DeepSeek V4 Flash optimization without sacrificing correctness.

**Architecture:** B300 is the reference host for official vLLM behavior and long-running quality/eval captures. SM120 is the implementation and performance host for TokenSpeed. Every risky kernel/runtime change must first pass focused unit tests, then token-level oracle comparison, then the broader eval gates before it can be treated as a candidate default.

**Tech Stack:** vLLM official DeepSeek V4 Flash serving on 2x B300, TokenSpeed on 2x RTX PRO 6000 Blackwell, repo-independent `ds4-sm120-harness`, CUDA SM120f native kernels, OpenAI-compatible HTTP evals.

---

### Task 1: Capture Current B300 Reference Baselines

**Files:**
- Read: `/root/vllm-ds4-sm120-harness`
- Read: `/root/vllm`
- Artifact: `/root/vllm-ds4-sm120-harness/artifacts/codex_official_b300_current/2x_nvidia_b300_sxm6_ac/...`
- Artifact: `/root/vllm-ds4-sm120-harness/baselines/YYYYMMDD_official_b300_tp2_current_*`

**Steps:**
1. Start official vLLM on B300 with TP=2, EP, FP8 KV cache, FP4 indexer cache, `deep_gemm_mega_moe`, DeepSeek V4 tokenizer/tool/reasoning parsers, and CUDA Graph enabled by default.
2. Export no-MTP oracle responses with top-20 logprobs for the checked-in harness cases.
3. Run acceptance gates: health, quick smoke, generation matrix, and ToolCall-15.
4. Run GSM8K through the harness `lm_eval` wrapper.
5. Run synthetic shape benchmarks for `1024x1024` and a long-prefill shape.
6. Stop the server and verify port 8000 and GPU compute processes are empty.
7. Repeat the same capture with MTP enabled using `{"method":"mtp","num_speculative_tokens":2}`.

### Task 2: Promote Eval Gates For TokenSpeed Iteration

**Files:**
- Modify: `docs/plans/2026-05-07-sm12x-mxfp4-native.md`
- Optional create/modify: TokenSpeed-side scripts or docs only if the existing harness invocation is too error-prone.

**Steps:**
1. Treat B300 no-MTP oracle as the strict token-level reference for current no-MTP TokenSpeed work.
2. Treat B300 MTP oracle, GSM8K, ToolCall-15, and generation transcripts as promotion gates for future MTP/CUDA Graph work.
3. Keep the existing SM120 vLLM no-MTP oracle available only as a compatibility baseline until the new B300 bundles are captured and copied locally.

### Task 3: Attack The Current SM120 Bottleneck

**Files:**
- Modify: `tokenspeed-kernel/python/tokenspeed_kernel/thirdparty/cuda/csrc/deepseek_v4_attention.cu`
- Modify: `tokenspeed-kernel/python/tokenspeed_kernel/thirdparty/cuda/deepseek_v4_attention.py`
- Modify: `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/deepseek_v4/`
- Modify: `python/tokenspeed/runtime/layers/attention/backends/deepseek_v4.py`
- Test: `tokenspeed-kernel/test/ops/test_deepseek_v4_reference.py`
- Test: `test/runtime/test_deepseek_v4_attention_ops.py`

**Steps:**
1. Write a failing kernel-level test for a direct FP8 cache-reading sparse MLA decode primitive that matches the existing dequantized-workspace reference.
2. Implement the direct cache read path behind an opt-in environment flag.
3. Run focused kernel/runtime tests on SM120.
4. Run oracle comparison against the current no-MTP reference.
5. Benchmark `1024x1024` and profile decode to confirm whether workspace dequantization and sparse MLA time moved.

### Task 4: Replace The Decode Indexer Hot Path

**Files:**
- Modify: DeepSeek V4 indexer runtime/kernel files identified by profiling.
- Test: Add focused top-k/indexer tests before implementation.

**Steps:**
1. Write tests that compare decode top-k indices and lengths against the current reference implementation over deterministic FP8 indexer-cache inputs.
2. Implement a CUDA decode top-k/indexer path behind an opt-in flag.
3. Verify token oracle correctness, then benchmark and profile.

### Task 5: Move MoE Toward The PR #324 Shape

**Files:**
- Modify: `tokenspeed-kernel/python/tokenspeed_kernel/thirdparty/cuda/csrc/sm12x_mxfp4_moe.cu`
- Modify: `tokenspeed-kernel/python/tokenspeed_kernel/thirdparty/cuda/sm12x_mxfp4.py`
- Modify: `tokenspeed-kernel/python/tokenspeed_kernel/ops/moe/sm12x_mxfp4/`
- Test: `tokenspeed-kernel/test/ops/test_sm12x_mxfp4.py`

**Steps:**
1. Preserve the current scalar/warp routes as correctness fallbacks.
2. Add a persistent routed schedule only after the attention/indexer bottleneck is reduced enough for MoE improvements to be measurable.
3. Fuse route expansion, W13, SwiGLU/quant, W2, and route reduce in stages, keeping each stage guarded by unit tests and oracle comparison.

### Promotion Gates

Before treating any implementation as a candidate default, require:

- Focused TokenSpeed unit tests pass on SM120.
- Token-level oracle comparison passes against the chosen baseline.
- GSM8K does not regress against the B300 reference band.
- ToolCall-15 is at least the B300 baseline score for the same mode.
- Generation samples are exported for human review with no obvious parser/template degradation.
- `1024x1024` output throughput improves without correctness regression.

---

### 2026-05-09 Rebase / Cleanup Note

This branch was rebased onto latest `upstream/main` after upstream added its
own DeepSeek V4 MegaMoE and compressed-KV performance path. The conflict
resolution keeps upstream's official DeepSeek V4 model, cache, and helper-op
files as the baseline, and layers back only the SM12x-only fallback and native
kernel entry points.

Earlier exploratory runtime switches that were correct but slower remain
historical evidence only. Do not treat mentions of removed switches in the
older progress log below as active runtime surfaces; the canonical rejected
experiment list is `docs/notes/2026-05-09-ds4-sm12x-rejected-experiments.md`.

### 2026-05-08 Progress Notes

- B300 current no-MTP official vLLM run dir:
  `/root/vllm-ds4-sm120-harness/artifacts/codex_official_b300_current/2x_nvidia_b300_sxm6_ac/nomtp_20260508132655`.
  Top-20 oracle export completed for five cases. The `1024x1024` random
  benchmark reported `74.29` output tok/s at C1, `143.91` at C2, `258.03` at
  C4, and `409.41` at C8. This run did not configure `--reasoning-config`, so
  request payloads that include `thinking_token_budget` were rejected by vLLM
  with HTTP 400. Keep this run for no-MTP speed/oracle evidence, but do not use
  it as the final generation-quality baseline.
- A corrected B300 no-MTP run was started at
  `/root/vllm-ds4-sm120-harness/artifacts/codex_official_b300_current/2x_nvidia_b300_sxm6_ac/nomtp_reasoningcfg_20260508153252`
  with explicit reasoning boundaries:
  `{"reasoning_parser":"deepseek_v4","reasoning_start_str":"<think>","reasoning_end_str":"</think>"}`.
  Use this run for generation, ToolCall-15, GSM8K, and later MTP comparison.
- Added a direct FP8 cache-reading sparse MLA decode primitive for DeepSeek V4.
  It is exported through `tokenspeed_kernel.ops.attention.deepseek_v4` and
  remains opt-in through `TOKENSPEED_DEEPSEEK_V4_CUDA_SPARSE_MLA_CACHE=1`.
  Focused SM120 tests passed, and the top-20 oracle gate accepted all five
  cases.
- With `TOKENSPEED_DEEPSEEK_V4_CUDA_SPARSE_MLA_CACHE_MAX_TOPK=512`, SM120
  `1024x1024 c2` improved from the previous `9.97` output tok/s reference to
  `11.31` output tok/s. Profiling showed 640-wide decode candidates were still
  falling back to the workspace path.
- Raising the cache-read guard to `1024` let those 640-wide candidates stay on
  the direct cache path. Oracle still passed, and `1024x1024 c2` improved to
  `12.26` output tok/s. The FP4 indexer-cache configuration was also tested
  and passed oracle, but regressed the same benchmark to `11.43` output tok/s,
  so FP8 indexer cache remains the faster TokenSpeed path for now.
- Added the full-candidate CSA indexer fast path. When the compressed candidate
  count is at or below `index_topk`, the runtime now returns the full local
  candidate range instead of reading the indexer cache and running weighted
  top-k. A second fast path returns before Q projection/prepare when no
  filtering is needed, while preserving compressor/cache insert so longer
  contexts can still start filtering later.
- After the indexer fast paths, SM120 no-MTP `1024x1024 c2` reached `14.03`
  output tok/s, `28.06` total tok/s, mean TTFT `1620.01 ms`, and mean TPOT
  `141.12 ms`. The run dir is
  `<remote-workspace>/tokenspeed_indexer_earlyfull_20260508223753`.
- The cache-read sparse MLA guard now defaults to `1024`, covering the observed
  DeepSeek V4 decode width without requiring an extra env knob. A dynamic-width
  full-candidate top-k follow-up keeps the decode TPOT in the same band
  (`139.91 ms` in `<remote-workspace>/tokenspeed_indexer_dynamicwidth_20260508225531`),
  though that run had a slower batched prefill and lower aggregate output tok/s
  because mean TTFT rose to `6010.64 ms`.
- With the same implementation and `max-num-seqs=4`, `1024x1024 c4` reached
  `23.27` output tok/s, `46.53` total tok/s, mean TTFT `9730.17 ms`, and mean
  TPOT `155.96 ms`. The run dir is
  `<remote-workspace>/tokenspeed_indexer_earlyfull_c4_20260508224651`. C4
  improves aggregate throughput, but prefill/TTFT grows sharply and decode TPOT
  is not better than C2, so this is not the path to the 80-120 tok/s target by
  itself.
- The latest focused profile run is
  `<remote-workspace>/tokenspeed_indexer_earlyfull_profile_20260508224228`.
  Decode `attention.indexer` dropped from about `1.45 ms` per record before
  the full-candidate fast paths to about `0.54 ms`. Remaining decode hotspots
  are now routed/shared MoE, direct cache-read sparse MLA, QKV/output
  projections, compressor/cache insert, and per-layer launch overhead.
- A sparse MLA register-reuse pass cached each thread's Q slice and reused the
  per-candidate KV values from the score dot product for value accumulation.
  Oracle comparison against the current B300 no-MTP top-20 bundle passed all
  five cases. `1024x1024 c2` improved to `14.81` output tok/s with mean TPOT
  `119.91 ms` in
  `<remote-workspace>/tokenspeed_sparsemla_regreuse_20260508231102`.
- A follow-up UE8M0 scale decode fast path also passed oracle but regressed the
  same `1024x1024 c2` benchmark to `14.23` output tok/s and mean TPOT
  `125.34 ms` in
  `<remote-workspace>/tokenspeed_sparsemla_scale_decode_20260508232221`.
  That change was rejected and reverted; keep the register-reuse pass.
- Existing SM12x MXFP4 `tensorcore` MoE forward with the W13/SwiGLU,
  W2/reduce, and N64 grouped-dense fusion env knobs was tested in
  `<remote-workspace>/tokenspeed_moe_tensorcore_20260508233604`. Oracle
  comparison passed all five B300 no-MTP cases, but `1024x1024 c2` regressed to
  `12.59` output tok/s with mean TPOT `152.97 ms`. Do not promote the current
  tensorcore switch as-is; the #324-style path needs a deeper persistent /
  mega-MoE rewrite instead of just enabling the existing segmented kernels.
- A sparse MLA warp-reduction pass replaced the shared-memory block reduction
  loops with shuffle-based warp sums plus one warp-partial reduction. Focused
  SM120 tests passed, B300 no-MTP oracle accepted all five cases, and
  `1024x1024 c2` improved to `15.28` output tok/s with mean TPOT `120.11 ms`
  in `<remote-workspace>/tokenspeed_sparsemla_warpreduce_20260508234748`.
  Keep this change with the register-reuse pass; the gain is modest but
  directionally positive and does not change the correctness envelope.
- A follow-up profile run with `TOKENSPEED_DEEPSEEK_V4_PROFILE=1` was captured
  in `<remote-workspace>/tokenspeed_sparsemla_warpreduce_profile_20260508235723`
  using a `1024x16 c1` request. The decode profile is now spread across
  several similarly sized costs rather than one dominant kernel: mean CUDA time
  per layer/rank was about `0.55 ms` for `attention.indexer`, `0.52 ms` for
  `attention.sparse_attention`, `0.40 ms` for `attention.project_qkv`,
  `0.31 ms` for `attention.output_projection`, `0.28 ms` for
  `attention.compressor`, and `0.25 ms` for `attention.sparse_mla_fp8_cache`.
  Existing MoE routed/shared stages were lower in this profile (`~0.21-0.23 ms`
  per layer/rank), so the next high-leverage path is reducing per-layer decode
  launch count and fusing indexer/attention work, not simply enabling the
  current tensorcore MoE switch.
- Added an opt-in CUDA FP8 CSA indexer cache insert path under
  `TOKENSPEED_DEEPSEEK_V4_CUDA_CSA_INDEXER_CACHE=1`. The kernel fuses the
  overlap-window softmax compression, RMSNorm, RoPE, Hadamard rotation, and FP8
  cache write for the indexer cache. The focused CUDA/env-route regression test
  passed, full attention ops passed (`42 passed`), and kernel/platform refs
  passed (`48 passed`). Run
  `<remote-workspace>/tokenspeed_csa_indexer_cuda_20260509001042` passed the
  B300 no-MTP oracle for all five cases and reached `15.34` output tok/s on
  `1024x1024 c2` with mean TPOT `115.24 ms`. This is a small decode-latency
  improvement over the warp-reduce run, but not a material throughput step; keep
  it opt-in while the larger launch-count / attention-indexer fusion work
  proceeds.
- Removed unnecessary padded-head work from the SM12x direct sparse MLA cache
  decode path. FlashMLA needs padded heads, but the TokenSpeed CUDA cache kernel
  can run on the real local-head count and slice `attn_sink` to match. The
  red/green regression test now asserts the cache path receives unpadded query
  heads, and the focused attention suite passed (`43 passed`). Run
  `<remote-workspace>/tokenspeed_unpadded_sparsemla_20260509001859` passed
  the B300 no-MTP oracle for all five cases and improved `1024x1024 c2` to
  `15.78` output tok/s with mean TPOT `116.00 ms`. This confirms padded-head
  work was real waste, but the remaining gap to the 80-120 tok/s target is still
  dominated by per-layer multi-stage decode overhead rather than this single
  sparse MLA head count.
- Removed the host-side `cudaMemsetAsync(output, 0, ...)` from the direct FP8
  cache-reading sparse MLA launch. The kernel already writes every output
  element, including the empty-candidate path, so the memset was unnecessary
  launch overhead on decode. Kernel/platform reference tests passed (`48
  passed`), the B300 no-MTP top-20 oracle accepted all five cases, and
  `<remote-workspace>/tokenspeed_sparsemla_nomemset_20260509002957` improved
  `1024x1024 c2` to `15.88` output tok/s with mean TPOT `115.21 ms`. This is a
  small but clean win; it also confirms the current bottleneck is no longer a
  single redundant sparse-MLA-side launch.
- Delayed padded-head query construction in decode until the workspace fallback
  or FlashMLA branch actually needs it. The direct SM12x FP8 cache path now
  fast-returns without building padded query heads. The regression test covers
  this by making `_pad_query` fail if the cache path is available. The B300
  no-MTP oracle still accepted all five cases, but
  `<remote-workspace>/tokenspeed_delaypad_20260509004410` measured `15.85`
  output tok/s and mean TPOT `115.42 ms` on `1024x1024 c2`, so this is a
  correctness-preserving cleanup rather than a confirmed throughput win.
- Added an opt-in CUDA decode-index generation path under
  `TOKENSPEED_DEEPSEEK_V4_CUDA_DECODE_INDICES=1`. It supports the current
  hot decode cases, SWA-only and CSA top-k, and leaves HCA on the Python
  fallback. The kernel maps per-token local indices through the request block
  table and compacts CSA top-k entries in one launch, avoiding the previous
  Python loops, `.item()` synchronizations, and small tensor ops on covered
  layers. Focused runtime tests passed, kernel/platform tests passed (`49
  passed`), ruff passed, the B300 no-MTP oracle accepted all five cases, and
  `<remote-workspace>/tokenspeed_decode_indices_cuda_20260509005715` improved
  `1024x1024 c2` to `16.68` output tok/s with mean TPOT `109.14 ms`. This is
  the first material post-sparse-MLA decode-stage win and should stay opt-in
  until broader eval gates pass.
- Reused the CPU `max_seq_len` already computed in DeepSeek V4 forward
  metadata to guard the CSA full-candidate indexer branch. This removes a
  per-CSA-layer `lengths.max().item()` GPU sync while preserving the long
  context fallback to the real indexer when `max_seq_len // compress_ratio`
  exceeds `index_topk`. Runtime tests passed (`46 passed`) and ruff passed.
  The SM120 oracle still accepted all five B300 no-MTP cases, and
  `<remote-workspace>/tokenspeed_metadata_maxseq_20260509011032` measured
  `16.20` output tok/s with mean TPOT `108.24 ms`; the lower aggregate
  throughput came from a TTFT outlier (`15674.89 ms`), while steady-state
  decode moved slightly in the right direction. A profile run at
  `<remote-workspace>/tokenspeed_metadata_maxseq_profile_20260509011510`
  showed `attention.indexer` around `0.314-0.319 ms`, down only modestly from
  the previous `0.324 ms`, so the remaining cost was the tensor-op top-k fill.
- Added `TOKENSPEED_DEEPSEEK_V4_CUDA_FULL_CANDIDATE_TOPK=1`, an opt-in CUDA
  kernel that fills the CSA full-candidate top-k table in one launch after the
  same `max_seq_len` safety guard. The red/green runtime route test and CUDA
  reference test passed, full attention ops passed (`47 passed`),
  kernel/platform refs passed (`50 passed`), and ruff passed. Run
  `<remote-workspace>/tokenspeed_fullcandidate_cuda_20260509012043` passed
  the B300 no-MTP oracle for all five cases and improved `1024x1024 c2` to
  `16.97` output tok/s with mean TPOT `107.10 ms`. The follow-up profile at
  `<remote-workspace>/tokenspeed_fullcandidate_cuda_profile_20260509012454`
  reduced `attention.indexer` to about `0.265-0.269 ms`. The hot path is now
  spread mainly across sparse attention, QKV/output projections, compressor,
  and direct FP8-cache sparse MLA; further indexer-only work has diminishing
  returns.
- Added a narrower direct full-candidate decode-index sentinel under
  `TOKENSPEED_DEEPSEEK_V4_CUDA_DIRECT_FULL_CANDIDATE_INDICES=1`. For the CSA
  decode case where the guarded candidate set is already the full compressed
  range, the runtime now passes a zero-width sentinel to the CUDA decode-index
  kernel and lets that kernel generate block-table-mapped slots directly. Full
  attention ops passed (`48 passed`), kernel/platform refs passed (`51
  passed`), and ruff passed. Run
  `<remote-workspace>/tokenspeed_direct_fullcandidate_indices_20260509013351`
  passed the B300 no-MTP oracle for all five cases and measured `16.43`
  output tok/s with mean TPOT `106.57 ms`; aggregate output throughput was
  lower than the previous run because TTFT again hit a `15665.91 ms` outlier.
  The profile at
  `<remote-workspace>/tokenspeed_direct_fullcandidate_indices_profile_20260509013827`
  put decode `attention.indexer` at about `0.259 ms`, only slightly below the
  full-candidate CUDA top-k route. Keep this as a cleaner decode-index path,
  but the next material performance work should target launch-count reduction
  and the larger per-layer stages rather than more standalone indexer kernels.
- Added `TOKENSPEED_DEEPSEEK_V4_CUDA_COMPRESSOR_STATE=1`, an opt-in CUDA
  state-save kernel for the DeepSeek V4 compressor/indexer compressor state.
  This removes the decode Python small-token branch with `.item()`, `copy_`,
  and `clone` from the hot path while preserving the exact APE write contract.
  The red/green CUDA reference test passed, full attention ops passed (`49
  passed`), kernel/platform refs passed (`52 passed`), and ruff passed. Run
  `<remote-workspace>/tokenspeed_compressor_state_cuda_20260509014907`
  passed the B300 no-MTP oracle for all five cases and measured `16.93`
  output tok/s with mean TPOT `102.94 ms`. The profile at
  `<remote-workspace>/tokenspeed_compressor_state_cuda_profile_20260509015318`
  reduced `attention.compressor` from about `0.279 ms` to `0.254 ms`,
  `indexer.compressor` from about `0.093 ms` to `0.064 ms`, and
  `attention.indexer` from about `0.261 ms` to `0.224 ms`.
- Added a decode-only non-boundary skip for compressed cache inserts when all
  active decode sequences have the same CPU-observed sequence length and that
  length is not on the compression boundary. This avoids entering the Python
  compressed-cache insert path, including its GPU `valid.any()` synchronization,
  on the common no-write decode steps. Full attention ops passed (`50
  passed`), kernel/platform refs passed (`52 passed`), and ruff passed. Run
  `<remote-workspace>/tokenspeed_skip_compressed_insert_20260509015858`
  passed the B300 no-MTP oracle for all five cases and improved `1024x1024 c2`
  to `17.65` output tok/s with mean TPOT `98.10 ms`. The profile at
  `<remote-workspace>/tokenspeed_skip_compressed_insert_profile_20260509020305`
  reduced decode `attention.compressor` further to about `0.155 ms` and
  `attention.indexer` to about `0.139 ms`; `indexer.cache_insert` profile
  records now appear only at compression-boundary steps.
- Added `TOKENSPEED_DEEPSEEK_V4_CUDA_INV_ROPE_GROUPED=1`, an opt-in CUDA
  inverse-RoPE grouped projection input kernel for the DeepSeek V4 output
  projection path. The kernel replaces the decode-side PyTorch inverse-RoPE,
  reshape, and contiguous materialization with one generic fp16/bf16 CUDA
  launch. The red/green CUDA reference test passed, full attention ops passed
  (`50 passed`), kernel/platform refs passed (`53 passed`), and ruff passed.
  Run `<remote-workspace>/tokenspeed_inv_rope_grouped_cuda_20260509020929`
  passed the B300 no-MTP oracle for all five cases and improved `1024x1024 c2`
  to `17.95` output tok/s with mean TPOT `96.24 ms`. The profile at
  `<remote-workspace>/tokenspeed_inv_rope_grouped_cuda_profile_20260509021328`
  reduced decode `attention.output_projection` to about `0.250 ms`. The
  remaining largest per-layer decode costs are now roughly
  `attention.sparse_attention` (`0.467 ms` outer, including
  `attention.sparse_mla_fp8_cache` at `0.248 ms`), `attention.project_qkv`
  (`0.404 ms`), `attention.output_projection` (`0.250 ms`), routed/shared MoE
  (`~0.21-0.23 ms` each), compressor (`0.155 ms`), and indexer (`0.138 ms`).
- The corrected B300 no-MTP run completed generation and ToolCall-15 capture.
  Generation exited cleanly with 315 JSONL rows, but ToolCall-15 exited nonzero:
  `/root/vllm-ds4-sm120-harness/artifacts/codex_official_b300_current/2x_nvidia_b300_sxm6_ac/nomtp_reasoningcfg_20260508153252/acceptance/toolcall15.json`
  reported `226/270` points, `84%`, and 26 failing case-rounds. Treat this as
  the current official-vLLM measurement, not as a passing promotion gate, until
  the expected ToolCall-15 threshold is clarified or a rerun changes the score.
- Added `TOKENSPEED_DEEPSEEK_V4_CUDA_COMPRESSED_KV_CACHE=1`, an opt-in CUDA
  compressed-KV cache insert path shared by CSA/HCA compression layouts. The
  kernel performs the compression-window softmax, RMS norm, RoPE, and FP8 cache
  write without entering the Python small-token branch. The red import failure
  was reproduced before implementation, then the focused CUDA/reference test
  passed, full attention ops passed (`51 passed`), kernel/platform refs passed
  (`23 passed`), and ruff passed. Run
  `<remote-workspace>/tokenspeed_compressed_kv_cuda_20260509022900` passed
  the B300 no-MTP oracle for all five cases and improved `1024x1024 c2` to
  `18.67` output tok/s with mean TPOT `91.93 ms`. The profile at
  `<remote-workspace>/tokenspeed_compressed_kv_cuda_profile_20260509023300`
  showed decode `attention.compressor` around `0.079 ms`; the remaining
  higher-cost decode stages were sparse attention, QKV projection, output
  projection, and decode index generation.
- Extended `TOKENSPEED_DEEPSEEK_V4_CUDA_DECODE_INDICES=1` to HCA-style dense
  compressed candidates (`compress_ratio=128`) instead of leaving those layers
  on the Python index builder. The new path uses the same metadata max sequence
  bound to allocate a dense compressed slot buffer without `positions.max()`
  synchronization, and maps local compressed positions through the request
  block table inside the CUDA decode-index kernel. The red/green CUDA
  reference and runtime route tests passed, full attention ops passed (`52
  passed`), kernel/platform refs passed (`24 passed`), and ruff passed. Run
  `<remote-workspace>/tokenspeed_hca_decode_indices_20260509024229` passed
  the B300 no-MTP oracle for all five cases and improved `1024x1024 c2` to
  `20.72` output tok/s with mean TPOT `85.86 ms`. The profile at
  `<remote-workspace>/tokenspeed_hca_decode_indices_profile_20260509024627`
  reduced decode `attention.decode_indices` to about `0.038 ms`; the largest
  remaining decode costs are now `attention.project_qkv` (`0.416 ms`),
  `attention.sparse_attention` (`0.372 ms`, including
  `attention.sparse_mla_fp8_cache` at `0.256 ms`), and
  `attention.output_projection` (`0.233 ms`).
- The B300 official-vLLM no-MTP GSM8K baseline completed with exit code 0 in
  `/root/vllm-ds4-sm120-harness/artifacts/codex_official_b300_current/2x_nvidia_b300_sxm6_ac/nomtp_reasoningcfg_20260508153252/eval_gsm8k`.
  `lm-eval` reported `gsm8k` 8-shot exact-match flexible `0.9514783927` and
  strict `0.9522365428`. Use this current B300 run as the GSM8K reference for
  later TokenSpeed correctness/quality gates, while keeping the ToolCall-15
  score above as a non-passing baseline until the threshold is clarified.
- Added a native SM12x CUDA online MXFP8 block-128 activation quantizer under
  `TOKENSPEED_SM12X_MXFP8_CUDA_QUANTIZE=1` for the
  `triton_mm_fp8_blockscale` path. The new unit test first failed on the
  missing op, then the CUDA quantizer and routed GEMM integration passed (`3
  passed`). Run `<remote-workspace>/tokenspeed_mxfp8_cuda_quant_20260509030440`
  passed the B300 no-MTP oracle for all five cases and improved `1024x1024 c2`
  to `21.13` output tok/s with mean TPOT `79.44 ms`. The profile at
  `<remote-workspace>/tokenspeed_mxfp8_cuda_quant_profile_20260509030820`
  showed QKV/output/shared-expert projection costs moving down, while sparse
  MLA and routed MoE remained the largest decode stages.
- Tried TokenSpeed's FlashMLA sparse decode route under
  `TOKENSPEED_DEEPSEEK_V4_FLASHMLA_DECODE_SM12X=1`, but the current packaged
  `flash_mla_cuda.sparse_decode_fwd` rejected SM120 with
  `Unsupported architecture for sparse decode fwd`. Keep this disabled until
  the FlashMLA package grows an SM12x sparse decode build.
- Cached the seven FP8 cache scale factors per sparse-MLA candidate in shared
  memory inside `sparse_mla_fp8_cache_kernel`. Focused sparse-MLA cache tests
  passed (`5 passed` with the quantizer tests included), and run
  `<remote-workspace>/tokenspeed_sparse_scale_cache_20260509031533` passed
  the B300 no-MTP oracle and measured `21.34` output tok/s with mean TPOT
  `78.56 ms`. The effect is small but positive enough to keep. A full profile
  at `<remote-workspace>/tokenspeed_sparse_scale_cache_fullprofile_20260509032142`
  showed that C2 decode records are mostly `tokens=2`; at that width,
  `attention.sparse_mla_fp8_cache` was still the main sparse-attention cost
  (`0.453 ms` mean, `0.211 ms` p50), rising to about `0.844 ms` at
  `extra=512,swa=128`.
- Added an opt-in online-softmax sparse MLA FP8 cache path,
  `TOKENSPEED_DEEPSEEK_V4_CUDA_SPARSE_MLA_ONLINE_SOFTMAX=1`, that streams
  candidates once instead of computing logits/max in one pass and values in a
  second pass. The new wrapper/runtime tests passed with the existing cache
  reference tests (`4 passed`). Run
  `<remote-workspace>/tokenspeed_sparse_online_softmax_20260509033556`
  passed the B300 no-MTP oracle and measured `22.30` output tok/s with mean
  TPOT `79.03 ms`; the follow-up profile
  `<remote-workspace>/tokenspeed_sparse_online_softmax_fullprofile_20260509033934`
  showed the real benefit in-kernel: `attention.sparse_mla_fp8_cache` dropped
  from `0.453/0.211 ms` mean/p50 to `0.306/0.157 ms`, and `extra=512,swa=128`
  dropped from `0.844 ms` to `0.551 ms`. Keep this as the current sparse MLA
  candidate path, but still treat end-to-end throughput as noisy until repeated
  longer runs confirm steady TPOT.
- Rejected a `__expf` fast-exp variant for the online sparse MLA softmax. It
  passed the focused tests and B300 no-MTP oracle, but
  `<remote-workspace>/tokenspeed_sparse_online_fast_exp_20260509034359`
  regressed C2 to `21.18` output tok/s and mean TPOT `79.23 ms`; the code was
  reverted to precise `expf`.
- A/B tested existing SM12x MXFP4 MoE implementation switches with online
  sparse MLA enabled. `TOKENSPEED_SM12X_MXFP4_MOE_IMPL=tensorcore` passed the
  current oracle threshold but regressed decode badly:
  `<remote-workspace>/tokenspeed_tensorcore_moe_online_20260509034939`
  measured only `19.05` output tok/s with mean TPOT `103.33 ms`. The `tile4`
  route also failed to improve the stable path:
  `<remote-workspace>/tokenspeed_tile4_moe_online_20260509035308` measured
  `21.35` output tok/s with mean TPOT `79.89 ms` and had a weaker long-prefill
  oracle trajectory. Keep the default `warp` MoE route for now. The next MoE
  step should be a real #324-style grouped MXFP4 tensorcore rewrite rather than
  simply enabling the current tensorcore wrapper.
- Cleaned up rejected runtime paths before the next MoE iteration:
  `TOKENSPEED_DEEPSEEK_V4_FLASHMLA_SM12X`,
  `TOKENSPEED_DEEPSEEK_V4_FLASHMLA_DECODE_SM12X`, and the
  `TOKENSPEED_SM12X_MXFP4_MOE_IMPL=tensorcore/tile4` deploy switches are
  removed. The lower-level grouped MXFP8xMXFP4 primitive tests remain as
  scaffolding for a fresh #324-style implementation.
- Added a clean experimental #324-style grouped tensorcore MoE route under
  `TOKENSPEED_SM12X_MXFP4_MOE_IMPL=grouped_tc`. It uses the existing route
  expansion plus MXFP8 hidden quantization, the grouped MXFP8xMXFP4 SwiGLU
  primitive, and a fused W2 route-reduce kernel. Focused SM120 MoE tests passed
  (`28 passed` before the N64 reducer, `29 passed` after it) and ruff passed.
  The first full run
  `<remote-workspace>/tokenspeed_grouped_tc_online_20260509045145` passed the
  B300 no-MTP oracle for all five cases, but regressed `1024x1024 c2` to
  `19.17` output tok/s with mean TPOT `103.32 ms`. Keep this route opt-in only;
  it is correct enough for further kernel work but not a performance candidate.
- Added an N64 cooperative W2 route-reduce variant for the grouped tensorcore
  route. The new reducer covers the DeepSeek V4 hidden dimension because
  `4096 % 64 == 0` and falls back to the N8 reducer otherwise. The focused
  N64-vs-N8 regression test passed with `1e-4` tolerance; exact equality is not
  expected because the grouped grid changes FP32 `atomicAdd` accumulation order
  across route blocks. Full SM12x MXFP4 tests passed (`29 passed`) and ruff
  passed. The full run
  `<remote-workspace>/tokenspeed_grouped_tc_n64_online_20260509050530`
  passed the B300 no-MTP oracle for all five cases and measured `20.02` output
  tok/s with mean TPOT `99.01 ms` on `1024x1024 c2`. This is a small improvement
  over the first `grouped_tc` run, but it still loses to the current default
  warp MoE route with online sparse MLA (`22.30` output tok/s, mean TPOT
  `79.03 ms`). The next MoE step should therefore focus on eliminating the
  route-expansion / W13 / W2 boundary overhead and moving toward a persistent or
  mega-MoE schedule, not just widening the W2 reducer further.
- A small grouped-tensorcore profile was captured in
  `<remote-workspace>/tokenspeed_grouped_tc_n64_profile_20260509051259`.
  For decode `tokens=1`, profile logs put `moe.routed_experts` around
  `1.17 ms` per layer/rank, while the current warp route in the online sparse
  MLA profile was around `0.22 ms` for `tokens=1` and `0.32 ms` for `tokens=2`.
  A one-off low-level microbench with DeepSeek V4 Flash shapes
  (`hidden=4096`, `moe_intermediate=2048`, `topk=6`, `local_experts=128`)
  measured full `warp` MoE at about `0.27 ms` and full `grouped_tc` at about
  `0.94 ms`. The grouped route breakdown was about `0.08 ms` for
  route expansion plus hidden quantization, `0.79 ms` for W13/SwiGLU/requant,
  and `0.08 ms` for N64 W2/reduce. The core problem is that the current grouped
  tensorcore route pads six real decode routes into 96 rows so W13 burns too
  much work. Do not spend more effort on W2-only widening until the routed W13
  schedule can avoid this padding waste or batch routes persistently across
  enough tokens/layers to amortize it.
- Cleanup follow-up: the failed `grouped_tc` deploy surface was removed after
  the measurements above. `TOKENSPEED_SM12X_MXFP4_MOE_IMPL` now accepts only
  `warp` and `scalar`; the public `sm12x_mxfp4_moe_forward_grouped_tc` wrapper
  and exports were removed. Lower-level grouped primitives remain because they
  are useful testable material for a future padding-aware or persistent rewrite.
  Rejected experiments are summarized in
  `docs/notes/2026-05-09-ds4-sm12x-rejected-experiments.md`.
- vLLM `ds4-sm120-preview-dev` inspection: the no-MTP/no-CUDA-graph
  100 tok/s path is not using the rejected grouped TensorCore MoE shape. The
  strongest portable ideas are the horizontal MLA preprocess fusion, per-layer
  auxiliary-stream overlap for the smaller compressor/indexer GEMMs, SM12x
  low-token FP8 GEMM shape choices, and direct FP8 sparse MLA decode kernels.
  TokenSpeed already had the same fused SWA insert boundary, but its kernel used
  one 256-thread block per token/head slot. The next cleanup-compatible change
  replaces that kernel with a warp-per-slot implementation under the existing
  API, avoiding another runtime switch while reducing decode launch work.
- Replaced the fused SWA Q RMSNorm/RoPE/KV-cache insert kernel with a
  warp-per-slot implementation under the existing public API. Focused SM120
  tests passed, and the `1024x1024 c2` run
  `<remote-workspace>/tokenspeed_warpslot_qnorm_online_20260509053547`
  measured `23.56` output tok/s with mean TPOT `79.23 ms`. This is only a
  small gain over the online sparse MLA baseline (`22.30` output tok/s), but it
  keeps the implementation cleaner and aligns TokenSpeed's boundary with the
  vLLM SM12x branch.
- Verification gotcha: TokenSpeed only returns completion sampled-token
  logprobs when the server is started with `--enable-output-logprobs`. The first
  oracle compare without that flag returned empty actual token traces despite
  HTTP 200 responses and matching prompt token ids. Re-running the same B300
  no-MTP top-20 oracle with that flag in
  `<remote-workspace>/tokenspeed_warpslot_qnorm_logprobs_20260509054446`
  accepted all five cases; `completion_short_math_logprobs20` forked only at a
  low-margin step and remained within the current gate.
- Tried a vLLM-style auxiliary-stream input GEMM overlap in TokenSpeed. The
  first variant passed all five B300 no-MTP oracle cases but regressed
  `1024x1024 c2` to `22.57` output tok/s; after removing the unnecessary
  `indexer.weights_proj` precompute on the common full-candidate decode path it
  still measured only `22.78` output tok/s. The runtime switch was removed and
  the failed attempt was recorded in
  `docs/notes/2026-05-09-ds4-sm12x-rejected-experiments.md`. The useful lesson
  is that TokenSpeed's current eager decode needs coarser launch/scheduling
  changes; layer-local Python stream events are too fine-grained.
