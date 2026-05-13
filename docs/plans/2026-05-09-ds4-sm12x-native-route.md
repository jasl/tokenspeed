# DeepSeek V4 SM12x Native Route

## Goal

Build the SM120/SM121 DeepSeek V4 Flash path as a native TokenSpeed backend:
correctness first, then decode throughput, while keeping upstream's SM100 path
unchanged.

The target remains the no-MTP 80-120 output tok/s milestone first. MTP and CUDA
graph support stay as follow-up gates after the non-MTP path is correct and
fast enough.

## Upstream Infrastructure To Keep

- Use upstream `DeepseekV4TokenToKVPool` and `DeepseekV4ForwardMetadata` as the
  cache and decode-metadata source of truth.
- Use upstream DeepSeek V4 attention ops as correctness boundaries:
  compressed slot mapping, HCA/CSA cache insert, indexer cache IO, and indexer Q
  prepare.
- Use upstream `DeepseekV4MegaMoEExperts` as the shape of the high-performance
  expert path: raw checkpoint weights, post-load transform/finalize, preallocated
  buffers, fused expert call, and separate shared-expert handling.
- Keep upstream `--speculative-config` parsing and warmup-style thinking as the
  future MTP entrypoint, but do not make MTP part of the current throughput gate.
- Keep TF32-on-default behavior for router GEMMs.

## SM12x Gates

- SM12x is NVIDIA-only for this project. It is acceptable for the SM12x route
  to depend on NVIDIA-owned CuTe/CUTLASS infrastructure when that reduces local
  kernel-maintenance burden.
- Do not pursue CuTile / `cuda.tile` for this route. Its direction is promising,
  but CUTLASS/CuTe DSL is the more mature NVIDIA-maintained stack already
  present in this repo and on the SM120 workstation.
- Keep generic TokenSpeed infrastructure changes thin: backend registration,
  SM120/SM121 gating, fallback selection, tests, and docs are acceptable; broad
  scheduler/model-executor rewrites should only happen when a measured SM12x
  kernel/dataflow need requires them.
- Do not directly enable upstream fast mHC on SM12x until a measured SM12x
  implementation is available. The current DeepGEMM package exposes the symbol
  but rejects SM120 at runtime.
- Do not use TRT-LLM FP8 activation quant helpers on SM12x by default. They can
  select unsupported kernels or return incompatible scale shapes on the SM120
  workstation.
- Do not route SM12x through FlashInfer/FlashMLA SM100 sparse decode or
  `tcgen05` paths. They remain official-path dependencies for SM100, not SM12x
  building blocks.
- Do not call upstream TRT-LLM `fast_topk_v2` for the DeepSeek V4 indexer on
  SM12x. The reference `torch.topk` path is slower, but it is the current
  correct route until TokenSpeed owns an SM12x top-k kernel.

## Native MoE Direction

The next MoE backend should be a TokenSpeed-owned SM12x analogue of MegaMoE,
not the current per-token warp fallback:

1. Add `DeepseekV4Sm12xMoEExperts` beside upstream `DeepseekV4MegaMoEExperts`.
   It should reuse the same DeepSeek V4 model-level selection flow but dispatch
   only when `--moe-backend sm12x_mxfp4` and the platform is SM120/SM121.
2. Move runtime work out of the hot decode step:
   transform weight scales at load/finalize time, preallocate scratch buffers,
   and keep route staging buffers stable for future CUDA graph capture.
3. Make CUTLASS/CuTe DSL the preferred implementation substrate for the next
   tensorcore MoE attempt. Use #324 as a reference for SM120 MMA wrappers, scale
   packing, and heuristic shapes, but not as a literal copy. Avoid SM100
   assumptions such as TMA-first data movement or `tcgen05`-only schedules;
   validate any such path on SM120 before depending on it.
4. Optimize for real DeepSeek V4 Flash decode shape first: tiny `M`, top-k 6,
   EP=2, `hidden=4096`, `moe_intermediate=2048`. Large-M grouped GEMM TFLOPS are
   secondary unless they improve this shape.
5. Replace the current padding-heavy grouped route with a persistent or
   padding-aware schedule. The rejected grouped tensorcore wrapper failed because
   six real decode routes were padded to 96 rows.

## Native Attention Direction

Use the vLLM SM120 branch as design evidence for sparse MLA, but keep the
TokenSpeed implementation inside `tokenspeed-kernel`:

- Keep the current upstream cache layout and compressed slot metadata.
- Build a model-local SM12x attention island rather than continuing isolated
  per-kernel switches. The island should own the Q/KV projection, cache insert,
  sparse MLA decode, inverse-RoPE output projection, and scratch buffers behind
  one stable runtime boundary.
- Replace the current output projection sequence
  `inverse_rope_reference -> dequantized wo_a -> torch.bmm` with the vLLM-style
  dataflow: fused inverse-RoPE + block FP8 quantization, SM12x
  `bhr,hdr->bhd` FP8 einsum for `wo_a`, then the existing `wo_b` path. This is
  the first prototype because it is a clear dataflow delta from vLLM and is
  independent of MoE routing.
- Keep the existing SM12x FP8-cache sparse MLA decode kernel as the baseline,
  then port or reimplement vLLM's direct paged FP8-cache C4/C128/SWA split only
  after the output projection path is measured.
- Revisit auxiliary stream GEMM overlap only inside this coarser island. The
  earlier Python/layer-local stream switch is rejected because its scheduling
  overhead dominated.
- Treat CUDA graph as the delivery mechanism for the island, not the first
  source of speed. The graph gate remains useful only after the island removes
  the current 40 ms/step GPU-work floor.

## Validation Gates

Every deployable step must pass these before benchmarking:

- Focused unit tests on SM120.
- Harness oracle comparison against the current B300/vLLM no-MTP baseline with
  `--enable-output-logprobs`.
- GSM8K, ToolCall-15, and subjective generation-quality smoke once the online
  throughput path changes.
- `1024x1024` online throughput benchmark with saved run directory and exact
  server args.

## Current State

`DeepseekV4Sm12xMoEExperts` now exists beside upstream's
`DeepseekV4MegaMoEExperts`. It owns the DeepSeek V4 routed expert checkpoint
layout directly, is selected only by `--moe-backend sm12x_mxfp4` on SM120/SM121,
finalizes bias dtype at post-load time, and calls the existing SM12x MXFP4 MoE
kernel through a model-level fused-experts contract. The class also owns reusable
output and intermediate work buffers, and the low-level Python wrapper accepts
those buffers as optional workspaces while preserving the older allocating API.

This is still an API and data-flow cleanup, not the final high-performance
kernel. The current implementation deliberately keeps DecoderLayer's existing
pre/post MoE communication path for SM12x, unlike SM100 MegaMoE, because the
current SM12x kernel does not own expert-parallel collectives.

The full eager serving path now reaches multi-token completions on the SM120
workstation with `attn_tp_size=2`, `ep_size=2`, FP8 KV cache, FP4 indexer cache,
and `--moe-backend sm12x_mxfp4`. The current clean smoke run is
`<remote-workspace>/tokenspeed_smoke_model_level_sm12x_clean_decode_20260509173619`
and returned an 8-token deterministic completion with output logprobs. This is a
correctness milestone, not a throughput milestone: at this checkpoint SM12x
still used reference paths for SWA insert and indexer top-k.

The decode indexer now skips cache reads and logits/top-k work when the
compressed candidate length already fits within `topk_tokens`; for the common
`1024` context and `topk_tokens=512` shape, all compressed candidates are
selected anyway. The behavior is covered by
`test_deepseek_v4_indexer_decode_all_candidate_shortcut_skips_cache_read`, and
the broader focused SM12x regression stayed green on `the primary SM120 workstation`. A matched
`1024x128 c1` token-id benchmark improved from
`<remote-workspace>/tokenspeed_bench_sm12x_1024x128_20260509173825`
(`5.18` output tok/s, mean TPOT `160.55 ms`) to
`<remote-workspace>/tokenspeed_bench_sm12x_allcandidate_20260509174713`
(`5.46` output tok/s, mean TPOT `151.47 ms`). This removed one avoidable
reference path for short decode contexts, but at that checkpoint the service was
still dominated by the staged SM12x MoE pipeline and reference-heavy attention
cache insert path.
Keep benchmark launches constrained with `--max-total-tokens 4096` and
`--chunked-prefill-size 1024`; otherwise the default KV pool can leave too
little memory for the current per-layer SM12x MoE work buffers.

The SM12x SWA insert fallback was removed after a direct RTX Pro 6000 smoke
showed the existing native `fused_qnorm_rope_kv_insert` kernel matches the torch
reference on SM120. The runtime now uses the native op on SM12x when the kernel
package is present and keeps the torch fallback only for missing-op development
environments. Tests cover both contracts:
`test_sm12x_fused_qnorm_rope_kv_insert_uses_native_op_when_available`,
`test_sm12x_fused_qnorm_rope_kv_insert_falls_back_when_native_op_missing`, and
the real-op reference match. On `the primary SM120 workstation`, `test_deepseek_v4_attention_ops.py`
plus the missing-op config boundary reported `17 passed, 1 skipped`.

A matched `1024x128 c1` token-id benchmark at
`<remote-workspace>/tokenspeed_bench_sm12x_native_insert_20260509192455`
improved the all-candidate shortcut baseline from `5.46` output tok/s and mean
TPOT `151.47 ms` to `6.98` output tok/s and mean TPOT `132.12 ms`. A follow-up
profile at
`<remote-workspace>/tokenspeed_profile_sm12x_native_insert_20260509192640`
exported traces under `/tmp/20260509-192716-TP-{0,1}.trace.json.gz`. The old
SWA insert hotspot is gone: `insert_swa_cache` is now about `32 ms` across 731
calls per TP rank, and the native fused insert CUDA kernel itself is about
`5.2 ms` total. The remaining visible hotspots are the reference-heavy indexer
path (`run_indexer` about `1.13 s`, including
`deepseek_v4_prepare_indexer_q_reference` about `880 ms`), reference sparse MLA
decode (about `820 ms`), MHC pre-processing (about `1.17 s`), and the current
two-kernel SM12x MoE warp path (about `1.18 s` for gate/up plus down).

The decode all-candidate shortcut now runs before indexer Q projection and Q
preparation in `DeepseekV4Indexer.forward`. This preserves the cache write for
future tokens, but when the compressed history already fits inside
`topk_tokens`, it returns the ascending candidate list without `wq_b`,
`weights_proj`, DeepGEMM-prep, or reference-prep work. The behavior is covered by
`test_deepseek_v4_indexer_decode_all_candidate_shortcut_skips_q_prepare` plus
the existing cache-read and cache-read-required tests. A matched `1024x128 c1`
benchmark at
`<remote-workspace>/tokenspeed_bench_sm12x_indexer_allcandidate_20260509193518`
improved native-insert throughput from `6.98` output tok/s and mean TPOT
`132.12 ms` to `7.77` output tok/s and mean TPOT `117.51 ms`. The follow-up
profile at
`<remote-workspace>/tokenspeed_profile_sm12x_indexer_allcandidate_20260509193651`
exported `/tmp/20260509-193725-TP-{0,1}.trace.json.gz`: decode Q preparation is
gone, leaving only 21 prefill-side reference prepare calls; `run_indexer` dropped
to about `0.72 s` per TP rank.

The first native SM12x mHC pre kernel is now wired behind the DeepSeek V4 mHC
gate for decode-sized CUDA inputs (`<=16` tokens). It matches the PyTorch
reference on RTX Pro 6000 and keeps CPU/large-prefill inputs on the reference
path without poisoning the SM12x native availability flag. The kernel is a
single-CTA compatibility implementation, not the final #324-style shape, but it
removes the largest mHC reference block from decode. The CUDA build path also
now preserves no-suffix SM12x arch overrides (`12.0`, `120`, `12.1`, `121`) so
the workstation can build real `sm_120` cubins; explicit suffixes like `120f`
are still accepted when needed.

A matched `1024x128 c1` token-id benchmark at
`<remote-workspace>/tokenspeed_bench_sm12x_mhc_pre_20260509195624` improved
the all-candidate/indexer baseline from `7.77` output tok/s and mean TPOT
`117.51 ms` to `11.24` output tok/s and mean TPOT `78.03 ms`. A short profile
at `<remote-workspace>/tokenspeed_profile_sm12x_mhc_pre_20260509195811`
exported `/tmp/20260509-195817-TP-{0,1}.trace.json.gz`: `sm12x_mhc_pre_kernel`
is about `193 ms` across `1376` calls per TP rank, while the larger remaining
hotspots are reference sparse MLA decode (`deepseek_v4_sparse_mla_reference`
about `800 ms` plus chunk accumulation/writeback), current SM12x MoE gate/down
warp kernels, and launch/type-conversion overhead.

The existing CUDA FP8-cache sparse MLA path is now wired as the SM12x default
for decode, with online softmax enabled by default on SM12x and env overrides
kept for explicit disable/opt-in on other platforms. The wrapper now accepts the
runtime cache view shape (`[blocks, block, 1, row_bytes]`) as well as the older
flat `[blocks, bytes]` test shape. Unit coverage includes runtime-view
correctness against the workspace reference and the SM12x default/override
selection contract.

With no sparse-MLA env vars set, the matched `1024x128 c1` benchmark at
`<remote-workspace>/tokenspeed_bench_sm12x_sparse_mla_default_20260509200827`
improved the native-mHC checkpoint from `11.24` output tok/s and mean TPOT
`78.03 ms` to `14.90` output tok/s and mean TPOT `56.18 ms`. The explicit-env
run at
`<remote-workspace>/tokenspeed_bench_sm12x_sparse_mla_cache_20260509200420`
matched this at `14.89` output tok/s and mean TPOT `56.22 ms`, confirming the
new default gate. The short profile at
`<remote-workspace>/tokenspeed_profile_sm12x_sparse_mla_cache_20260509200535`
exported `/tmp/20260509-200540-TP-{0,1}.trace.json.gz`: decode
`sparse_mla_fp8_cache_online_softmax_kernel` is about `116 ms` across 688 calls
per TP rank, replacing the prior decode reference sparse MLA hotspot. The
remaining sparse MLA reference time in that profile is prefill-only (`43`
calls).

The SM12x mHC post native decode path is now wired beside mHC pre. It matches
the PyTorch lane orientation reference on RTX Pro 6000 and keeps non-SM12x,
CPU, and large-prefill inputs on the reference path. The matched `1024x128 c1`
benchmark at
`<remote-workspace>/tokenspeed_bench_sm12x_mhc_post_20260509201730` measured
`15.05` output tok/s and mean TPOT `55.49 ms`, only a small improvement over the
sparse-MLA default (`14.90` tok/s, `56.18 ms`). The profile at
`<remote-workspace>/tokenspeed_profile_sm12x_mhc_post_20260509201837`
exported `/tmp/20260509-201839-TP-{0,1}.trace.json.gz`; `sm12x_mhc_post_kernel`
is only about `20 ms` across `2752` calls, so mHC post is no longer a meaningful
decode bottleneck.

The DeepSeek V4 FP8 activation quant-dequant simulation now uses an exact
SM12x CUDA kernel for block-128 UE8M0 scaling. It preserves bit-level equality
with the old PyTorch reference and leaves the rejected TRT-LLM helper disabled
on SM12x. A focused `[1,4096]` activation microbench on RTX Pro 6000 measured
about `0.009 ms` for the native kernel versus `0.068 ms` for the PyTorch
reference. The matched `1024x128 c1` benchmark at
`<remote-workspace>/tokenspeed_bench_sm12x_fp8_native_20260509202927`
improved throughput from `15.05` output tok/s and mean TPOT `55.49 ms` to
`16.23` output tok/s and mean TPOT `50.73 ms`. The short profile at
`<remote-workspace>/tokenspeed_profile_sm12x_fp8_native_20260509203017`
exported `/tmp/20260509-203019-TP-{0,1}.trace.json.gz`: the native
`mxfp8_block128_quant_dequant_ue8m0_kernel` is about `8.3 ms` across `7095`
calls per TP rank, and the previous FP8 quantization `pow/log2/ceil/clamp` CUDA
fragmentation is gone.

The mHC pre decode path now uses a split SM12x implementation for DeepSeek V4
decode-sized shapes. The split path breaks `hidden=4096,hc_mult=4` into width
slices, computes per-slice partial reductions, and finalizes the RMS/mix output
with one second-stage CTA per token. The matched `1024x128 c1` benchmark at
`<remote-workspace>/tokenspeed_bench_sm12x_mhc_pre_split_20260509203923`
improved throughput from `16.23` output tok/s and mean TPOT `50.73 ms` to
`18.36` output tok/s and mean TPOT `43.61 ms`. The profile at
`<remote-workspace>/tokenspeed_profile_sm12x_mhc_pre_split_20260509204006`
exported `/tmp/20260509-204008-TP-{0,1}.trace.json.gz`: mHC pre is down to
about `178 ms` per TP rank across partial plus finalize kernels, versus roughly
`385 ms` for the previous single-CTA kernel. Focused SM12x correctness on
`the primary SM120 workstation` reported `118 passed, 1 skipped`.

Build note: generic SM12x CUDA kernels can build with bare `sm_120`, but the
`sm12x_mxfp4_moe` group uses FP8/MXFP4 block-scale MMA instructions and must be
rebuilt with `TOKENSPEED_CUDA_ARCH_LIST=120f` on the RTX Pro 6000 workstation.
Using bare `12.0`/`sm_120` for that group fails ptxas feature checks for
`kind::mxf8f6f4` / `block_scale`.

The direct-warp SM12x MoE backend now has a DeepSeek V4 Flash decode-shape
specialization for `hidden=4096`, `moe_intermediate=2048`, `top_k=6`,
`w13_packed_k=2048`, and `w2_packed_k=1024`. It keeps the current two-stage
warp schedule but removes generic loop and shape overhead from the hot
decode path. Focused SM120 correctness for the real DeepSeek V4 shape passes
against the reference implementation, including varied expert routing and EP
masking.

The fixed-shape MoE microbench improved from about `0.141 ms` to `0.126 ms` per
call with an unchanged checksum. Matched graph bs=2 serving runs measured
`19.21` output tok/s and mean TPOT `42.23 ms` for `1024x128 c1` at
`<remote-workspace>/tokenspeed_bench_ds4shape_moe_20260509223735`, and
`22.07` output tok/s and mean TPOT `44.08 ms` for `1024x1024 c1` at
`<remote-workspace>/tokenspeed_bench_ds4shape_moe_1024x1024_20260509223901`.
The previous graph bs=2 baseline was `18.42` and `21.61` output tok/s
respectively.

The profile at
`/tmp/tokenspeed_profile_ds4shape_20260509224219/20260509-224304-TP-{0,1}.trace.json.gz`
shows gate/down warp totals dropping from roughly `828/489 ms` to
`739/452 ms` per TP rank over the profiled window. This is useful but still
small relative to the target; MoE remains the largest bottleneck, followed by
sparse MLA, NCCL all-reduce, mHC pre, and an unchanged cublas float GEMV
family around `203 ms`. The next route should therefore be a larger
persistent or padding-aware MoE rewrite, not more local scalar dequant tweaks.

The split SM12x mHC pre path now has a DeepSeek V4 decode-shape parallel-apply
stage. The finalize kernel computes and stores the four pre-mix coefficients in
the existing partial scratch buffer, and a separate apply kernel writes the
4096-wide `layer_input` in parallel. This keeps the correctness properties of
the split reduction while avoiding a single CTA doing all lane mixing.

Focused SM12x regression on `the primary SM120 workstation` reported `47 passed`. The isolated
`hidden=4096,hc_mult=4` mHC pre microbench measured `0.024192 ms/call`.
Decode-only profiling at
`/tmp/tokenspeed_profile_stage_mhc_parallel_apply_20260509225638/stage-mhc-parallel-20260509225638-TP-{0,1}-DECODE.trace.json.gz`
shows mHC pre total moving from about `178.5 ms` to about `171.2 ms` over the
32-step window. End-to-end graph bs=2 serving measured `19.29` output tok/s and
mean TPOT `42.02 ms` for `1024x128 c1` at
`<remote-workspace>/tokenspeed_bench_mhc_parallel_apply_20260509225504`, and
`22.17` output tok/s and mean TPOT `43.86 ms` for `1024x1024 c1` at
`<remote-workspace>/tokenspeed_bench_mhc_parallel_apply_1024x1024_20260509225838`.

The current decode-only profile after these two retained steps is roughly:
sparse MLA `232-234 ms`, MoE gate/down `226-234 ms`, compressor/indexer float
GEMV family `203 ms`, mHC pre `171-172 ms`, NCCL all-reduce `49-52 ms`, over
32 decode steps per TP rank. That means the next meaningful jump needs a
larger dataflow change around sparse MLA, MoE, or the compressor/indexer
linear/cache path; isolated scalar or launch-level tweaks are now unlikely to
move more than a few tenths of a millisecond per token.

After rejecting CuTile, the first SM12x/CUTLASS-direction MoE feasibility check
validated a weight-major FP4xFP8 MMA tile on RTX Pro 6000. The new
`sm12x_mxfp4_mxfp8_mma_tile` probe uses SM120 block-scaled
`e2m1.e4m3` MMA with FP4 weights as operand A and FP8 route activations as
operand B. It is not wired into runtime dispatch; it exists to prove the core
orientation needed for a decode MoE design that treats output rows as MMA `M`
and local routes as MMA `N`, avoiding the previous grouped-GEMM failure mode
where a handful of real decode routes were padded to many fake `M` rows per
expert. Focused SM120 validation reported `31 passed` in
`tokenspeed-kernel/test/ops/test_sm12x_mxfp4.py`, and a tile launch microbench
measured about `4.78 us`, matching the existing FP8xMXFP4 tile launch cost.

The follow-up MoE cleanup keeps the deployable direct `warp`/`scalar` path but
switches the new tensorcore foundation to the route-major orientation needed by
future decode MoE. `sm12x_mxfp4_mxfp8_dense` computes checkpoint-layout FP4
weight rows against FP8 route activations with SM120 inline block-scale MMA and
returns standard `[routes, output_channels]` output. The old grouped dense,
route-expansion, grouped SwiGLU/requant, and grouped W2/reduce APIs/tests were
removed because they only supported the rejected M-grouped padding design.

The attention output-projection island (`T1-γ`) is landed end-to-end. CUDA-native
`sm12x_deepseek_v4_inv_rope_fp8_quant_kernel` (REG:24) and
`deepseek_v4_grouped_fp8_einsum_kernel` (REG:39) replaced the previous
Triton pair via `_project_attention_output` dispatch (commits `486ccc1`,
`7eb2845`, `aff5c53`, `0831acd`, `2b4fbf9`, `84773aa`). The CUDA-vs-reference
correctness gate is the only test net (Triton siblings deleted), and the
microbench harness stays as a regression watchdog.

The graph-capture island (`T1-α`) is now also landed for the decode forward
pass. Two CPU<->CUDA sync sites added by the PoC dispatch were removed:
`read_deepseek_v4_indexer_fp8_cache` was rewritten to a single vectorized
gather (commit `aa88a47`), and `_prefill_workspace` picks `max_gather_len` /
`compressed_base` from config-derived upper bounds during capture so the
gather/combine kernels see a stable workspace shape (commit `57534ba`). The
SM12x sparse-MLA CUDA fast path's default `max_topk=1024` gate also turned out
to be tighter than DSv4-Flash's CSA layer width (`swa_w=128` + `extra_w=8192`
= `8320`), which made every decode silently fall through to the BF16 reference
path; the default gate is now removed and the env knob retained as an explicit
kill switch (commit `d3212ac`). With these in place, `--max-cudagraph-capture-size
2 --disable-cuda-graph-padding --disable-prefill-graph` captures cleanly on the
SM120 workstation for `bs=1` and `bs=2`, and a `1024x1024 c1` smoke bench at
`<remote-workspace>/tokenspeed_t1_gamma_bench_20260513_033719` reports
`17.76` output tok/s and mean TPOT `52.89` ms. The matched eager-mode bench at
`<remote-workspace>/tokenspeed_t1_gamma_bench_20260513_032336` reports only
`6.94` tok/s / `140.76` ms with the same fast path engaged, confirming that
the win is from eliding per-layer kernel launch overhead across the captured
graph rather than from the kernel itself. The pre-T1-α eager number with the
gate active and the slow reference path engaged was `14.54` tok/s. The 100
tok/s vLLM-revenge target stays well out of reach until the MoE island lands.

The MoE island (`T3-α`) tensorcore prototype is in place but does not yet
deliver a net end-to-end win:

* The new pieces are landed and individually correct -- a tensorcore W13
  GEMM (commit `a4b813e`), a tensorcore W2 GEMM plus deterministic
  weighted-reduce (commit `069dc17`), a Python orchestrator
  `sm12x_mxfp4_moe_forward_tensorcore` that chains
  quantize -> W13 -> SwiGLU+quantize -> W2 -> reduce
  (commit `d7fc65e`), and a `TOKENSPEED_SM12X_MXFP4_MOE_IMPL=persistent`
  dispatch switch (commit `97e63c9`). All correctness tests pass
  (CUDA-vs-reference + CUDA-vs-warp tolerances).
* Isolated microbenches confirm the tensorcore path is faster: the warp
  full forward is `0.430` ms / call (bs=2) and `0.232` ms (bs=1) at the
  DSv4-Flash decode shape; the tensorcore orchestrator is `0.339` ms
  (bs=2) and `0.194` ms (bs=1) -- a `1.20` - `1.27`x speedup.
* End-to-end, the persistent path regresses by `6%`: the warp `1024x1024 c1`
  graph-mode bench reports `17.73` tok/s / `53.00` ms TPOT pre-rebase
  (`17.13` / `52.82` after the upstream-main rebase), while the
  persistent variant reports `16.65` / `57.84` ms pre-rebase and
  `16.10` / `57.80` ms post-rebase. The microbench predicts a
  `~1` ms / step saving for 30 MoE layers; the actual delta is `+5` ms.
* Diagnostic loop:
    1. Isolated 30-layer MoE-only CUDA-graph capture+replay benches show
       the tensorcore path is **faster** by ~`1.18x` (5.92 ms vs 6.96 ms),
       so the kernels themselves are not the regression source.
    2. Cumulative bench delta tracks back entirely to the **non-MoE**
       layer kernels (attention, indexer, etc.) -- they get ~`5` ms
       slower per step when the persistent backend is selected.
    3. `nsys` profiling of the live server proved impractical: server
       boot under `nsys --trace=cuda` overhead extends from `~30s` to
       `~10` min, and the timing-based `--delay` / `--duration` windows
       never landed during decode in the iterations we tried.
    4. The "allocator pool fragments because the orchestrator allocates
       4 fresh scratch buffers per call" hypothesis was tested by an
       A1 patch (`6245872`) that caches `gate_up` / `per_pair` scratch
       at module level, keyed on shape. The A1 A/B bench measured
       persistent_A1 at `16.12` / `57.76` ms vs unfixed persistent
       `16.10` / `57.80` ms -- effectively identical, **ruling out**
       graph-pool fragmentation as the cause.
* Verdict (`2026-05-13`): keep the tensorcore stack in tree but ship
  with `warp` as the production default. `T3-α` is parked as
  validated infrastructure -- correctness is solid and the kernels
  beat warp in isolation; the orchestrator-vs-warp end-to-end gap
  needs a proper profiler (NVTX-instrumented `nsys` with
  `--capture-range=nvtx` once we add a `decode_step` marker around
  the forward pass) before we know which direction to push, and that
  diagnostic isn't blocking the next throughput lever.
* `w2_bias` and the persistent dispatch are intentionally left in
  place so a follow-up can flip the default once the regression is
  understood.

The NVTX nsys diagnosis landed (`26092c8` adds the
``decode_step@tokenspeed`` marker; capture wrapper at
``/tmp/tk-serve/nsys_decode_step.sh`` on the workstation). Side-by-
side NVTX summary (warp vs persistent, 5 decode_step ranges, 5
process trace files each):

| Range | warp ns/step | persistent ns/step | Δ |
|---|---|---|---|
| ``tokenspeed:decode_step`` | 2247.45 ms | 2306.78 ms | +2.6% |
| ``tokenspeed:v4:ffn_total`` | 186.03 ms | 231.29 ms | **+24.3%** |
| ``tokenspeed:v4:moe_sm12x_experts`` | 51.24 ms | 94.74 ms | **+84.9%** |
| ``tokenspeed:v4:attn_total`` | 1337.49 ms | 1350.58 ms | +1.0% |
| ``NCCL:ncclAllReduce`` | 15.01 ms | 11.25 ms | -25.0% |

The CUDA-kernel summary inside ``moe_sm12x_experts`` pins the
regression on the tensorcore MoE GEMMs themselves (same ``344``
layer-calls on both paths):

| Family | Kernels | Total GPU time |
|---|---|---|
| warp | ``gate_activation_warp_ds4_decode`` (``32.4 ms``) + ``down_warp_ds4_decode`` (``22.9 ms``) | **``55.3 ms``** |
| persistent | ``mxfp4_moe_w13_tensorcore`` (``47.4 ms``) + ``mxfp4_moe_w2_tensorcore`` (``53.3 ms``) + ``swiglu/mxfp8/reduce`` (``1.3 ms``) | **``102.1 ms``** |

Persistent is **``1.82x``** slower per layer at runtime, which directly
matches the isolated-microbench inversion: the microbench (``bs=2``
control shape, ``M~12``) measured tensorcore as ``1.20-1.27x``
faster, but the real decode shape is ``bs=1`` x ``top_k=6`` per
layer i.e. effective ``M=6`` after expert masking. Tensorcore MMA
tiles (``16x16`` minimum on Blackwell SM120 with block-scaled MXFP4
× FP8) waste ``~10/16`` of the M axis at ``M=6``; the warp kernel
processes one route per warp with no tile padding, so it wins on
underutilised M.

This is the same lesson that killed the earlier grouped
tensorcore wrapper (``rejected M-grouped padding design`` -- six
routes padded to ``96`` rows per expert). The persistent path
inherits a smaller version of the same problem at the inverse
extreme.

Side observation worth keeping: ``NCCL:ncclAllReduce`` drops
``-7.75 ms (-25%)`` on the persistent path. The expert-parallel
scheduling apparently overlaps better with the all-reduce window
on the persistent backend. Irrelevant to the regression
diagnosis but interesting for a future overlap study.

* Verdict (`2026-05-13` updated): keep ``warp`` as the SM12x ship
  default. The persistent tensorcore path is structurally wrong
  for the bs=1 decode shape on DSv4-Flash. The fix is not a
  tweak to the persistent orchestrator -- it is a different
  kernel family or a small-``M``/large-``M`` dispatch.

The small-``M`` / large-``M`` dispatch landed in commit ``1aa425c`` as
``TOKENSPEED_SM12X_MXFP4_MOE_IMPL=auto`` (the new SM120 ship default).
Per-call selection: ``M = num_tokens * top_k < threshold (=16)`` -> warp,
``M >= threshold`` -> tensorcore. The ``w2_bias`` contract is honoured
by detecting a strictly-zero ``w2_weight_bias`` once at finalize-weights
time in ``DeepseekV4Sm12xMoEExperts`` and stripping it from subsequent
forwards; non-zero ``w2_bias`` keeps the call on warp regardless of M.
The legacy global-warp behaviour is one env flag away
(``TOKENSPEED_SM12X_MXFP4_MOE_IMPL=warp``).

SM120 bench at ``1024x1024 c1`` (graph-mode, 2 prompts):

| Variant | tok/s | TTFT (ms) | TPOT (ms) |
|---|---|---|---|
| ``baseline_unset`` (pre-flip warp default) | 19.64 | 5703 | 45.39 |
| ``auto_default`` (new SM120 default) | **20.14** | **4477** | **45.31** |
| ``auto_threshold1`` (force tensorcore everywhere) | 18.30 | 4470 | 50.33 |

Decode is unchanged (``TPOT 45.31`` vs ``45.39 ms``; ``auto`` routes
``M=6`` decode to warp, identical to baseline). Prefill is
``-21.5%`` on time-to-first-token (``5703 -> 4477 ms``; ``M=6144``
prefill chunks route to tensorcore where it wins). Tok/s
``+2.5%`` is the residual of the prefill speedup leaving more
decode budget in the fixed bench window. The
``auto_threshold1`` smoke (forcing tensorcore at all ``M``)
reproduces the T3-alpha-traced persistent regression
(``18.30`` tok/s, ``50.33 ms`` TPOT), confirming the dispatch
actually flips and that the regression is structural -- not a
measurement artifact.

Updated cumulative trajectory (no MTP,
``--max-cudagraph-capture-size 2``, ``1024x1024 c1`` graph-mode):

| Checkpoint | tok/s | TPOT | TTFT |
|---|---|---|---|
| Pre-T1-α | 14.54 | -- | -- |
| Post-T1-α + gates (warp baseline, k=1) | 17.13 | 52.84 ms | -- |
| T2-α-1 default-flip (warp + k=8) | 19.66 | 45.35 ms | -- |
| **M-aware MoE default-flip** (auto) | **20.14** | **45.31 ms** | **4477 ms** |

That's ``+38%`` tok/s and a meaningful first-token-latency win
vs the pre-T1-α reference path on the same hardware.

The DeepGEMM FP8-linear SM12x fallback noise from PR #93 was suppressed by
gating the new probe off on SM12x (commit `16ca299`): per-layer
`Unknown SF transformation` warnings are gone, but the post-rebase
warp-baseline regression of `17.73` → `17.12` tok/s did not measurably
move (`+0.05` tok/s after the fix), so the rest of that 3% gap lives
elsewhere in upstream and is deferred.

The sparse-MLA fp8-cache K-axis split (`T2-α-1`) landed in commit
`ea4803f` with the correctness test in `06a3cb0`. Behaviour:

* The single-block ``sparse_mla_fp8_cache_online_softmax_kernel`` now has
  a sibling ``sparse_mla_fp8_cache_online_softmax_k_split_kernel`` plus a
  small ``sparse_mla_fp8_cache_partials_reduce_kernel``. Each chunk
  block processes ``total_len / k_split`` contiguous candidates and
  writes ``(max, denom, acc[head_dim])`` to a scratch ``float32`` buffer
  of size ``num_tokens * num_heads * k_split * (head_dim + 2)``. The
  reduce kernel merges K partials + the optional attention sink using
  the standard online-softmax rescale formula.
* Opt-in is via
  ``TOKENSPEED_DEEPSEEK_V4_CUDA_SPARSE_MLA_CACHE_K_SPLIT=N`` (``1`` to
  ``16``). Default is ``1`` (legacy single-block).
* Microbench at the DSv4-Flash CSA decode shape (head_dim=512, 64 heads,
  swa=128, extra=8192) confirms the K-split path scales nearly linearly
  on Blackwell:

  | bs | k=1 | k=2 | k=4 | k=8 |
  |---|---|---|---|---|
  | 1 | 4.88 ms | 2.45 ms (1.99x) | 1.32 ms (3.71x) | -- |
  | 2 | 4.88 ms | 2.63 ms (1.85x) | 1.45 ms (3.36x) | -- |
* End-to-end ``1024x1024`` graph-mode bench:

  | k_split | tok/s | TPOT |
  |---|---|---|
  | 1 (baseline) | 17.16 | 52.77 ms |
  | 2 | 18.47 | 48.62 ms (+7.6%) |
  | 4 | 19.26 | 46.40 ms (+12.2%) |
  | 8 | **19.65** | **45.35 ms (+14.5%)** |
  | 16 | 19.67 | 45.30 ms |

  Diminishing returns past ``k=8`` -- 8 chunks per (token, head) is the
  recommended setting on Blackwell SM120; the kernel is then within
  reach of the HBM-bandwidth floor on the per-head workload.

The K-split default flipped from ``1`` to ``8`` in commit ``72f005a``
after broader correctness coverage (8 K-split unit cases in commit
``2b5c0de`` + greedy oracle on the SM120 workstation: first ~8 tokens
bit-identical, then expected fp-accumulation-order drift). The
default-flip ``1024x1024 c1`` bench:

| Variant | tok/s | TPOT |
|---|---|---|
| ``default_unset`` (new SM120 default = ``k=8``) | **19.66** | **45.35 ms** |
| ``killswitch_k1`` (env ``=1``, legacy single-block) | 17.13 | 52.84 ms |
| ``explicit_k8`` (env ``=8``) | 19.60 | 45.46 ms |

The new default matches explicit ``=8`` within noise (``19.66`` vs
``19.60`` tok/s), and the env ``=1`` kill switch reverts to the
pre-T2-α-1 single-block baseline (``17.13`` matches the
``17.16`` measured before T2-α-1 landed). Net win at the new
default vs the kill switch is ``+14.8%`` tok/s / ``-14.2%`` mean
TPOT.

Cumulative trajectory (no MTP, ``--max-cudagraph-capture-size 2``,
``1024x1024 c1`` graph-mode):

| Checkpoint | tok/s | TPOT |
|---|---|---|
| Pre-T1-α (reference-heavy attn + gate miscalibration) | 14.54 | -- |
| Post-T1-α + gates fixed (warp baseline, k=1) | 17.13 | 52.84 ms |
| T2-α-1 default-flip (warp + k=8) | **19.66** | **45.35 ms** |

That is a ``+35%`` jump from the pre-T1-α reference path on the
same hardware; the next milestone is the warp-vs-persistent ``T3-α``
gap and the T2-α-2 per-kind K-split tuning.

The T2-α-2 per-kind K-split tuning was measured and rejected. Layer
mix on DSv4-Flash: ``swa`` 3/44 (``compress_ratio=0``), ``hca``
20/44 (``compress_ratio=128``), ``csa`` 21/44 (``compress_ratio=4``).
The theory was that short-``total_len`` layers (``swa`` ~``128``,
``hca`` ~``136`` at 1024 ctx) would prefer ``k=1-2`` because the
kernel's per-chunk arithmetic would underutilise the threads. The
SM120 bench (``1024x1024 c1`` graph-mode, commit ``a237607`` ->
``3ce60ee``) disproved that:

| Variant | tok/s | TPOT | vs ``global_k8`` |
|---|---|---|---|
| ``global_k8`` (all kinds ``k=8``) | **19.69** | **45.25 ms** | -- |
| ``per_kind_default`` (``swa=1,hca=2,csa=8``) | 19.31 | 46.23 | -1.9% |
| ``hca_k1`` (``swa=1,hca=1,csa=8``) | 19.01 | 47.05 | -3.5% |
| ``hca_k2`` (``swa=1,hca=2,csa=8``) | 19.41 | 45.97 | -1.4% |
| ``hca_k4`` (``swa=1,hca=4,csa=8``) | 19.50 | 45.74 | -1.0% |
| ``killswitch_k1`` (all kinds ``k=1``) | 17.10 | 52.93 | -13.2% |

The kernel is parallelism-bound on the 140-SM SM120 at the bs=1
decode shape, not arithmetic-bound: the chunk-kernel grid is ``num_tokens
* num_heads * k_split``, so at ``bs=1 * 64`` heads we get ``512``
blocks at ``k=8`` vs ``128`` at ``k=2``, and the extra block-level
parallelism wins even when each chunk only walks ~``17``
candidates. The marginal cost of higher ``k_split`` (``k_split``-x
partials memory + an O(``k_split``) reduce step) is negligible
versus the parallelism win.

Verdict: revert the per-kind defaults to all ``k=8`` so end-to-end
matches the T2-α-1 ship default, but keep the per-kind dispatch
+ env knobs (``..._K_SPLIT``, ``..._K_SPLIT_SWA``,
``..._K_SPLIT_HCA``, ``..._K_SPLIT_CSA``) as benchmarking levers
for future per-kind kernel work and for alternative bench shapes.
Path B (a per-kind specialised kernel that drops the
``use_extra`` branch for SWA-only) was the T2-α-2 fall-back plan;
given this negative result on Path A the upside is bounded by
branch-elimination on 7% of layers, so it stays parked behind
T3-α.

## Next Step

T2-α-1 K-axis split is on by default; T2-α-2 per-kind tuning was
measured and rejected; T3-α nsys profile captured + the regression
root-caused (persistent tensorcore MoE under-utilises the MMA
tile at the bs=1 decode shape ``M=6``); M-aware MoE dispatch
(``auto``) landed and shipped as the new default (commit ``1aa425c``).
Remaining levers, in priority order:

1. **``T3-α`` W2 bias fuse**. The tensorcore MoE path currently
   silently drops ``w2_bias``; the M-aware dispatch papers over
   this by routing ``w2_bias != None`` calls to warp regardless
   of M, and by stripping a strictly-zero ``w2_bias`` at finalize
   so the persistent kernel can be selected for the DSv4-Flash
   checkpoint. Models with a real non-zero ``w2_bias`` would
   currently miss the large-M tensorcore path -- the W2 bias fuse
   would broaden ``auto``'s reach to those checkpoints. Lower
   priority on DSv4-Flash specifically since the bias is zero.
2. **Per-kind specialised sparse-MLA kernel** (the ``T2-α-2`` Path
   B). Only revisit if a later trace shows the K-split kernel
   itself is the next dominant cost on a per-step basis. Current
   nsys decomposition keeps it well behind MoE.
3. **Sparse-MLA + indexer overlap study**. The persistent-path
   trace showed ``NCCL:ncclAllReduce`` -25% on the same workload --
   evidence that expert-parallel scheduling can shape the all-
   reduce overlap window. A future overlap-engineering pass could
   pursue that lever, but it's behind every other item here.
4. **Auto threshold tuning for batched decode shapes**. The current
   ``M >= 16`` threshold was chosen at the kernel's MMA tile width
   and validated at the bench's bs=1 decode shape (M=6 -> warp).
   For ``--max-num-seqs >= 3`` decode (M = bs * top_k >= 18) the
   dispatch would route to tensorcore; the microbench at M~12 says
   tensorcore wins by ``1.20-1.27x`` so this should be a small win,
   but it's untested at runtime. A sweep at higher batch sizes
   could refine the threshold.

The CuTile route stays rejected for now; do not add CuTile dependencies or
switches.
