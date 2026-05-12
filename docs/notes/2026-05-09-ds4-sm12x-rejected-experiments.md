# DeepSeek V4 SM12x Rejected Experiments

This note records SM12x DeepSeek V4 experiments that were correct enough to
evaluate but should not remain as deployable runtime switches. Keep these as
historical evidence and avoid reintroducing them without a new hypothesis and a
fresh benchmark.

## Rebase Cleanup

- Date: 2026-05-09.
- Context: after rebasing onto latest `upstream/main`, upstream's official
  DeepSeek V4 model/cache/op helpers became the base implementation.
- Code retained: SM12x platform helpers, SM12x MoE backend/kernel modules, the
  SM12x-only sparse MLA fallback in the DeepSeek V4 attention backend, and the
  SM12x-only PyTorch Hadamard fallback for CSA indexer prep.
- Code removed/dropped during conflict resolution: old standalone
  `deepseek_v4_profile.py` profiling helper and its tests, plus pre-rebase
  DeepSeek V4 model/op rewrites that were superseded by upstream's official
  implementation.
- Decision: future work should add SM12x behavior through narrow SM12x gates or
  `tokenspeed-kernel` SM12x op modules, not by replacing upstream's generic
  DeepSeek V4 path.
- Verification note: the Hadamard fallback stays because
  `fast_hadamard_transform` currently has no SM120 kernel image on the test
  workstation.

## CuTile Python Route

- Date: 2026-05-10.
- Attempt considered: use NVIDIA CuTile / `cuda.tile` as the primary SM12x
  kernel authoring path.
- Evidence: the SM120 workstation has `cuda.tile`, but the API and ecosystem
  are less mature than CUTLASS/CuTe DSL for this codebase. TokenSpeed already
  depends on `nvidia-cutlass-dsl`, and `tokenspeed-mla` has a working CuTe DSL
  packaging and launch pattern.
- Decision: do not build the SM12x DeepSeek V4 route on CuTile. Prefer
  CUTLASS/CuTe DSL for NVIDIA-owned long-term maintainability, and keep CuTile
  out of the implementation plan unless NVIDIA's stack converges later.

## FlashMLA SM12x Sparse Decode

- Switches removed:
  `TOKENSPEED_DEEPSEEK_V4_FLASHMLA_SM12X` and
  `TOKENSPEED_DEEPSEEK_V4_FLASHMLA_DECODE_SM12X`.
- Evidence: the packaged `flash_mla_cuda.sparse_decode_fwd` rejected SM120 with
  `Unsupported architecture for sparse decode fwd`.
- Decision: keep disabled until the FlashMLA package ships an SM12x sparse
  decode build.

## Upstream Fast mHC On SM12x

- Date: 2026-05-09.
- Attempt: use upstream's TileLang + DeepGEMM `tf32_hc_prenorm_gemm` fast mHC
  path on the SM120 workstation.
- Evidence: the remote environment has `tilelang` and a `deep_gemm` package
  exposing both `tf32_hc_prenorm_gemm` and `fp8_fp4_mega_moe`, but the first
  SM120 fast-mHC call fails with `Unsupported architecture` from DeepGEMM's
  hyperconnection API and then falls back to the PyTorch reference.
- Decision: gate fast mHC off by default on SM12x so production requests do not
  pay a known exception path. Revisit only after the SM12x DeepGEMM path is
  installed and benchmarked as faster than the PyTorch fallback.

## Upstream TRT-LLM FP8 Activation Quant On SM12x

- Date: 2026-05-09.
- Attempt: use upstream's TRT-LLM `per_token_group_quant_8bit` helper inside the
  DeepSeek V4 reference FP8 activation quantization path.
- Evidence: full DeepSeek V4 config tests on SM120 reached a `no kernel image`
  / bad-scale-shape failure in the helper before the PyTorch reference fallback.
- Decision: gate this third-party helper off by default on SM12x. The local
  PyTorch reference remains the correctness path until a native SM12x FP8
  quantizer is wired through `tokenspeed-kernel`.

## Upstream DeepGEMM FP4 Indexer On SM12x

- Date: 2026-05-09.
- Attempt: use upstream's DeepGEMM FP8xFP4 MQA logits helpers for DeepSeek V4
  CSA indexer top-k with the FP4 indexer cache enabled.
- Evidence: a real eager DeepSeek V4 Flash SM120 serve smoke loaded all 46
  checkpoint shards and initialized the KV pool, but the first decode request
  failed in `deep_gemm.get_paged_mqa_logits_metadata()` with
  `Unsupported architecture`.
- Decision: gate DeepGEMM FP4 indexer logits off by default on SM12x. The
  existing reference indexer path remains the correctness route until a native
  SM12x indexer logits kernel is available.

## Upstream TRT-LLM Fast Top-K On SM12x

- Date: 2026-05-09.
- Attempt: keep upstream's `tokenspeed_kernel.thirdparty.trtllm.fast_topk_v2`
  on the DeepSeek V4 indexer decode fallback when `topk_tokens=512`.
- Evidence: after gating fast mHC, TRT-LLM FP8 activation quant, DeepGEMM FP4
  indexer logits, and the fused QNorm/RoPE/SWA insert op, a real SM120 eager
  serve could generate one token but crashed on the next decode step with
  `CUDA error: no kernel image is available for execution on the device`.
  Synchronization probes narrowed the failure to the CSA indexer path after
  reference top-k. Gating `fast_topk_v2` off and using `torch.topk` produced a
  clean 8-token completion in
  `<remote-workspace>/tokenspeed_smoke_model_level_sm12x_clean_decode_20260509173619`.
- Decision: gate `fast_topk_v2` off by default on SM12x. Replace it later with
  a TokenSpeed-owned SM12x top-k kernel; do not keep a runtime switch for the
  unsupported TRT-LLM path.

## Sparse MLA Fast Exp

- Rejected path: replacing precise `expf` with `__expf` in the online softmax
  sparse MLA FP8 cache kernel.
- Evidence: focused tests and B300 no-MTP oracle passed, but
  `<remote-workspace>/tokenspeed_sparse_online_fast_exp_20260509034359`
  regressed `1024x1024 c2` to `21.18` output tok/s from the online-softmax
  baseline's `22.30` output tok/s.
- Decision: precise `expf` remains the active implementation.

## Old MXFP4 MoE Deploy Switches

- Switches removed:
  `TOKENSPEED_SM12X_MXFP4_MOE_IMPL=tensorcore` and
  `TOKENSPEED_SM12X_MXFP4_MOE_IMPL=tile4`.
- Evidence:
  `<remote-workspace>/tokenspeed_tensorcore_moe_online_20260509034939`
  passed the oracle threshold but measured only `19.05` output tok/s with mean
  TPOT `103.33 ms`; `<remote-workspace>/tokenspeed_tile4_moe_online_20260509035308`
  measured `21.35` output tok/s and had a weaker long-prefill oracle trajectory.
- Decision: keep the default warp MoE route. Do not revive these switch names.

## Grouped Tensorcore MoE Wrapper

- Switch removed: `TOKENSPEED_SM12X_MXFP4_MOE_IMPL=grouped_tc`.
- Code removed: the public/deployable
  `sm12x_mxfp4_moe_forward_grouped_tc` wrapper and exports.
- Follow-up cleanup on 2026-05-10 removed the lower-level grouped
  MXFP8xMXFP4 primitives, route expansion helpers, grouped SwiGLU/requant, and
  grouped W2/reduce public surfaces and tests as well. They encoded the same
  padding-heavy route assumption and should not remain as a supported API.
- Evidence:
  `<remote-workspace>/tokenspeed_grouped_tc_online_20260509045145` passed the
  B300 no-MTP oracle but regressed `1024x1024 c2` to `19.17` output tok/s with
  mean TPOT `103.32 ms`. The N64 reducer run
  `<remote-workspace>/tokenspeed_grouped_tc_n64_online_20260509050530`
  improved that to `20.02` output tok/s with mean TPOT `99.01 ms`, still below
  the warp route with online sparse MLA.
- Root cause: a focused DeepSeek V4 Flash microbench showed full warp MoE around
  `0.27 ms` and full grouped tensorcore MoE around `0.94 ms` for `tokens=1`,
  `topk=6`, `hidden=4096`, and `moe_intermediate=2048`. The grouped route
  spends about `0.79 ms` in W13/SwiGLU/requant because six real decode routes
  are padded to 96 rows.
- Decision: do not expose this wrapper as a runtime option. Future MoE work
  should first eliminate decode padding waste or use a persistent schedule that
  amortizes route padding across enough work.

## Direct Warp MXFP4 Scale Broadcast

- Date: 2026-05-09.
- Attempt: reduce direct-warp MXFP4 MoE scalar overhead by loading each UE8M0
  block scale once per warp/k-block and broadcasting it with `__shfl_sync`,
  instead of letting every lane call the generic `load_mxfp4_value` helper.
- Evidence: focused DeepSeek V4 Flash microbench on `the primary SM120 workstation` for
  `tokens=1`, `topk=6`, `hidden=4096`, and `moe_intermediate=2048` measured
  about `0.307 ms` after the change, worse than the previously measured direct
  warp route around `0.27 ms`.
- Decision: revert the kernel change and keep only the stronger varied-scale
  correctness coverage. The extra shuffle/branch structure is not a useful
  MoE path; revisit scale handling only inside a larger schedule rewrite.

## Decode Route-Parallel Down Projection

- Date: 2026-05-09.
- Attempt: for decode-sized direct-warp MoE, replace the down projection's
  one-warp-per-output-element kernel with a one-block-per-output-element kernel
  where up to eight warps compute top-k routes in parallel and reduce in shared
  memory.
- Evidence: the isolated `tokens=1`, `topk=6`, `hidden=4096`,
  `moe_intermediate=2048` microbench improved from restored direct warp
  `0.256 ms` to `0.219 ms`, and MoE correctness stayed green (`28 passed`).
  However the full DeepSeek V4 Flash `1024x128 c1` benchmark regressed from the
  split-mHC baseline `18.36` output tok/s to `18.02` and `18.04` output tok/s in
  `<remote-workspace>/tokenspeed_bench_sm12x_route_parallel_down_20260509205211`
  and
  `<remote-workspace>/tokenspeed_bench_sm12x_route_parallel_down_repeat_20260509205243`.
- Decision: revert the deployable kernel path. The extra block scheduling
  overhead does not pay off in the full model. Revisit route-level parallelism
  only if it is fused with a larger persistent MoE schedule.

## Compact Route-Table Gate Activation

- Date: 2026-05-09.
- Attempt: build a compact local-route table for decode MoE and run gate
  activation only over local routes, while keeping the validated direct-warp
  down projection. The goal was to remove non-local top-k empty gate warps
  without reintroducing grouped tensorcore route padding.
- Evidence: focused SM120 microbench on `the primary SM120 workstation` for `tokens=1`,
  `topk=6`, `hidden=4096`, `moe_intermediate=2048`, `ep_size=2`, and three
  local routes had exact output agreement (`max_diff=0.0`) but measured
  `direct_warp mean_ms=0.1512` versus `routed_gate mean_ms=0.1528`.
- Decision: remove the deployable routed-gate entrypoint and route-buffer
  runtime wiring. The extra route-table build kernel costs more than the empty
  non-local gate warps save. Revisit compact route tables only inside a larger
  persistent/fused MoE schedule where route metadata is reused across more work.

## Direct Warp UE8M0 Bit Decode

- Date: 2026-05-09.
- Attempt: replace the direct-warp MXFP4 dequant scale factor
  `exp2f(encoded_scale - 127)` with a bit-level UE8M0 power-of-two decode inside
  `load_mxfp4_value`.
- Evidence: focused SM120 microbench on `the primary SM120 workstation` for `tokens=1`, `topk=6`,
  `hidden=4096`, `moe_intermediate=2048`, `ep_size=2`, and three local routes
  regressed from `0.1411 ms` to `0.1481 ms`, with identical checksum.
- Decision: revert the kernel change. Keep the UE8M0 minimum subnormal
  correctness test as guard coverage, but do not replace `exp2f` in the current
  direct-warp schedule.

## Direct Block-FP8 GEMM For DeepSeek V4 FP8 Linear

- Date: 2026-05-09.
- Attempt: replace selected DeepSeek V4 decode FP8 linear paths that currently
  dequantize FP8 weights and use FP32 cublas GEMV with the existing
  `tokenspeed_kernel.mm(..., quant="mxfp8")` block-FP8 Triton path, using the
  native SM12x block-128 activation quantizer.
- Evidence: focused SM120 microbench for decode-sized `M=1` shapes showed the
  existing FP32 reference path is still faster for the compressor/indexer-sized
  matrices: `K=4096,N=1024` was `0.0150 ms` reference versus `0.0272 ms`
  block-FP8; `K=4096,N=2048` was `0.0149 ms` reference versus `0.0268 ms`
  block-FP8. The larger `K=4096,N=8192` shape was slightly faster with
  block-FP8 (`0.0482 ms` versus `0.0582 ms`) but had larger numerical drift and
  is not the dominant compressor/indexer shape in the decode profile.
- Decision: do not replace the current FP8 linear path with the generic
  block-FP8 Triton kernel. Revisit only with an SM12x-tuned `M=1` FP8 GEMV/GEMM
  kernel or a fused compressor/cache-insert design that removes the standalone
  GEMV family altogether.

## SM12x TRT-LLM One-Shot All-Reduce Auto-Configuration

- Date: 2026-05-09.
- Attempt: automatically configure TokenSpeed's existing TRT-LLM Lamport
  one-shot all-reduce backend for SM12x process groups during distributed
  initialization, so decode-sized MoE/attention all-reduces avoid NCCL latency.
- Evidence: a two-rank SM120 microbench for `1x4096` BF16 all-reduce showed
  the backend was correct (`max_diff=0.0`) and faster than NCCL
  (`nccl mean_ms=0.0205`, `trt mean_ms=0.0119`). However full DeepSeek V4
  Flash `1024x128 c1` regressed from the split-mHC baseline `18.36` output
  tok/s to `18.24` and `18.23` output tok/s in
  `<remote-workspace>/tokenspeed_bench_sm12x_trtllm_ar_20260509212424` and
  `<remote-workspace>/tokenspeed_bench_sm12x_trtllm_ar_repeat_20260509212456`.
- Decision: remove the automatic configuration. The one-shot backend may still
  be useful inside a larger fused all-reduce+norm or graph-safe communication
  plan, but standalone replacement does not improve the current eager full
  model path.

## Attention Aux-Stream Input GEMM Overlap

- Switch removed: `TOKENSPEED_DEEPSEEK_V4_AUX_INPUT_GEMMS`.
- Attempt: mirror the vLLM DeepSeek V4 branch by precomputing attention
  compressor and indexer-compressor input GEMMs on auxiliary CUDA streams while
  the main stream runs QKV projection, Q projection, and SWA cache insert.
- Evidence:
  `<remote-workspace>/tokenspeed_aux_input_gemms_oracle_20260509060859`
  passed all five B300 no-MTP oracle cases, but
  `<remote-workspace>/tokenspeed_aux_input_gemms_online_20260509061127`
  regressed `1024x1024 c2` to `22.57` output tok/s with mean TPOT
  `82.92 ms`. A corrected variant that stopped precomputing
  `indexer.weights_proj` on the common full-candidate decode path improved only
  to `22.78` output tok/s with mean TPOT `82.09 ms` in
  `<remote-workspace>/tokenspeed_aux_input_gemms_nowproj_online_20260509061615`.
  The stable warp-slot QNorm path remains faster at `23.56` output tok/s with
  mean TPOT `79.23 ms`.
- Root cause: in TokenSpeed's current eager decode path, the full-candidate
  indexer shortcut already avoids one of vLLM's auxiliary GEMMs, and the
  remaining two GEMMs do not hide enough work to pay for Python-level
  stream/event scheduling and later synchronization.
- Decision: do not keep a runtime switch for layer-local aux streams. Revisit
  overlap only as part of a coarser persistent/layer executor where launch and
  synchronization overhead can be amortized globally.

## Verification Gotcha: Empty Oracle Logprobs

- Symptom: oracle comparison can report `actual_token_count=0` for every case
  while the server logs show successful HTTP 200 completion requests.
- Root cause: TokenSpeed intentionally returns empty sampled-token logprobs
  unless the server was started with `--enable-output-logprobs`.
- Evidence: the first warp-slot SWA insert oracle run without the flag produced
  empty actual token traces. Re-running with the flag in
  `<remote-workspace>/tokenspeed_warpslot_qnorm_logprobs_20260509054446`
  accepted all five B300 no-MTP top-20 oracle cases.
- Decision: treat empty actual token traces as an invalid oracle setup unless
  `--enable-output-logprobs` is present. Do not attribute that failure to kernel
  correctness.

## Directly Enabling Existing Fast mHC On SM120

- Switch kept disabled: `_deepseek_v4_fast_mhc_enabled_for_platform()` still
  returns false on SM12x.
- Attempt: call `tokenspeed.runtime.layers.deepseek_v4_mhc.mhc_pre` and
  `mhc_post` directly on RTX Pro 6000 for small CUDA tensors, bypassing the
  model-level platform gate.
- Evidence: direct smoke on `the primary SM120 workstation` for hidden sizes `128`, `512`, and
  `2048` failed before correctness comparison. The TileLang module imported, but
  DeepGEMM's `tf32_hc_prenorm_gemm` raised
  `RuntimeError: Assertion error (csrc/apis/hyperconnection.hpp:56): Unsupported architecture`.
- Decision: do not merely remove the SM12x gate for fast mHC. The next useful
  path is an SM12x-native mHC pre implementation or a rewrite that avoids the
  unsupported DeepGEMM hyperconnection primitive.

## mHC Pre Fast Math Exp

- Date: 2026-05-09.
- Attempt: replace the SM12x native mHC pre `sigmoid` and Sinkhorn softmax
  exponentials with CUDA `__expf` while leaving the sparse MLA online softmax
  unchanged.
- Evidence: the focused DeepSeek V4 shape correctness test passed, but the
  isolated `tokens=1,hc_mult=4,hidden=4096` microbench on `the primary SM120 workstation` moved
  from the retained split-apply baseline `0.024192 ms/call` to
  `0.024384 ms/call` (`min_ms=0.024362`, `max_ms=0.024391`).
- Decision: keep precise `expf` in mHC pre. This was not a measurable win and
  consumes numerical-risk budget without throughput upside.

## MoE SwiGLU Fast Exp

- Date: 2026-05-09.
- Attempt: replace the native SM12x MXFP4 MoE `apply_swiglu` sigmoid
  exponentials with CUDA `__expf`.
- Evidence: focused MoE correctness on `the primary SM120 workstation` still passed
  (`tokenspeed-kernel/test/ops/test_sm12x_mxfp4.py`: `30 passed`), but the
  isolated DeepSeek V4 decode-shape MoE microbench regressed from the retained
  DS4 direct-warp baseline `0.125853 ms/call` to `0.133723 ms/call`
  (`median_ms=0.133568`, checksum `0.004159`).
- Decision: revert to precise `expf` in `apply_swiglu`. The sigmoid exp is not
  the limiting cost in the current direct-warp schedule; weight loads and dot
  products dominate.

## Sparse MLA Online Softmax Fast Exp

- Date: 2026-05-09.
- Attempt: replace the SM12x sparse MLA FP8-cache online-softmax `expf` calls
  for score rescaling and candidate weights with CUDA `__expf`.
- Evidence: focused sparse MLA correctness passed on `the primary SM120 workstation`, but two
  full-model `1024x128 c1` runs were indistinguishable from the retained FP8
  weight GEMV baseline: `20.20` output tok/s, TPOT `39.67 ms` in
  `<remote-workspace>/tokenspeed_bench_sparse_fast_exp_20260509233349`, then
  `20.17` output tok/s, TPOT `39.74 ms` in
  `<remote-workspace>/tokenspeed_bench_sparse_fast_exp_repeat_20260509233420`.
  The retained precise-exp baseline was `20.18` output tok/s, TPOT `39.73 ms`.
- Decision: keep precise `expf` in sparse MLA. This change only trades
  numerical-risk budget for measurement noise.
