# SM12x MXFP4 Native MoE Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a first-class TokenSpeed MXFP4 MoE backend for SM120/SM121 without reusing the SM100 FlashInfer TRT-LLM path or the high-peak Triton `convert_layout` load path.

**Architecture:** Runtime backend selection owns the SM12x policy and chooses a new `sm12x_mxfp4` backend only on SM120/SM121. Kernel-facing helpers live under `tokenspeed_kernel.ops.moe.sm12x_mxfp4`, with a low-peak canonical packed-weight layout, a reference forward for tests, and a native CUDA smoke kernel that avoids SM100 FlashInfer/TRT-LLM and Triton `convert_layout` load paths.

**Tech Stack:** PyTorch runtime modules, TokenSpeed MoE backend registry, TokenSpeed kernel ops registry boundaries, remote SM120 validation on `the primary SM120 workstation`.

---

### Task 1: Backend Selection

**Files:**
- Modify: `python/tokenspeed/runtime/layers/moe/core/selector.py`
- Modify: `python/tokenspeed/runtime/layers/moe/backends/__init__.py`
- Modify: `python/tokenspeed/runtime/layers/moe/utils.py`
- Test: `test/runtime/layers/test_moe_selector.py`

**Steps:**
1. Write failing tests that SM120/SM121 MXFP4 selects `Mxfp4Sm12xBackend`.
2. Add the new backend name to auto preference and optional forced CLI enum.
3. Register the backend family with `supported_arches={"sm120", "sm121"}`.
4. Re-run the selector tests.

### Task 2: Low-Peak Packed Layout Boundary

**Files:**
- Create: `python/tokenspeed/runtime/layers/moe/backends/mxfp4/sm12x.py`
- Create: `tokenspeed-kernel/python/tokenspeed_kernel/ops/moe/sm12x_mxfp4/__init__.py`
- Test: `test/runtime/layers/test_moe_selector.py`

**Steps:**
1. Write failing tests that post-load processing keeps DeepSeek V4 concatenated weights in place and canonicalizes interleaved weights per expert.
2. Implement metadata attachment without calling Triton `convert_layout`.
3. Keep the packed checkpoint shape as the canonical kernel input: `w13=[E,2I,H/2]`, `w13_scale=[E,2I,H/32]`, `w2=[E,H,I/2]`, `w2_scale=[E,H,I/32]`.

### Task 3: Correctness Oracle

**Files:**
- Create: `tokenspeed-kernel/python/tokenspeed_kernel/ops/moe/sm12x_mxfp4/reference.py`
- Test: `tokenspeed-kernel/test/ops/test_sm12x_mxfp4.py`

**Steps:**
1. Write failing tests for packed E2M1+UE8M0 dequantization and a tiny routed MoE forward.
2. Implement dequant helpers and a reference forward that dequantizes only selected experts.
3. Gate runtime reference forward behind `TOKENSPEED_SM12X_MXFP4_REFERENCE_FORWARD=1`; production forward raises until the native CUDA kernel is added.

### Task 4: Remote Verification

**Files:**
- Modify: `docs/plans/2026-05-07-ds4-sm12x-poc.md`

**Steps:**
1. Sync TokenSpeed to `<remote-workspace>/tokenspeed`.
2. Run focused unit tests in the remote `.venv`.
3. Run a SM120 serve smoke only after backend import/selection and load-path tests pass.
4. Record whether the next blocker is kernel implementation, EP communication, or attention.

**2026-05-07 result:**
- Focused remote tests passed in `<remote-workspace>/tokenspeed/.venv`:
  `test/runtime/layers/test_moe_selector.py` -> `7 passed`; SM12x platform,
  DeepSeek V4 attention reference, and SM12x MXFP4 reference tests -> `12 passed`.
- Serve smoke run
  `<remote-workspace>/tokenspeed_smoke_sm12x_native_20260507230709` used
  `--attn-tp-size 2`, `--enable-expert-parallel`, and
  `--moe-backend sm12x_mxfp4`. It loaded all checkpoint shards and initialized
  the DeepSeek V4 KV pool with MoE `tp=1 ep=2`.
- The run stopped at the intentional
  `SM12x MXFP4 native CUDA MoE kernel is not implemented yet` runtime boundary.
  Attention and EP communication reached the first MoE forward; the next blocker
  is the native SM12x MXFP4 MoE kernel implementation.

**2026-05-07 native CUDA update:**
- Added a first native CUDA MXFP4 MoE smoke kernel under
  `tokenspeed_kernel.thirdparty.cuda.csrc.sm12x_mxfp4_moe`, exported through
  `tokenspeed_kernel.ops.moe.sm12x_mxfp4`. It consumes the canonical checkpoint
  layout directly: `w13=[E,2I,H/2]`, `w13_scale=[E,2I,H/32]`,
  `w2=[E,H,I/2]`, `w2_scale=[E,H,I/32]`.
- The current CUDA path is correctness-oriented: it dequantizes packed MXFP4
  values on the fly, computes gate/up into a temporary float32 activation, and
  runs the down projection with scalar loops. It verifies the native TokenSpeed
  boundary and EP masking, but it is not the intended performance kernel.
- Added SM12x fallback support needed to reach a full DeepSeek V4 request:
  PyTorch FWHT fallback for the DeepSeek V4 Hadamard rotate, SM12x exclusion of
  SM100 FlashInfer/DeepGEMM FP8 blockscale kernels, a Triton FP8 blockscale
  fallback with PyTorch activation quantization, and a 2D `_fp8_linear` path
  that calls `tokenspeed_kernel.mm` without full weight dequantization.
- Serve smoke run
  `<remote-workspace>/tokenspeed_smoke_sm12x_native_cuda_20260507233552` used
  `--attn-tp-size 2`, `--enable-expert-parallel`, and
  `--moe-backend sm12x_mxfp4`. It mapped MoE as `tp=1 ep=2`, loaded all 46
  shards, initialized the DeepSeek V4 KV pool, served `/get_model_info`, and
  completed one `/generate` request with HTTP 200. The recorded harness status
  is `137` because the resident server was manually stopped after the successful
  request.

**2026-05-08 final target update:**
- The final SM12x MXFP4 MoE target is a TokenSpeed-owned SM120f tensor-core
  implementation based on the DeepGEMM PR #324 design direction: native SM120
  block-scaled MMA, FP8 activation by MXFP4/E2M1 weight GEMM, hardware-assisted
  FP4 shared-memory load/unpack, tiled persistent execution, fused activation
  quantization, and route-weighted scatter/reduce.
- The current scalar, warp-reduction, tile4, FlashInfer grouped GEMM, and
  standalone dense-GEMM experiments are intermediate states only. They remain
  valuable as correctness oracles, safety fallbacks, and benchmark baselines,
  but they are not the long-term backend architecture.
- Implementation should progress through explicit, testable tensor-core
  milestones:
  1. Add isolated SM120 MMA/fragment helpers and a tiny FP8xMXFP4 tile test.
  2. Add a dense `FP8 activation x canonical MXFP4 weight -> BF16/FP32` GEMM
     primitive for DS4-shaped dimensions.
  3. Add routed/expert-grouped scheduling over canonical `w13` and `w2`
     checkpoint layout, preserving the current low-peak load path.
  4. Fuse W13 GEMM, SwiGLU, intermediate FP8 quantization, W2 GEMM, and
     route-weighted scatter/reduce into the production MoE path.
  5. Promote the tensor-core path to the default only after harness oracle
     correctness and focused performance gates both beat the warp fallback.
- Do not pivot to NVFP4/B12x semantics for DeepSeek V4 MXFP4 correctness. If a
  layout conversion becomes necessary, make it an explicit opt-in experiment
  with its own correctness delta and memory peak measurement.

**2026-05-08 tensor-core primitive update:**
- Added isolated SM120 `mma.sync.aligned.kind::mxf8f6f4` coverage for a
  `m16n8k32` FP8 activation x MXFP4/E2M1 weight tile. The test covers canonical
  packed FP4 input, hardware `.b4x16_p64` shared-memory load/unpack, and
  non-uniform UE8M0 activation/weight block scales.
- Added `sm12x_mxfp8_mxfp4_dense`, a correctness-first FP32 dense primitive for
  `M%16==0`, `N%8==0`, `K%32==0` shapes. It keeps the canonical weight layout
  `[N,K/2]` plus `[N,K/32]` scales and loops over K blocks with the same SM120
  MMA tile.
- Added `sm12x_mxfp8_mxfp4_grouped_dense`, the first routed/expert-grouped
  primitive over canonical `[E,N,K/2]` weights. The current contract expects
  expanded activation rows to be grouped in 16-row expert tiles; this keeps the
  implementation explicit until the production scheduler is added.
- Added `sm12x_mxfp4_moe_forward_tensorcore`, a wrapper-level fused-MoE
  correctness path that orchestrates expert grouping, 16-row padding, hidden
  FP8 block quantization, W13 grouped dense, SwiGLU, intermediate FP8 block
  quantization, W2 grouped dense, and route-weighted reduction. This is still
  an intermediate state: the scheduling/reduce work is intentionally not fused
  into one CUDA kernel yet.
- Wired `TOKENSPEED_SM12X_MXFP4_MOE_IMPL=tensorcore` into the native runtime
  dispatcher as an explicit opt-in. The default remains `warp` until the
  tensor-core path passes the harness correctness gate and a focused
  performance gate.
- Added tensor-core wrapper bias handling for DeepSeek V4's current
  `with_bias=True` MoE construction: W13 bias is applied before SwiGLU and W2
  bias is applied before route-weighted reduction.
- Remote validation on `the primary SM120 workstation`:
  `tokenspeed-kernel/test/ops/test_sm12x_mxfp4.py` -> `13 passed`.
- Runtime validation on `the primary SM120 workstation`: the remote `.venv` now has the main
  `tokenspeed` package installed editable against `~/Workspace/tokenspeed/python`
  so runtime tests and serve use the workspace source.
  `test/runtime/layers/test_moe_selector.py` -> `8 passed`.
- Tensor-core opt-in serve smoke run dir:
  `<remote-workspace>/tokenspeed_smoke_sm12x_tensorcore_20260508070950`.
  It used `TOKENSPEED_SM12X_MXFP4_MOE_IMPL=tensorcore`, `--attn-tp-size 2`,
  `--enable-expert-parallel`, and `--moe-backend sm12x_mxfp4`; the server
  loaded DeepSeek V4, served `/get_model_info`, completed `/v1/completions`
  with HTTP 200 and logprobs, then exited cleanly after SIGTERM.
- Harness oracle gate against the SM120 no-MTP vLLM baseline passed with
  `TOKENSPEED_SM12X_MXFP4_MOE_IMPL=tensorcore`. Run dir:
  `<remote-workspace>/tokenspeed_oracle_sm12x_tensorcore_short_20260508071504`.
  The four short oracle cases all returned `ok=true`; `code_probe` and
  `translation` were exact token matches, while `raw_intro` and `short_math`
  stopped only at low-margin trajectory forks with matching prompt token ids.
  The long-prefill oracle case also returned `ok=true`, with matching prompt
  token ids and a low-margin fork at step 7.
- Profiled tensor-core long-prefill run dir:
  `<remote-workspace>/tokenspeed_profile_sm12x_tensorcore_long_20260508071819`.
  The long oracle case again returned `ok=true`. Profile aggregation shows
  prefill `moe.routed_experts` at `6036.384 ms` across 86 layer/rank records
  (`70.191 ms` mean, `91.648 ms` max). Including decode, total
  `moe.routed_experts` was `12882.398 ms` across 2666 records. This makes the
  tensor-core path the right next base for fusing scheduler/reduce into CUDA.

**2026-05-08 CUDA route expansion update:**
- Added `sm12x_mxfp4_build_route_expansion`, exported through
  `tokenspeed_kernel.ops.moe.sm12x_mxfp4`. It uses CUDA kernels to count local
  EP routes, group them by local expert, pad each expert to 16 rows, gather
  hidden rows, and emit route metadata: expanded token ids, route weights,
  local expert ids, and an `is_real` mask.
- `sm12x_mxfp4_moe_forward_tensorcore` now uses this CUDA route expansion
  primitive instead of Python `nonzero`/list/`torch.cat` grouping. W13/W2
  tensor-core grouped dense and the final Python `index_add_` reduce are still
  intentionally separate intermediate steps.
- TDD red/green was run on `the primary SM120 workstation`: the new route expansion test first
  failed on the missing public op, then passed after the CUDA and Python
  wrappers were added. Focused remote validation passed:
  `tokenspeed-kernel/test/ops/test_sm12x_mxfp4.py` -> `14 passed`; ruff on the
  changed Python wrapper/export/test files passed.
- Short oracle gate against the SM120 no-MTP vLLM baseline passed after the
  route expansion change. Run dir:
  `<remote-workspace>/tokenspeed_oracle_sm12x_routeexp_20260508073524`.
  All four non-long-prefill cases returned `ok=true`; `code_probe`,
  `raw_intro`, and `translation` were exact token matches, while `short_math`
  stopped at a low-margin fork with matching prompt token ids and
  trajectory top-1 match rate `1.0`.
- The next fused-kernel step is to move route-weighted scatter/reduce into CUDA
  and then begin fusing route expansion, W13, activation quantization, W2, and
  reduce around the SM120 tensor-core grouped dense primitive.

**2026-05-08 CUDA route reduce update:**
- Added `sm12x_mxfp4_reduce_routes`, exported through
  `tokenspeed_kernel.ops.moe.sm12x_mxfp4`. It zeroes an FP32 token output
  buffer and performs route-weighted atomic accumulation from expanded routed
  rows back to token rows, matching the previous Python `index_add_` contract.
- `sm12x_mxfp4_moe_forward_tensorcore` now uses the CUDA reduce primitive for
  the final scatter/reduce step, while preserving FP32 accumulation before the
  final cast back to the hidden-state dtype.
- TDD red/green was run on `the primary SM120 workstation`: the new reduce test first failed on
  the missing public op, then passed after the CUDA and Python wrappers were
  added. Focused remote validation passed:
  `tokenspeed-kernel/test/ops/test_sm12x_mxfp4.py` -> `15 passed`; ruff on the
  changed Python wrapper/export/test files passed.
- Short oracle gate against the SM120 no-MTP vLLM baseline passed after the
  CUDA reduce change. Run dir:
  `<remote-workspace>/tokenspeed_oracle_sm12x_reduce_20260508074757`.
  All four non-long-prefill cases returned `ok=true`; `code_probe` and
  `translation` were exact token matches, while `raw_intro` and `short_math`
  stopped at low-margin forks with matching prompt token ids and trajectory
  top-1 match rate `1.0`.
- Long-prefill oracle gate also passed. Run dir:
  `<remote-workspace>/tokenspeed_oracle_sm12x_reduce_long_20260508075000`.
  `completion_long_prefill_2048_logprobs20` returned `ok=true`, exact token
  match, matching prompt token ids, trajectory top-1 match rate `0.96`, and
  trajectory top-k overlap mean `0.892`.
- The next fused-kernel step is to remove the remaining wrapper-level staging
  around grouped dense: start fusing W13 output, SwiGLU, intermediate FP8
  quantization, W2, and reduce into a single routed SM120 tensor-core kernel.

**2026-05-08 CUDA SwiGLU MXFP8 quantization update:**
- Added `sm12x_mxfp4_swiglu_mxfp8_quantize`, exported through
  `tokenspeed_kernel.ops.moe.sm12x_mxfp4`. It fuses W13 bias application,
  SwiGLU limit/alpha/beta handling, intermediate FP8 e4m3fn quantization, and
  UE8M0 scale emission for each 32-wide activation block.
- `sm12x_mxfp4_moe_forward_tensorcore` now routes W13 grouped-dense output
  through this CUDA primitive instead of wrapper-level PyTorch bias,
  activation, and `_quantize_mxfp8_ue8m0` staging. W2 grouped dense and final
  CUDA route reduce remain separate intermediate kernels.
- TDD red/green was run on `the primary SM120 workstation`: the new public-op test first failed
  on the missing export, then passed after the CUDA and Python wrappers were
  added. The unit test now uses randomized 16-row/64-column routed inputs and
  verifies bit-level equality against the PyTorch FP8/UE8M0 reference.
- Focused remote validation passed after the change:
  `tokenspeed-kernel/test/ops/test_sm12x_mxfp4.py` -> `16 passed`; ruff on the
  changed Python wrapper/export/test files passed.
- Short oracle gate against the SM120 no-MTP vLLM baseline passed. Run dir:
  `<remote-workspace>/tokenspeed_oracle_sm12x_swigluquant_20260508080657`.
  All four non-long-prefill cases returned `ok=true`; `code_probe` and
  `translation` were exact token matches, while `raw_intro` and `short_math`
  stopped at low-margin forks with matching prompt token ids and trajectory
  top-1 match rate `1.0`.
- Long-prefill oracle gate also passed in the same run dir.
  `completion_long_prefill_2048_logprobs20` returned `ok=true`, matching prompt
  token ids, trajectory top-1 match rate `1.0`, and trajectory top-k overlap
  mean `0.906`; the only token fork was classified as low-margin.
- The next fused-kernel step is to pull W2 output bias and route reduce closer
  to the grouped-dense kernels, then replace the wrapper orchestration with a
  single routed SM120 tensor-core MoE kernel.

**2026-05-08 CUDA W2 bias reduce update:**
- Extended `sm12x_mxfp4_reduce_routes` with optional `expanded_experts` and
  `w2_bias` inputs. The CUDA reduce kernel now adds W2 bias per expanded route
  before applying the route weight and atomic-accumulating into token output,
  matching the previous wrapper-level `expanded_out + w2_bias[expert]`
  contract.
- `sm12x_mxfp4_moe_forward_tensorcore` now passes W2 bias into the reduce
  primitive instead of materializing a biased expanded-output tensor in Python.
  The wrapper orchestration is still intentionally staged around W13 grouped
  dense, fused SwiGLU quantization, W2 grouped dense, and CUDA reduce.
- TDD red/green was run on `the primary SM120 workstation`: the new reduce-bias test first failed
  on the missing `expanded_experts` API, then passed after the CUDA and Python
  wrappers were updated.
- Focused remote validation passed after the change:
  `tokenspeed-kernel/test/ops/test_sm12x_mxfp4.py` -> `17 passed`; ruff on the
  changed Python wrapper/export/test files passed.
- Oracle gate against the SM120 no-MTP vLLM baseline passed. Run dir:
  `<remote-workspace>/tokenspeed_oracle_sm12x_w2biasreduce_20260508082046`.
  All four non-long-prefill cases returned `ok=true`; `code_probe` and
  `translation` were exact token matches, while `raw_intro` and `short_math`
  stopped at low-margin forks with matching prompt token ids. The
  `completion_long_prefill_2048_logprobs20` case returned `ok=true` with exact
  token match and matching prompt token ids.
- For remote iteration, keep `the primary SM120 workstation:~/Workspace/tokenspeed` as the
  canonical remote checkout and reuse `~/Workspace/tokenspeed/.venv`. Avoid
  creating per-worktree venvs for this PoC.
- Development-environment update: `tokenspeed-kernel/python/setup.py` now
  supports `TOKENSPEED_KERNEL_SKIP_SATISFIED_BUILD_REQUIREMENTS=1`. When this
  env var is set, `build_native` checks the selected backend requirements
  against the active venv and skips `pip install -r requirements/<backend>.txt`
  only if every applicable requirement is already installed with a compatible
  version. Missing or mismatched requirements still fall back to the normal
  install path.
- Verified remote editable install command:
  `TOKENSPEED_KERNEL_SKIP_SATISFIED_BUILD_REQUIREMENTS=1 TOKENSPEED_KERNEL_BACKEND=cuda TOKENSPEED_CUDA_ARCH_LIST=12.0f MAX_JOBS=8 .venv/bin/python -m pip install -e tokenspeed-kernel/python --no-build-isolation -v`.
  On `~/Workspace/tokenspeed/.venv`, this printed
  `skipping pip install` and `Skipped 15 up-to-date kernel group(s)`, avoiding
  the repeated `flashinfer_jit_cache` download path.
- The next fused-kernel step is to collapse the remaining staged grouped-dense
  orchestration into a routed SM120 tensor-core MoE kernel, starting with a
  kernel-local schedule that owns W13 -> activation quant -> W2 -> reduce for
  one expert group tile.

**2026-05-08 CUDA hidden MXFP8 quantization update:**
- Added `sm12x_mxfp4_mxfp8_quantize`, exported through
  `tokenspeed_kernel.ops.moe.sm12x_mxfp4`. It converts 2-D float32, float16, or
  bfloat16 activation rows to FP8 e4m3fn plus UE8M0 scale, one 32-wide block
  per warp.
- `sm12x_mxfp4_moe_forward_tensorcore` now uses this CUDA primitive for hidden
  activation quantization before W13 grouped dense, replacing the previous
  wrapper-level PyTorch `_quantize_mxfp8_ue8m0(expanded_hidden)` staging.
- TDD red/green was run on `the primary SM120 workstation`: the new quantization test first failed
  on the missing public op, then passed after the CUDA and Python wrappers were
  added. The test verifies bit-level equality against the PyTorch reference for
  float32, float16, and bfloat16 inputs.
- Focused remote validation passed:
  `tokenspeed-kernel/test/ops/test_sm12x_mxfp4.py` plus
  `tokenspeed-kernel/test/test_sm12x_platform.py` -> `31 passed`; ruff on the
  changed setup/wrapper/export/test files passed.
- Oracle gate against the SM120 no-MTP vLLM baseline passed. Run dir:
  `<remote-workspace>/tokenspeed_oracle_sm12x_hiddenquant_20260508083352`.
  All four non-long-prefill cases returned `ok=true`; `code_probe` and
  `translation` were exact token matches, while `raw_intro` and `short_math`
  stopped at low-margin forks with matching prompt token ids. The
  `completion_long_prefill_2048_logprobs20` case returned `ok=true` with exact
  token match and matching prompt token ids.
- The next fused-kernel step should target the W13/W2 grouped-dense boundary:
  either move from the standalone grouped dense calls toward a single
  expert-group tile kernel, or first fuse route gather directly into hidden
  quantization to avoid materializing expanded hidden rows.

**2026-05-08 fused route-gather hidden quantization update:**
- Added `sm12x_mxfp4_build_route_expansion_mxfp8_quantize`, exported through
  `tokenspeed_kernel.ops.moe.sm12x_mxfp4`. It reuses the CUDA route count,
  route fill, and expert padding kernels, then directly gathers each routed
  hidden row into FP8 e4m3fn plus UE8M0 scale without materializing the
  intermediate expanded hidden tensor.
- `sm12x_mxfp4_moe_forward_tensorcore` now uses this fused route-gather +
  hidden-quant primitive before W13 grouped dense. The remaining staged
  operations are W13 grouped dense, fused SwiGLU/intermediate quantization, W2
  grouped dense, and CUDA route reduce.
- Added TDD coverage comparing the fused primitive against the previous
  two-step `sm12x_mxfp4_build_route_expansion` +
  `sm12x_mxfp4_mxfp8_quantize` path for float32, float16, and bfloat16 hidden
  states. The tests verify exact metadata, exact UE8M0 scale, and bit-level FP8
  equality.
- Focused remote validation passed on `the primary SM120 workstation`:
  `test/test_bench_token_ids.py` -> `3 passed`;
  `tokenspeed-kernel/test/ops/test_sm12x_mxfp4.py` plus
  `tokenspeed-kernel/test/test_sm12x_platform.py` -> `34 passed`; ruff on the
  changed Python files passed. The kernel editable install rebuilt only the
  `sm12x_mxfp4_moe` group while reusing `~/Workspace/tokenspeed/.venv`.
- Harness oracle gate against the SM120 no-MTP vLLM baseline passed. Run dir:
  `<remote-workspace>/tokenspeed_oracle_fused_route_quant_logprobs_20260508092922`.
  All five oracle cases returned `ok=true` with matching prompt token ids.
  `code_probe` and `translation` were exact token matches; `raw_intro` and
  `short_math` stopped at known low-margin forks; the long-prefill case
  returned `ok=true`.
- A reusable tokenizer-free serving benchmark path now exists in
  `tokenspeed bench serve --dataset-name token_ids`. It sends integer token ids
  to `/v1/completions`, skips tokenizer initialization, and is documented in
  `docs/guides/benchmarking.md`.
- `1024x1024 c2` benchmark before this fused change:
  `<remote-workspace>/tokenspeed_bench_token_ids_1024x1024_20260508091433`
  reported `9.96` output tok/s and `19.92` total tok/s. After the fused
  route-gather hidden quantization change:
  `<remote-workspace>/tokenspeed_bench_token_ids_1024x1024_fused_20260508093123`
  reported `9.95` output tok/s and `19.90` total tok/s. This confirms the
  change is correctness-neutral but not an end-to-end performance lever by
  itself; decode remains dominated by the staged MoE grouped dense and reduce
  pipeline.
- The next performance step should move past wrapper-level orchestration:
  implement an expert-group tile kernel that owns W13 -> activation quant -> W2
  -> route-weighted reduce for one local expert tile, then compare it against
  the current tensor-core wrapper with the same `token_ids` 1024x1024 benchmark
  and the oracle gate.

**2026-05-08 fused W2 grouped-dense route-reduce update:**
- Added `sm12x_mxfp8_mxfp4_grouped_dense_reduce_routes`, exported through
  `tokenspeed_kernel.ops.moe.sm12x_mxfp4`. It reuses the existing SM120
  MXFP8xMXFP4 tensor-core grouped dense tile, but applies optional W2 bias and
  route weights inside the tile kernel and atomically accumulates directly into
  the token output. This removes the `expanded_out` tensor and separate CUDA
  route-reduce launch from the tensorcore MoE wrapper when enabled with
  `TOKENSPEED_SM12X_MXFP4_FUSE_W2_REDUCE=1`.
- TDD red/green was run on `the primary SM120 workstation`: the new test first failed on the
  missing public op, then passed after the CUDA FFI export, Python wrapper, and
  public exports were added. The test compares the fused op against the previous
  two-step `sm12x_mxfp8_mxfp4_grouped_dense` + `sm12x_mxfp4_reduce_routes`
  sequence, including duplicate token accumulation and W2 bias.
- Focused remote validation passed:
  `tokenspeed-kernel/test/ops/test_sm12x_mxfp4.py` plus
  `tokenspeed-kernel/test/test_sm12x_platform.py` -> `35 passed`; ruff and
  compileall on the changed Python files passed. After the bench-only result
  showed no gain, the fused W2+reduce path was kept opt-in and the broad
  focused regression was rerun -> `36 passed`. The kernel editable install
  rebuilt only the `sm12x_mxfp4_moe` group while reusing
  `~/Workspace/tokenspeed/.venv`.
- Harness oracle gate against the SM120 no-MTP vLLM baseline passed. Run dir:
  `<remote-workspace>/tokenspeed_w2reduce_gate_20260508095343`. All five
  oracle cases returned `ok=true` with matching prompt token ids. `code_probe`,
  `translation`, and `long_prefill_2048` were exact token matches; `raw_intro`
  and `short_math` stopped at accepted low-margin forks with trajectory top1
  match rate `1.0`.
- `1024x1024 c2` tokenizer-free benchmark in the same run dir reported `9.80`
  output tok/s, `19.61` total tok/s, mean TTFT `6792.85 ms`, and mean TPOT
  `197.58 ms`. This is correctness-neutral but not an end-to-end improvement
  over the previous `9.95` output tok/s fused route-gather result.
- A bench-only rerun without `--enable-output-logprobs` confirmed the same
  conclusion. Run dir:
  `<remote-workspace>/tokenspeed_w2reduce_bench_only_20260508100021`;
  result `9.64` output tok/s, `19.27` total tok/s, mean TTFT `2774.65 ms`, and
  mean TPOT `204.46 ms`.
- Conclusion: fusing only W2 grouped dense with route reduce is not enough, so
  it should not be the default path. The next kernel must own the expensive
  boundary earlier in the MoE path, likely starting with W13 output ->
  activation quant -> W2 within one expert-group schedule, and then eventually
  collapsing route gather through final reduce into a #324-style native expert
  tile pipeline.

**2026-05-08 fused W13 grouped-dense SwiGLU quantization update:**
- Added `sm12x_mxfp8_mxfp4_grouped_swiglu_mxfp8_quantize`, exported through
  `tokenspeed_kernel.ops.moe.sm12x_mxfp4`. It computes paired gate/up W13
  tensor-core tiles for one 16-row expert group and one 32-wide intermediate
  block, applies optional W13 bias and SwiGLU, then emits FP8 e4m3fn
  intermediate activations plus UE8M0 scales. This removes the `gate_up`
  intermediate tensor and the separate SwiGLU quantization launch when enabled
  with `TOKENSPEED_SM12X_MXFP4_FUSE_W13_SWIGLU=1`.
- TDD red/green was run on `the primary SM120 workstation`: the new test first failed on the
  missing public op, then passed after the CUDA kernel, FFI export, Python
  wrapper, and public exports were added. A wrapper dispatch test keeps this
  path opt-in.
- Focused remote validation passed:
  `tokenspeed-kernel/test/ops/test_sm12x_mxfp4.py` plus
  `tokenspeed-kernel/test/test_sm12x_platform.py` -> `38 passed`; ruff and
  compileall on the changed Python files passed. The kernel editable install
  rebuilt only the `sm12x_mxfp4_moe` group while reusing
  `~/Workspace/tokenspeed/.venv`.
- Harness oracle gate against the SM120 no-MTP vLLM baseline passed with the
  fused W13+SwiGLU env enabled. Run dir:
  `<remote-workspace>/tokenspeed_w13swiglu_gate_20260508102038`. All five
  oracle cases returned `ok=true` with matching prompt token ids; all divergent
  trajectories stopped at accepted low-margin forks.
- `1024x1024 c2` benchmark in the same logprobs-enabled run reported `8.93`
  output tok/s, `17.85` total tok/s, mean TTFT `2107.78 ms`, and mean TPOT
  `222.22 ms`. A bench-only rerun without `--enable-output-logprobs` confirmed
  the regression: run dir
  `<remote-workspace>/tokenspeed_w13swiglu_bench_only_20260508102634`,
  `8.63` output tok/s, `17.27` total tok/s, mean TTFT `2503.54 ms`, mean TPOT
  `228.88 ms`.
- Conclusion: the naive W13+SwiGLU fusion is correctness-neutral but slower,
  likely because it increases per-CTA work/register pressure and does not
  address the broader expert scheduling/occupancy problem. Keep it as an
  opt-in experiment, not the default. The next attempt should move toward a
  #324-style expert tile pipeline with a schedule designed around reuse and
  occupancy from the start, not incremental launch-count fusion.

**2026-05-08 N=32 grouped dense tile update:**
- Added `sm12x_mxfp8_mxfp4_grouped_dense_n32`, exported through
  `tokenspeed_kernel.ops.moe.sm12x_mxfp4`, plus an opt-in tensorcore dispatch
  controlled by `TOKENSPEED_SM12X_MXFP4_GROUPED_DENSE_N32=1`. The kernel keeps
  the same SM120 MXFP8xMXFP4 MMA tile shape internally, but one CTA now owns
  four adjacent N=8 output tiles and reuses the loaded A fragment across N=32.
  This is a small, controlled step toward larger expert tiles without changing
  the default runtime path.
- TDD red/green was run on `the primary SM120 workstation`: the public-op test first failed
  because `sm12x_mxfp8_mxfp4_grouped_dense_n32` was missing, then passed after
  the CUDA FFI export and Python wrapper were added. A second dispatch test
  first failed because the tensorcore path still called the N=8 grouped dense
  op when the env knob was enabled, then passed after adding the dispatch
  helper.
- Focused remote validation passed:
  `tokenspeed-kernel/test/ops/test_sm12x_mxfp4.py` plus
  `tokenspeed-kernel/test/test_sm12x_platform.py` -> `40 passed`; ruff and
  compileall on the changed Python files passed. The kernel editable install
  rebuilt only the `sm12x_mxfp4_moe` group while reusing
  `~/Workspace/tokenspeed/.venv`.
- Harness oracle gate against the SM120 no-MTP vLLM baseline passed with the
  N=32 grouped-dense env enabled. Run dir:
  `<remote-workspace>/tokenspeed_n32_oracle_20260508174928`. All five oracle
  cases returned `ok=true` with matching prompt token ids; exact-match cases
  remained exact and divergent trajectories stopped at accepted low-margin
  forks.
- `1024x1024 c2` bench-only run without `--enable-output-logprobs` reported
  `9.60` output tok/s, `19.20` total tok/s, mean TTFT `2111.77 ms`, and mean
  TPOT `206.48 ms`. Run dir:
  `<remote-workspace>/tokenspeed_n32_bench_20260508175110`.
- Conclusion: simply widening the current warp-tile grouping to N=32 is
  correctness-neutral but still not a performance lever. It reduces some A
  reload work, but the runtime remains dominated by the staged MoE schedule and
  small CTA occupancy/dispatch overhead. Keep this as an opt-in experiment. The
  next useful step is a real expert-persistent schedule: larger per-expert work
  units, explicit route-block ownership, better occupancy/wave scheduling, and
  then W13 activation and W2 accumulation inside that schedule.

**2026-05-08 N=64/8-warp grouped dense and attention bottleneck update:**
- Added `sm12x_mxfp8_mxfp4_grouped_dense_n64_warp8`, exported through
  `tokenspeed_kernel.ops.moe.sm12x_mxfp4`, plus opt-in dispatch controlled by
  `TOKENSPEED_SM12X_MXFP4_GROUPED_DENSE_N64_WARP8=1`. One CTA owns one
  16-row expert block and N=64, with 8 warps covering adjacent N=8 tiles and
  cooperatively loading the A tile into shared memory once per K block.
- TDD red/green was run on `the primary SM120 workstation`: the public-op test first failed on the
  missing export, then passed after the CUDA kernel, FFI export, Python wrapper,
  and public exports were added. A wrapper dispatch test then failed until the
  tensorcore path selected the N=64/8-warp op under the env knob.
- Low-level microbench showed the first clear grouped-dense primitive gain:
  DS4-shaped `hidden=4096`, `out=14336`, 64 experts, 16-row expert groups went
  from `6.881 ms` with N=8 to `4.180 ms` with N=64/8-warp at 64 active groups
  (`17.5` -> `28.8` TFLOP/s). Smaller group counts also improved by roughly
  `1.5x`-`1.7x`.
- Focused remote validation passed:
  `tokenspeed-kernel/test/ops/test_sm12x_mxfp4.py` plus
  `tokenspeed-kernel/test/test_sm12x_platform.py` -> `42 passed`; ruff and
  compileall on changed Python/test files passed. The kernel editable install
  rebuilt only the affected CUDA group while reusing
  `~/Workspace/tokenspeed/.venv`.
- Harness oracle gate against the SM120 no-MTP vLLM baseline passed with the
  N=64/8-warp env enabled. Run dir:
  `<remote-workspace>/tokenspeed_n64w8_oracle_20260508180744`. All five
  oracle cases returned `ok=true` with matching prompt token ids; exact-match
  cases stayed exact and divergent trajectories stopped at accepted low-margin
  forks.
- `1024x1024 c2` bench-only run without `--enable-output-logprobs` reported
  `9.80` output tok/s, `19.61` total tok/s, mean TTFT `6298.14 ms`, and mean
  TPOT `198.07 ms`. Run dir:
  `<remote-workspace>/tokenspeed_n64w8_bench_20260508180946`.
- Profiling showed why the low-level MoE gain does not lift end-to-end decode:
  SM12x still forces DeepSeek V4 attention through the TokenSpeed reference
  sparse MLA path. `flash_mla` imports, but both FlashMLA sparse prefill and
  sparse decode reject SM120 at runtime: prefill reports only SM90a/SM100f
  support, and decode reports unsupported architecture.

**2026-05-08 CUDA sparse-MLA decode probe:**
- Added `deepseek_v4_sparse_mla_cuda`, a TokenSpeed-owned CUDA primitive for
  the already-dequantized DeepSeek V4 sparse MLA workspace. It is exported
  through `tokenspeed_kernel.ops.attention.deepseek_v4` and remains opt-in via
  `TOKENSPEED_DEEPSEEK_V4_CUDA_SPARSE_MLA=1`.
- Runtime dispatch is deliberately narrow: it only applies in decode mode and
  only when `topk_width <= TOKENSPEED_DEEPSEEK_V4_CUDA_SPARSE_MLA_MAX_TOPK`
  (default `192`). Large topk and all prefill/extend calls stay on the PyTorch
  reference path because the naive per-head CUDA kernel is slower at
  `topk>=384` and is not a prefill kernel.
- TDD red/green was run on `the primary SM120 workstation`: the CUDA op test first failed on the
  missing public export, then passed after the CUDA kernel and Python/ops
  wrappers were added. Runtime dispatch tests verify CUDA opt-in, large-topk
  fallback, and extend-mode fallback.
- Low-level microbench for `tokens=1`, `heads=64`, `head_dim=512` showed the
  kernel is useful only in the small-topk region: `topk=17` was `14.3x` faster,
  `topk=136` was `1.66x` faster, while `topk=384` was `0.87x` of the reference
  speed. This is why the runtime gate is thresholded.
- Full harness oracle gate passed with N=64/8-warp MoE plus CUDA sparse-MLA
  decode gate. Run dir:
  `<remote-workspace>/tokenspeed_cuda_sparse_mla_oracle_20260508183348`. All
  five oracle cases returned `ok=true`; low-margin forks remained accepted and
  prompt token ids matched.
- `1024x1024 c2` bench-only run without logprobs reported `9.97` output tok/s,
  `19.94` total tok/s, mean TTFT `1965.26 ms`, and mean TPOT `198.65 ms`. Run
  dir: `<remote-workspace>/tokenspeed_cuda_sparse_mla_bench_20260508183540`.
  This is correctness-neutral and roughly tied with the previous best no-MTP
  TokenSpeed result, not a material throughput jump.
- Profile run
  `<remote-workspace>/tokenspeed_cuda_sparse_mla_profile_20260508184016`
  confirmed the substage moved in the right direction: sparse MLA decode
  aggregate dropped from roughly `19.3 ms/decode` to `12.0 ms/decode`, but
  `decode_reference_workspace` is still about `17.0 ms/decode` and
  `attention.indexer` about `31.1 ms/decode`. The next high-leverage attention
  step is therefore a direct cache-reading sparse MLA decode kernel that fuses
  FP8 cache dequantization with attention, followed by replacing the indexer
  top-k reference path. Incremental MoE launch fusion should remain secondary
  until those attention/indexer costs are reduced.
