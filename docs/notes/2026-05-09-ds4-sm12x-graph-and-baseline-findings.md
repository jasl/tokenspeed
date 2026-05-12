# DeepSeek V4 SM12x Graph and Baseline Findings

Date: 2026-05-09.

## Apples-to-Apples Baselines

All runs below used `the primary SM120 workstation` with two RTX Pro 6000 Blackwell GPUs and the
same `tokenspeed bench serve --dataset-name token_ids` client.

TokenSpeed, eager, SM12x native path:

- `1024x128 c1`: `18.12` output tok/s, mean TPOT `44.32 ms`.
- `1024x128 c2`: `19.07` output tok/s, mean TPOT `51.57 ms`.
- `1024x1024 c1`: `21.05` output tok/s, mean TPOT `46.13 ms`.

vLLM SM120 branch, `--enforce-eager`, no FP4 indexer cache:

- `1024x128 c1`: `15.69` output tok/s, mean TPOT `62.62 ms`.
- `1024x128 c2`: `28.67` output tok/s, mean TPOT `65.34 ms`.
- `1024x1024 c1`: `15.86` output tok/s, mean TPOT `63.01 ms`.

vLLM SM120 branch, default CUDA graph, no FP4 indexer cache:

- `1024x128 c1`: `79.35` output tok/s, mean TPOT `11.51 ms`.
- `1024x128 c2`: `105.04` output tok/s, mean TPOT `12.85 ms`.
- `1024x1024 c1`: `85.70` output tok/s, mean TPOT `11.58 ms`.

Follow-up context: the recorded vLLM graph run still included extra
synchronization from the branch's CUDA graph safety issue. The user reported
that a later fixed local version exceeded `100` output tok/s without MTP. Treat
the numbers above as a lower bound for the vLLM graphable dataflow, not as the
ceiling.

vLLM rejects `--attention_config.use_fp4_indexer_cache=True` on SM120 with an
assertion that the FP4 indexer cache is only supported on Blackwell datacenter
`sm_10x` GPUs.

## TokenSpeed CUDA Graph Bring-Up

TokenSpeed graph capture initially failed on two graph-unsafe operations:

- `_e2m1_values()` constructed a CUDA tensor from a Python list during capture.
- `_deepseek_v4_indexer_topk_from_cache_batched()` synchronized
  `compressed_lens.max().item()` and used dynamic boolean compaction.

The retained fixes are:

- E2M1 decode now uses pure tensor arithmetic instead of constructing a lookup
  table.
- Decode indexer all-candidate shortcut can use a static capture max length.
- CudaGraphWrapper warm-up runs on the same stream used for capture.

With `--max-cudagraph-capture-size 2`, TokenSpeed captures and replays decode
graphs successfully, but throughput is still effectively unchanged:

- `1024x128 c1`: `18.42` output tok/s, mean TPOT `43.05 ms`.
- `1024x128 c2`: `19.25` output tok/s, mean TPOT `49.89 ms`.
- `1024x1024 c1`: `21.61` output tok/s, mean TPOT `44.89 ms`.

Using the default `--max-cudagraph-capture-size 160` currently OOMs during
capture after high batch sizes because graph/private-pool memory leaves too
little free memory for another `read_deepseek_v4_indexer_mxfp4_cache`
allocation. Limit capture size while the SM12x path is still memory-tight.

## Current Kernel Bottlenecks

Profile: `/tmp/20260509-221210-TP-{0,1}.trace.json.gz`.

TP0 top kernel totals over the profiled window:

- `gate_activation_warp_kernel`: `832.63 ms`.
- `down_warp_kernel`: `492.08 ms`.
- `sparse_mla_fp8_cache_online_softmax_kernel`: `232.34 ms`.
- `ncclDevKernel_AllReduce_Sum_bf16_RING_LL`: `198.46 ms`.
- `sm12x_mhc_pre_split_finalize_kernel`: `145.17 ms`.

Decision: CUDA graph enablement is necessary for parity, but not sufficient for
TokenSpeed's current SM12x path. TokenSpeed already replays a graph but still
spends roughly the whole TPOT in GPU kernels. The vLLM delta is therefore the
graphable DeepSeek V4 decode dataflow itself: fused attention/output projection,
direct sparse MLA, stable buffers, and MoE scheduling. Prioritize those islands
over launch-level or scalar kernel tweaks.

## DS4-Shape Direct-Warp MoE Specialization

The direct-warp MoE path now has an SM12x-only DeepSeek V4 Flash decode-shape
branch for `hidden=4096`, `moe_intermediate=2048`, `top_k=6`, and EP-sized
local routing. This is not the final persistent/#324-style MoE design, but it is
a low-risk cleanup of the current warp schedule.

Matched graph bs=2 serving runs on `the primary SM120 workstation`:

- `1024x128 c1`:
  `<remote-workspace>/tokenspeed_bench_ds4shape_moe_20260509223735`,
  `19.21` output tok/s, mean TPOT `42.23 ms`.
- `1024x1024 c1`:
  `<remote-workspace>/tokenspeed_bench_ds4shape_moe_1024x1024_20260509223901`,
  `22.07` output tok/s, mean TPOT `44.08 ms`.

Compared with the graph bs=2 baseline above, this is a small but consistent
improvement: `18.42 -> 19.21` output tok/s for `1024x128 c1` and
`21.61 -> 22.07` output tok/s for `1024x1024 c1`.

Profile:
`/tmp/tokenspeed_profile_ds4shape_20260509224219/20260509-224304-TP-{0,1}.trace.json.gz`.

TP0 aggregate comparison against the pre-specialization graph profile:

- gate warp kernel: `832.63 ms -> 743.54 ms`.
- down warp kernel: `492.08 ms -> 455.84 ms`.
- sparse MLA: unchanged at about `232 ms`.
- mHC pre split finalize: unchanged at about `145 ms`.
- cublas float GEMV family: unchanged at about `203 ms`.
- total CUDA kernel time in the profiled window: `2686.00 ms -> 2541.83 ms`.

Decision: keep the specialization. It does not solve the main throughput gap,
but it proves the current MoE schedule is still the dominant local lever and
gives a cleaner baseline before moving to a larger persistent/padding-aware
MoE rewrite.

## DS4-Shape mHC Parallel Apply

The split SM12x mHC pre path now keeps the split partial reduction, but for the
DeepSeek V4 decode shape it separates the final `layer_input` mix into a
parallel apply kernel. The coefficient/finalize kernel stores the four `pre`
values in the existing partial scratch buffer, and the apply kernel writes the
4096-wide `layer_input` with many CTAs instead of a single CTA.

Correctness:

- `47 passed` on `the primary SM120 workstation` for the focused SM12x MoE/mHC suite, including a
  new `hidden=4096,hc_mult=4` mHC pre reference comparison.

Performance evidence:

- Isolated DS4-shape `sm12x_mhc_pre` microbench:
  `0.024192 ms/call`.
- `1024x128 c1`:
  `<remote-workspace>/tokenspeed_bench_mhc_parallel_apply_20260509225504`,
  `19.29` output tok/s, mean TPOT `42.02 ms`.
- `1024x1024 c1`:
  `<remote-workspace>/tokenspeed_bench_mhc_parallel_apply_1024x1024_20260509225838`,
  `22.17` output tok/s, mean TPOT `43.86 ms`.

Decode-only profile:
`/tmp/tokenspeed_profile_stage_mhc_parallel_apply_20260509225638/stage-mhc-parallel-20260509225638-TP-{0,1}-DECODE.trace.json.gz`.

TP0 mHC pre aggregate moved from `178.50 ms` in the prior stage profile
(`33.35 ms` partial plus `145.14 ms` finalize) to `171.17 ms`
(`33.36 ms` partial, `134.60 ms` finalize, `3.20 ms` apply). Total kernel time
in the same 32-step decode window dropped from `1338.77 ms` to `1331.10 ms`.

Decision: keep it. The online improvement is small because sparse MLA, MoE, and
compressor/indexer GEMV remain comparable bottlenecks, but this is a clean
SM12x-only shape specialization with positive correctness and throughput data.

## SM12x FP8 Weight GEMV For Decode Linear Layers

DeepSeek V4 FP8 dense linears previously dequantized FP8 weights into a cached
FP32 tensor and used `torch.matmul` for both prefill and decode. The SM12x path
now keeps prefill on that cached FP32-weight matmul, but routes decode-sized
2-D BF16 inputs (`<= 16` rows) through a native exact FP8-weight GEMV kernel:
activations still use the exact DeepSeek V4 FP8 quant-dequant simulation, while
the kernel dequantizes FP8 weights and UE8M0/F32 scales on the fly.

Correctness:

- RED/GREEN runtime dispatch tests on `the primary SM120 workstation` verify both sides of the
  split: decode calls the SM12x FP8 weight GEMV, and prefill-sized inputs do
  not.
- FP8 focused suite:
  `tokenspeed-kernel/test/ops/test_sm12x_fp8_gemm.py` plus the runtime FP8
  activation/linear dispatch tests: `9 passed`.

Performance evidence:

- `1024x128 c1`:
  `<remote-workspace>/tokenspeed_bench_fp8_weight_gemv_20260509232307`,
  `20.18` output tok/s, mean TPOT `39.73 ms`.
- `1024x1024 c1`:
  `<remote-workspace>/tokenspeed_bench_fp8_weight_gemv_1024x1024_20260509232339`,
  `23.44` output tok/s, mean TPOT `41.42 ms`.

Compared with the mHC parallel-apply baseline, this moves `1024x128 c1` from
`19.29 -> 20.18` output tok/s and `1024x1024 c1` from `22.17 -> 23.44`
output tok/s.

Decode-only profile:
`/tmp/tokenspeed_profile_stage_fp8_weight_gemv_20260509232550/stage-fp8weight-20260509232550-TP-{0,1}-DECODE.trace.json.gz`.

TP0 aggregate comparison against the mHC parallel-apply decode profile:

- total CUDA kernel time: `1331.10 ms -> 1259.80 ms`.
- cublas kernel family: `282.67 ms -> 89.86 ms`.
- new `fp8_weight_gemv_ue8m0_kernel`: `126.01 ms`.
- sparse MLA remains about `232 ms`.
- MoE gate/down remains about `230 ms`.
- mHC pre remains about `171 ms`.

Decision: keep it. This is still a simple per-output GEMV schedule, but it is
exact, SM12x-scoped, avoids the prefill regression trap, and removes about
`70 ms` from the 32-step decode profile. The next larger dataflow target should
remain sparse MLA plus MoE schedule fusion/persistence rather than further
polishing this first GEMV kernel.
