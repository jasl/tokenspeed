# DSv4 SM12x T1-γ: Native-CUDA Output Projection Einsum — Workstation Runbook

This runbook executes the 4-piece evidence for the new SM12x native-CUDA
DeepSeek V4 attention output-projection einsum
(`sm12x_deepseek_v4_grouped_fp8_gemv`). Run every step on the SM120 workstation
at `10.0.0.110` (SSH alias: `workstation`) with `TP=2`.

## Change Under Test

- Adds `tokenspeed-kernel/.../csrc/sm12x_deepseek_v4_output_proj.cu` — a
  per-group batched FP8 GEMV computing the `bhr,hdr->bhd` einsum used by
  `_project_attention_output` after the inverse-RoPE + FP8 quant step.
- Adds the Python wrapper, kernel registration (`solution="cuda"`), and
  routes `_project_attention_output` to prefer CUDA over the existing Triton
  kernel, with Triton and then BF16-bmm fallbacks gated behind separate
  `_UNAVAILABLE` flags.
- No runtime change for non-SM12x platforms; no upstream DSv4 path replaced.

## 1. Build

```bash
ssh workstation
cd ~/Workspace/tokenspeed
git fetch origin && git checkout codex/ds4-sm12x-poc && git pull --ff-only
# SM12x MoE family still needs the `120f` suffix for FP8/MXFP4 block-scale MMA;
# the new output-proj group is plain FP8 + scalar dequant and builds with
# bare `120` too, but reusing `120f` keeps the env consistent with the rest of
# the SM12x kernels.
TOKENSPEED_CUDA_ARCH_LIST=120f \
  pip install --no-build-isolation -e tokenspeed-kernel/python/
```

Verify the new `.so` landed:

```bash
ls -la tokenspeed-kernel/python/tokenspeed_kernel/thirdparty/cuda/objs/sm12x_deepseek_v4_output_proj/
# Expect: sm12x_deepseek_v4_output_proj.so
```

## 2. Focused unit tests (1st piece of evidence — correctness)

```bash
pytest -xvs tokenspeed-kernel/test/ops/test_deepseek_v4_projection.py 2>&1 \
  | tee ~/Workspace/tokenspeed_t1_gamma_cuda_tests_$(date +%Y%m%d_%H%M%S).log
```

Expected: all CUDA tests pass, including:

- `test_deepseek_v4_fp8_einsum_cuda_matches_reference`
- `test_deepseek_v4_fp8_einsum_cuda_handles_strided_activation`
- `test_deepseek_v4_fp8_einsum_cuda_accepts_ue8m0_weight_scales`
- `test_deepseek_v4_fp8_einsum_cuda_matches_triton_at_decode_shape[tokens=1|2|8]`
- `test_deepseek_v4_fp8_einsum_cuda_handles_zero_tokens`
- Pre-existing Triton + reference tests still pass.

Also run the broader DSv4 attention regression to make sure the dispatch
change in `_project_attention_output` didn't regress non-SM12x paths or other
DSv4 attention ops:

```bash
pytest -xvs tokenspeed-kernel/test/ops/test_deepseek_v4_reference.py
pytest -xvs test/runtime/test_deepseek_v4_attention_ops.py
```

## 3. Isolated microbench (2nd piece of evidence — kernel win)

```bash
TOKENSPEED_ENABLE_PROJECTION_MICROBENCH=1 \
  pytest -xvs -k microbench \
    tokenspeed-kernel/test/ops/test_deepseek_v4_projection.py 2>&1 \
  | tee ~/Workspace/tokenspeed_t1_gamma_cuda_microbench_$(date +%Y%m%d_%H%M%S).log
```

Look for lines like:

```
[t=1] cuda_einsum   min=0.0XX ms mean=0.0XX ms max=0.0XX ms
[t=1] triton_einsum min=0.XX ms mean=0.XX ms max=0.XX ms
[t=1] speedup mean=Yx min=Zx
```

Decision rule: **CUDA `mean_ms` must be < Triton `mean_ms`** for tokens=1, 2, 8.
If CUDA loses at any of these decode shapes, do **not** keep the runtime
preference for CUDA — log the regression into
`docs/notes/2026-05-09-ds4-sm12x-rejected-experiments.md` and either tune the
kernel further or revert to Triton-as-default.

## 4. Full-model bench (3rd piece of evidence — end-to-end Δ)

Build a matched pair: first run with the CUDA dispatch enabled (default after
this change), then a control run with the CUDA path disabled to compare against
the unchanged Triton baseline.

```bash
# Server args used everywhere on the workstation. Keep --max-total-tokens and
# --chunked-prefill-size constrained per the plan-of-record so the SM12x MoE
# work buffers still fit in the KV pool.
SERVER_ARGS=(
  --model-path deepseek-ai/DeepSeek-V4-Flash
  --attn-tp-size 2
  --ep-size 2
  --moe-backend sm12x_mxfp4
  --kv-cache-dtype fp8
  --max-total-tokens 4096
  --chunked-prefill-size 1024
  --enable-output-logprobs
)

BENCH_ARGS=(
  --backend openai
  --model deepseek-ai/DeepSeek-V4-Flash
  --dataset-name token_ids
  --input-len 1024
  --output-len 1024
  --num-prompts 2
  --max-concurrency 1
  --request-rate inf
  --extra-body '{"temperature": 0}'
)

run_bench() {
  local label="$1"
  local out_dir="$HOME/Workspace/tokenspeed_bench_t1_gamma_cuda_${label}_$(date +%Y%m%d_%H%M%S)"
  mkdir -p "$out_dir"
  # ... start server, wait for ready ...
  tokenspeed serve "${SERVER_ARGS[@]}" --output-dir "$out_dir" &
  SERVE_PID=$!
  # Wait for /health (loop until ready), then:
  tokenspeed bench serve "${BENCH_ARGS[@]}" --output-dir "$out_dir"
  kill $SERVE_PID
  echo "saved: $out_dir"
}

# Run 1: CUDA dispatch on (default).
run_bench cuda_on

# Run 2: control. Disable the CUDA path via the dev-environment flag so the
# runtime falls back to Triton einsum (still with the inv-RoPE + FP8 quant
# CUDA-prep). Set _DEEPSEEK_V4_SM12X_OUTPUT_PROJ_CUDA_UNAVAILABLE=True before
# the engine builds the model, e.g. via a pytest fixture or a small
# `--engine-script` hook. If you don't have a hook handy, simply remove the
# import line for `deepseek_v4_fp8_einsum_sm12x_cuda` and rebuild.
run_bench triton_baseline
```

Cross the two bundles:

- `output_throughput_tok_per_s` (CUDA on) must be >= baseline.
- `mean_tpot_ms` (CUDA on) must be <= baseline.
- No regression in `p99_tpot_ms` greater than 1 ms.

Profile capture (optional but recommended once the bench passes):

```bash
TOKENSPEED_PROFILE_DECODE_STEPS=32 \
  tokenspeed serve ... &
# trigger one decode batch, then SIGTERM
# Traces land at /tmp/<timestamp>-TP-{0,1}.trace.json.gz
```

Compare the `_fp8_einsum*` kernel time per decode step between the two profiles.
The vLLM reference shows ~0.5 ms / 32 steps for the Triton equivalent; expect
CUDA to be substantially lower.

## 5. Oracle (4th piece of evidence — top-20 logprobs)

```bash
cd ~/Workspace/ds4-sm120-harness
./scripts/run_acceptance.sh \
  --target tokenspeed \
  --target-rev "$(git -C ~/Workspace/tokenspeed rev-parse HEAD)" \
  --oracle-top-n 20 \
  --min-top1-match-rate 0.80 \
  --min-topk-overlap-mean 0.80
```

All five oracle cases (short math / raw intro / translation / code / long
prefill 2048) must pass with `top1_match_rate >= 0.80` and `topk_overlap_mean
>= 0.80`. If any case fails, treat as a correctness regression and either fix
the kernel or revert to Triton-as-default.

## 6. Record outcome

Update `docs/plans/2026-05-09-ds4-sm12x-native-route.md` Current State section
with:

- microbench numbers (cuda_ms vs triton_ms at tokens=1/2/8)
- full-model bench paths (cuda_on / triton_baseline bundles)
- profile path (if captured)
- oracle pass / fail summary

If everything passes, the existing Triton path stays in tree as the immediate
fallback. Do not delete `deepseek_v4_fp8_einsum_sm12x_triton` yet — it is the
load-bearing safety net for environments without the SM12x build.

## 7. Commit message scaffold

```
feat(deepseek-v4): add SM12x CUDA attention output-projection einsum

Replace the Triton bhr,hdr->bhd kernel with a native CUDA per-group batched
FP8 GEMV at the DeepSeek V4 attention output projection, behind the existing
SM12x decode-sized predicate. The CUDA kernel mirrors
`fp8_weight_gemv_ue8m0_kernel` (block-128 + UE8M0/fp32 scales, bf16 output)
and adds a group axis so all local groups process in one launch.

Triton einsum and BF16 inverse-RoPE/bmm reference paths remain wired as
fallbacks; the dispatch in `_project_attention_output` prefers CUDA, falls
through to Triton on launch failure, and finally to the reference.

Tests: focused correctness vs reference + cross-check vs Triton at the
DSv4-Flash decode shape (tokens=1/2/8, groups=8, hidden=2048, out_rank=1024);
microbench under `TOKENSPEED_ENABLE_PROJECTION_MICROBENCH=1`.

Evidence:
- pytest log: ~/Workspace/tokenspeed_t1_gamma_cuda_tests_<ts>.log
- microbench: ~/Workspace/tokenspeed_t1_gamma_cuda_microbench_<ts>.log
- bench (cuda_on):       ~/Workspace/tokenspeed_bench_t1_gamma_cuda_cuda_on_<ts>
- bench (triton baseline): ~/Workspace/tokenspeed_bench_t1_gamma_cuda_triton_baseline_<ts>
- oracle: <oracle output summary>

Signed-off-by: jasl <jasl9187@hotmail.com>
```
