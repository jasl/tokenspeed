# DSv4 SM12x T1-γ: Native-CUDA Output Projection — Workstation Runbook

This runbook executes the 4-piece evidence for the SM12x native-CUDA
DeepSeek V4 attention output projection island (inverse-RoPE + FP8 quant
and the `bhr,hdr->bhd` per-group FP8 GEMV). Run every step on the SM120
workstation at `10.0.0.110` (SSH alias: `workstation`) with `TP=2`.

## Change Under Test

- `tokenspeed-kernel/.../csrc/sm12x_deepseek_v4_output_proj.cu` exports
  two SM120 kernels: `sm12x_deepseek_v4_inv_rope_fp8_quant` and
  `sm12x_deepseek_v4_grouped_fp8_gemv`.
- `_project_attention_output` in
  `python/tokenspeed/runtime/models/deepseek_v4.py` dispatches to those
  two CUDA kernels on SM120/SM121 for decode-sized attention outputs
  (`attn_output.shape[0] <= 16`, `bf16`, `wo_a` FP8); a single
  `_DEEPSEEK_V4_SM12X_OUTPUT_PROJ_UNAVAILABLE` flag flips the whole
  island onto the upstream BF16 inverse-RoPE / `torch.bmm` reference on
  any launch-time failure.
- No runtime change for non-SM12x platforms; no upstream DSv4 path
  replaced.

## 1. Build

```bash
ssh workstation
cd ~/Workspace/tokenspeed  # or the active checkout, see `git status`
git fetch origin && git checkout codex/ds4-sm12x-poc && git pull --ff-only
TOKENSPEED_CUDA_ARCH_LIST=120f \
  pip install --no-build-isolation -e tokenspeed-kernel/python/
```

Verify the `.so` landed:

```bash
ls -la tokenspeed-kernel/python/tokenspeed_kernel/thirdparty/cuda/objs/sm12x_deepseek_v4_output_proj/
# expect: sm12x_deepseek_v4_output_proj.so
export PATH=/usr/local/cuda-13.1/bin:$PATH
cuobjdump --dump-resource-usage \
  tokenspeed-kernel/python/tokenspeed_kernel/thirdparty/cuda/objs/sm12x_deepseek_v4_output_proj/sm12x_deepseek_v4_output_proj.so \
  | grep -E "Function|REG:|SHARED:|STACK:|LOCAL:"
# expect: inv_rope_fp8_quant REG:24 STACK:0 LOCAL:0
#         grouped_fp8_einsum REG:39 STACK:0 LOCAL:0
# fail-fast: any STACK>0 / LOCAL>0 / REG>=80 means an occupancy regression
# crept in -- do not proceed to microbench until cuobjdump is clean.
```

## 2. Focused unit tests (1st piece of evidence — correctness)

```bash
pytest -xvs tokenspeed-kernel/test/ops/test_deepseek_v4_projection.py 2>&1 \
  | tee ~/Workspace/tokenspeed_t1_gamma_cuda_tests_$(date +%Y%m%d_%H%M%S).log
```

Expected: every CUDA test passes (CUDA-vs-reference for inverse-RoPE +
FP8 quant at `tokens={1,2,3,5,8}`, CUDA-vs-reference for the einsum,
strided activation, UE8M0 weight scales, zero-tokens edge case).

Broader regression coverage (the dispatch change is non-trivial):

```bash
pytest -xvs tokenspeed-kernel/test/ops/test_deepseek_v4_reference.py
pytest -xvs test/runtime/test_deepseek_v4_attention_ops.py
```

## 3. Isolated microbench (2nd piece — kernel win)

Both CUDA kernels print min/mean/max ms when the opt-in env var is set:

```bash
TOKENSPEED_ENABLE_PROJECTION_MICROBENCH=1 \
  pytest -xvs -k microbench \
    tokenspeed-kernel/test/ops/test_deepseek_v4_projection.py 2>&1 \
  | tee ~/Workspace/tokenspeed_t1_gamma_cuda_microbench_$(date +%Y%m%d_%H%M%S).log
```

Decision rule: the bench is informational once Triton is gone; the
absolute numbers anchor the kernel-level cost so a future
micro-optimisation has something concrete to compare against. Any single
case >2x slower than the last recorded run is a regression worth
investigating.

## 4. Full-model bench (3rd piece — end-to-end Δ)

The current branch ships eager-mode at workstation memory budget; graph
mode OOMs at default settings and is tracked separately as T1-α.

```bash
SERVER_ARGS=(
  --model deepseek-ai/DeepSeek-V4-Flash
  --trust-remote-code
  --tokenizer-mode deepseek_v4
  --kv-cache-dtype fp8
  --attn-tp-size 2
  --ep-size 2
  --enable-expert-parallel
  --moe-backend sm12x_mxfp4
  --max-total-tokens 4096
  --chunked-prefill-size 1024
  --enforce-eager
  --gpu-memory-utilization 0.80
  --disable-kvstore
  --max-num-seqs 4
  --enable-output-logprobs
  --host 127.0.0.1
  --port 8000
)

BENCH_ARGS=(
  --backend openai
  --model deepseek-ai/DeepSeek-V4-Flash
  --base-url http://127.0.0.1:8000
  --dataset-name token_ids
  --token-ids-input-len 1024
  --token-ids-output-len 1024
  --num-prompts 2
  --max-concurrency 1
  --request-rate inf
  --extra-body '{"temperature": 0}'
)

# 1) Start the server in a tmux session (no manual cleanup needed; tmux
#    kill-session does the right thing).
tmux new-session -d -s tkserve "tokenspeed serve ${SERVER_ARGS[*]} \
  > /tmp/tk-serve/log 2>&1"
until curl -s -m 2 http://127.0.0.1:8000/health >/dev/null; do sleep 4; done

# 2) Run the bench, save the bundle.
OUT=$HOME/Workspace/tokenspeed_bench_t1_gamma_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT"
tokenspeed bench serve "${BENCH_ARGS[@]}" 2>&1 | tee "$OUT/bench.log"

# 3) Tear the server down.
tmux kill-session -t tkserve
```

The bench should produce `output_throughput_tok_per_s` and
`mean_tpot_ms`. Compare against the previously recorded eager runs in
`docs/plans/2026-05-09-ds4-sm12x-native-route.md` and the harness
baselines.

Optional profile capture for the engineering loop:

```bash
TOKENSPEED_PROFILE_DECODE_STEPS=32 \
  tokenspeed serve "${SERVER_ARGS[@]}" &
# trigger one decode batch, then SIGTERM
# Traces land at /tmp/<timestamp>-TP-{0,1}.trace.json.gz
```

Compare the `sm12x_deepseek_v4_inv_rope_fp8_quant_kernel` and
`deepseek_v4_grouped_fp8_einsum_kernel` per-step time against the
previous Triton equivalents recorded in the plan-of-record.

## 5. Oracle (4th piece — top-20 logprobs)

```bash
cd ~/Workspace/ds4-sm120-harness
./scripts/run_acceptance.sh \
  --target tokenspeed \
  --target-rev "$(git -C ~/Workspace/tokenspeed rev-parse HEAD)" \
  --oracle-top-n 20 \
  --min-top1-match-rate 0.80 \
  --min-topk-overlap-mean 0.80
```

All five oracle cases must pass with `top1_match_rate >= 0.80` and
`topk_overlap_mean >= 0.80`. If any case fails, the kernel changed
numerical behaviour; treat as a correctness regression and either fix
the kernel or revert to upstream's BF16 reference path.

## 6. Record outcome

Update `docs/plans/2026-05-09-ds4-sm12x-native-route.md` Current State
section with:

- microbench numbers for the two CUDA kernels at the recorded shapes
- full-model bench path (this run's bundle dir)
- profile path (if captured)
- oracle pass / fail summary

## 7. Commit message scaffold

```
<feat|fix>(deepseek-v4): <one-line summary>

<paragraph 1: what changed and why>

Evidence:
- pytest log: ~/Workspace/tokenspeed_t1_gamma_cuda_tests_<ts>.log
- microbench: ~/Workspace/tokenspeed_t1_gamma_cuda_microbench_<ts>.log
- bench: ~/Workspace/tokenspeed_bench_t1_gamma_<ts>
- oracle: <oracle output summary>

Signed-off-by: jasl <jasl9187@hotmail.com>
```
