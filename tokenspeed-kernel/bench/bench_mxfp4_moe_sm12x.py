#!/usr/bin/env python3
# Copyright (c) 2026 LightSeek Foundation
# SPDX-License-Identifier: MIT
"""Standalone microbench for the sm12x Triton MXFP4 MoE matmul path.

Replays the EXACT production call path of
``tokenspeed_kernel.ops.moe.triton.mxfp4.triton_mxfp4_moe_apply`` (both the
gather gate/up matmul and the scatter down matmul, incl. the unfused
silu(gate)*up and the final top-k sum) at DeepSeek-V4-Flash GB10 TP=2 non-EP
shapes:

    256 routed experts/rank, hidden=4096, moe_intermediate per rank=1024
    (2048 / tp2), top-6 routed experts/token, bf16 activations, MXFP4 weights
    (weights preprocessed with the production ``triton_mxfp4_moe_weights``
    swizzle, so BlackwellMXValueLayout/BlackwellMXScaleLayout match serving).

Run on GB10 inside ``~/tokenspeed-sm12x/.venv-ts`` (tokenspeed_kernel
installed editable):

    python bench/bench_mxfp4_moe_sm12x.py
    python bench/bench_mxfp4_moe_sm12x.py \
        --constraints '{"block_n": 128, "num_stages": 2, "epilogue_subtile": 2}'

Constraint keys (triton_kernels opt_flags, NVIDIA): block_m, block_n, block_k,
num_warps, num_stages, epilogue_subtile, is_persistent, split_k, idle_sms,
max_allowable_mn, disable_mx4_block_swap. Notes for this path: block_n and
block_k must be multiples of 128 (BlackwellMXScaleLayout swizzle);
is_persistent=False and split_k>1 raise InapplicableConstraint (swizzled scale
layout requires the persistent kernel; ragged+mx forbids split_k). User
constraints are applied AFTER weight preprocessing, so they override the
production defaults set there (is_persistent=True, epilogue_subtile=1).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import types

import torch

# Import the production op FIRST: it performs the tokenspeed-triton redirect
# and leaves triton_kernels.* cached in sys.modules against the right triton.
import tokenspeed_kernel.ops.moe.triton.mxfp4  # noqa: E402,F401

# Safe now: triton_kernels.* are cached in sys.modules by the import above.
import triton_kernels.matmul as tk_matmul  # noqa: E402
import triton_kernels.matmul_details.opt_flags as opt_flags_mod  # noqa: E402
from triton_kernels.matmul_details.opt_flags import (  # noqa: E402
    InapplicableConstraint,
)

# Import the production op module FIRST: it imports triton_kernels under the
# tokenspeed_triton redirect and installs the has_tma_gather monkeypatch.
from tokenspeed_kernel.ops.moe.triton.mxfp4 import (  # noqa: E402  isort: skip
    triton_mxfp4_moe_apply,
    triton_mxfp4_moe_weights,
)

# DeepSeek-V4-Flash reference shapes (third_party/deepseek_v4_reference_
# inference/inference/config.json), TP=2 non-EP.
DSV4_EXPERTS = 256
DSV4_HIDDEN = 4096
DSV4_INTERMEDIATE_PER_RANK = 1024  # moe_inter_dim 2048 / tp 2
DSV4_TOPK = 6  # n_activated_experts (the shared expert is a separate MLP)
DSV4_ROUTE_SCALE = 1.5
DSV4_SWIGLU_LIMIT = 10.0
MXFP4_BLOCK = 32


def _round_up(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


def build_fake_moe_module(
    num_experts: int,
    hidden: int,
    ispp: int,
    top_k: int,
    device: torch.device,
) -> torch.nn.Module:
    """Random-weight nn.Module shaped exactly like the production MoE layer.

    Mirrors tokenspeed.runtime.layers.moe.weights.mxfp4.create_mxfp4_weight_pair
    (Blackwell branch: ispp padded to 64, hidden unpadded) plus the module
    attributes triton_mxfp4_moe_apply reads.
    """
    ispp_padded = _round_up(ispp, 64)
    module = torch.nn.Module()
    module.top_k = top_k
    module.num_experts = num_experts
    module.ep_size = 1
    module.w13_input_layout = "concatenated"  # DSv4: [gate|up], unfused silu
    module.swiglu_arg = types.SimpleNamespace(alpha=1.0, limit=DSV4_SWIGLU_LIMIT)

    def _u8(*shape: int) -> torch.nn.Parameter:
        data = torch.randint(0, 256, shape, dtype=torch.uint8, device=device)
        return torch.nn.Parameter(data, requires_grad=False)

    def _scale(*shape: int) -> torch.nn.Parameter:
        # e8m0: 127 == 1.0; stay near 1.0 so activations remain finite.
        data = torch.randint(118, 130, shape, dtype=torch.uint8, device=device)
        return torch.nn.Parameter(data, requires_grad=False)

    def _bias(*shape: int) -> torch.nn.Parameter:
        data = 0.01 * torch.randn(*shape, dtype=torch.bfloat16, device=device)
        return torch.nn.Parameter(data, requires_grad=False)

    module.register_parameter(
        "w13_weight", _u8(num_experts, 2 * ispp_padded, hidden // 2)
    )
    module.register_parameter(
        "w13_weight_scale", _scale(num_experts, 2 * ispp_padded, hidden // MXFP4_BLOCK)
    )
    module.register_parameter("w2_weight", _u8(num_experts, hidden, ispp_padded // 2))
    module.register_parameter(
        "w2_weight_scale", _scale(num_experts, hidden, ispp_padded // MXFP4_BLOCK)
    )
    module.register_parameter("w13_weight_bias", _bias(num_experts, 2 * ispp_padded))
    module.register_parameter("w2_weight_bias", _bias(num_experts, hidden))
    return module


def make_routing(
    n_tokens: int,
    num_experts: int,
    top_k: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = torch.rand(n_tokens, num_experts, device=device)
    topk_ids = scores.topk(top_k, dim=-1).indices.to(torch.int32)
    raw = torch.rand(n_tokens, top_k, device=device, dtype=torch.float32) + 0.1
    weights = raw / raw.sum(dim=-1, keepdim=True) * DSV4_ROUTE_SCALE
    return weights.to(torch.bfloat16), topk_ids


@contextlib.contextmanager
def record_opt_flags(records: list):
    """Wrap triton_kernels.matmul.make_opt_flags to capture chosen flags."""
    original = tk_matmul.make_opt_flags

    def recording(*args, **kwargs):
        flags = original(*args, **kwargs)
        records.append({"m": args[5], "n": args[6], "k": args[7], "flags": flags})
        return flags

    tk_matmul.make_opt_flags = recording
    try:
        yield
    finally:
        tk_matmul.make_opt_flags = original


def format_flags(flags) -> str:
    return (
        f"block_m={flags.block_m} block_n={flags.block_n} block_k={flags.block_k} "
        f"warps={flags.num_warps} stages={flags.num_stages} split_k={flags.split_k} "
        f"persistent={flags.is_persistent} ep_subtile={flags.epilogue_subtile} "
        f"group_m={flags.group_m}"
    )


def time_us_per_call(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1e3 / iters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--constraints",
        type=str,
        default=None,
        help="JSON dict for opt_flags.update_opt_flags_constraints, e.g. '{\"block_n\": 128}'",
    )
    parser.add_argument("--tokens", type=str, default="1,8,128,512,2048")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--experts", type=int, default=DSV4_EXPERTS)
    parser.add_argument("--hidden", type=int, default=DSV4_HIDDEN)
    parser.add_argument(
        "--intermediate-per-rank", type=int, default=DSV4_INTERMEDIATE_PER_RANK
    )
    parser.add_argument("--topk", type=int, default=DSV4_TOPK)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assert torch.cuda.is_available(), "CUDA device required"
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    print(
        f"device={props.name} cc={props.major}.{props.minor} sms={props.multi_processor_count} "
        f"smem_optin={props.shared_memory_per_block_optin}"
    )

    module = build_fake_moe_module(
        args.experts, args.hidden, args.intermediate_per_rank, args.topk, device
    )
    # Production weight preprocessing: swizzles values/scales into Blackwell
    # layouts and sets the baseline constraints (is_persistent, epilogue_subtile).
    triton_mxfp4_moe_weights({}, module)
    torch.cuda.synchronize()

    if args.constraints:
        user_constraints = json.loads(args.constraints)
        opt_flags_mod.update_opt_flags_constraints(user_constraints)
    active = opt_flags_mod._get_opt_flags_constraints()
    print(f"active opt_flags constraints: {active}")

    token_counts = [int(t) for t in args.tokens.split(",") if t]
    # dense-math-equivalent FLOPs per routed row: gate/up (K=hidden, N=2*ispp)
    # + down (K=ispp, N=hidden) == 2*hidden*ispp*3
    flops_per_row = 2 * args.hidden * args.intermediate_per_rank * 3

    header = f"{'tokens':>7} {'rows':>7} {'us/call':>10} {'TFLOP/s':>9}  status"
    print(header)
    for n_tokens in token_counts:
        x = 0.1 * torch.randn(
            n_tokens, args.hidden, dtype=torch.bfloat16, device=device
        )
        router_logits = torch.zeros(
            n_tokens, args.experts, dtype=torch.bfloat16, device=device
        )
        topk_weights, topk_ids = make_routing(n_tokens, args.experts, args.topk, device)

        def run():
            return triton_mxfp4_moe_apply(
                {},
                x,
                module,
                router_logits,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
            )

        try:
            records: list = []
            with record_opt_flags(records):
                out = run()
            torch.cuda.synchronize()
            status = "ok" if bool(torch.isfinite(out).all()) else "NON-FINITE OUTPUT"
            for rec in records:
                print(
                    f"  [flags T={n_tokens}] gemm m={rec['m']} n={rec['n']} "
                    f"k={rec['k']}: {format_flags(rec['flags'])}"
                )
            us = time_us_per_call(run, args.warmup, args.iters)
            tflops = flops_per_row * n_tokens * args.topk / (us * 1e-6) / 1e12
            rows = n_tokens * args.topk
            print(f"{n_tokens:>7} {rows:>7} {us:>10.1f} {tflops:>9.2f}  {status}")
        except InapplicableConstraint as exc:
            print(f"{n_tokens:>7} {'-':>7} {'-':>10} {'-':>9}  INAPPLICABLE: {exc}")
        except Exception as exc:  # OutOfResources, launch errors, etc.
            print(f"{n_tokens:>7} {'-':>7} {'-':>10} {'-':>9}  ERROR: {exc}")


if __name__ == "__main__":
    main()
