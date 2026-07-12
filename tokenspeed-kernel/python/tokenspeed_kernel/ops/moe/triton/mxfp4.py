# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

import copy
import functools
import json
import os
from contextlib import contextmanager

import tokenspeed_kernel
import torch
from tokenspeed_kernel._triton import redirect_triton_to_tokenspeed_triton
from tokenspeed_kernel.ops.activation import fused_silu_and_mul
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

with redirect_triton_to_tokenspeed_triton():
    import triton_kernels  # noqa: F401
    import triton_kernels.matmul  # noqa: F401
    import triton_kernels.matmul_details  # noqa: F401
    import triton_kernels.matmul_details.opt_flags  # noqa: F401
    import triton_kernels.numerics  # noqa: F401
    import triton_kernels.swiglu  # noqa: F401
    import triton_kernels.tensor  # noqa: F401
    import triton_kernels.tensor_details  # noqa: F401
    import triton_kernels.tensor_details.layout  # noqa: F401
    import triton_kernels.topk  # noqa: F401

import triton_kernels.matmul_details.opt_flags as opt_flags
from triton_kernels.matmul import (
    FlexCtx,
    FnSpecs,
    FusedActivation,
    PrecisionConfig,
    matmul,
)
from triton_kernels.matmul_details.opt_flags import scoped_opt_flags_constraints
from triton_kernels.numerics import InFlexData
from triton_kernels.swiglu import swiglu_fn
from triton_kernels.tensor import (
    FP4,
    RaggedTensorMetadata,
    convert_layout,
    make_ragged_tensor_metadata,
    wrap_torch_tensor,
)
from triton_kernels.tensor_details import layout
from triton_kernels.topk import topk

# isort: off
from tokenspeed_kernel.ops.quantization.triton import fp8_quantize

platform = current_platform()

# Consumer Blackwell (sm_120/sm_121) has FP4 tensor cores and basic TMA but NOT
# the `.tile::gather4`/`.tile::scatter4` TMA tile gather/scatter — that is a
# datacenter-Blackwell (sm_100/sm_103) feature and ptxas aborts on sm_120a.
# triton_kernels' `has_tma_gather()` only tests `cuda_capability >= (10, 0)`, which
# wrongly includes sm12x, so the mxfp4 MoE matmul emits an unsupported gather/
# scatter epilogue. Report no TMA gather on consumer Blackwell so the (still
# persistent, still native-FP4) matmul falls back to indexed gather/scatter.
import triton_kernels.target_info as _tk_target_info  # noqa: E402

_tk_orig_has_tma_gather = _tk_target_info.has_tma_gather


def _has_tma_gather_consumer_safe() -> bool:
    if current_platform().is_consumer_blackwell:
        return False
    return _tk_orig_has_tma_gather()


if getattr(_tk_target_info.has_tma_gather, "__name__", "") != (
    "_has_tma_gather_consumer_safe"
):
    _tk_target_info.has_tma_gather = _has_tma_gather_consumer_safe

MXFP4_BLOCK = 32
MXFP4_ACTIVATION_SCALE_LAYOUT = "linear"


def _uses_dynamic_mxfp4_activations(w: torch.nn.Module) -> bool:
    quant_config = getattr(w, "quant_config", None)
    return current_platform().is_amd and bool(
        getattr(quant_config, "use_dynamic_mxfp4_activations", False)
    )


def _quantize_mxfp4_activation(
    activations: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return tokenspeed_kernel.quantize_mxfp4(
        activations.contiguous(),
        scale_size=MXFP4_BLOCK,
        scale_layout=MXFP4_ACTIVATION_SCALE_LAYOUT,
        solution="triton",
        enable_pdl=False,
    )


def _with_activation_mx_scale(
    precision_config: PrecisionConfig | None,
    activation_scale: torch.Tensor,
) -> PrecisionConfig:
    if precision_config is None:
        precision_config = PrecisionConfig()
    precision_config = copy.copy(precision_config)
    precision_config.a_mx_scale = activation_scale
    precision_config.a_microblock_size = MXFP4_BLOCK
    return precision_config


def _release_parameter(module: torch.nn.Module, name: str) -> None:
    if name in module._parameters:
        module.register_parameter(name, None)
    elif hasattr(module, name):
        delattr(module, name)


def _silu_gate_up(
    gate_up: torch.Tensor,
    *,
    output_dtype: torch.dtype,
    gate_second: bool = False,
) -> torch.Tensor:
    if gate_second:
        # GEMM1 emitted [up | gate] rather than the usual [gate | up]: the
        # sm12x_hybrid solution backs this Triton path with the FlashInfer
        # CUTLASS weight residency, whose w13 is reordered to [w3|w1] == [up|gate]
        # (see flashinfer.cutlass_mxfp4._reorder_w13). fused_silu_and_mul consumes
        # [gate | up] (gate first), so swap the halves back before the fused
        # activation. The swap is a byte-exact reorder of the same activation
        # values, so the result is bit-identical to the native [gate|up] path.
        half = gate_up.shape[-1] // 2
        gate_up = torch.cat((gate_up[..., half:], gate_up[..., :half]), dim=-1)
    return fused_silu_and_mul(gate_up, output_dtype=output_dtype)


def _is_bf16_mxfp4(x, w, precision_config):
    if precision_config is None:
        return False
    if getattr(precision_config, "b_mx_scale", None) is None:
        return False
    x_dtype = getattr(x, "dtype", None)
    if x_dtype not in (torch.float16, torch.bfloat16):
        return False
    w_bw = getattr(getattr(w, "dtype", None), "bitwidth", None)
    return w_bw == 4


def _lds_guard_should_apply(x, w, precision_config):
    if scoped_opt_flags_constraints is None:
        return False
    if not current_platform().is_cdna4:
        return False
    return _is_bf16_mxfp4(x, w, precision_config)


@contextmanager
def _maybe_lds_guard(x, w, precision_config):
    if not _lds_guard_should_apply(x, w, precision_config):
        yield
        return
    with scoped_opt_flags_constraints({"block_m": 64, "block_n": 128, "block_k": 256}):
        yield


# Consumer Blackwell (sm_120/121): the datacenter-Blackwell opt_flags
# heuristics size the software pipeline for ~227KB smem (B200) while sm12x
# has ~101KB (and the model double-counts the fp4 pad on the bf16-upcast
# path), so num_stages collapses to 1 -- zero pipelining -- at every
# DeepSeek-V4 shape, and large-M tiles invert efficiency (T=2048 slower per
# FLOP than T=512). Measured on GB10 (DSv4-Flash TP=2 shapes): these
# constraints give 12518us vs 82489us baseline at T=2048 (6.6x, 24.7 TFLOP/s)
# and 4.2x at T=1024; below ~4096 gathered rows the stock small-M defaults
# (block_m 16-32, stages 1) remain faster, so decode/small-prefill keep them.
_SM12X_MOE_LARGE_M_CONSTRAINTS = {
    "block_m": 64,
    "block_n": 128,
    "num_stages": 2,
    "num_warps": 8,
}
_SM12X_MOE_LARGE_M_MIN_ROWS = 4096


@functools.cache
def _sm12x_moe_constraints(which: str) -> dict:
    """Per-gemm large-M constraints, env-overridable for tuning sweeps.

    ``TOKENSPEED_MOE_SM12X_GEMM1_CONSTRAINTS`` / ``..._GEMM2_CONSTRAINTS``
    (JSON) override the measured defaults per gemm -- gate/up is
    K=hidden/N=2*ispp while down-proj is K=ispp/N=hidden, so their optima
    can diverge. Empty/absent env keeps the shared defaults.
    """
    raw = os.environ.get(f"TOKENSPEED_MOE_SM12X_{which}_CONSTRAINTS", "")
    if raw.strip():
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict) and parsed:
                return parsed
        except ValueError:
            pass
    return _SM12X_MOE_LARGE_M_CONSTRAINTS


def _sm12x_moe_tuning_should_apply(x, w, precision_config, num_rows):
    if scoped_opt_flags_constraints is None:
        return False
    if num_rows < _SM12X_MOE_LARGE_M_MIN_ROWS:
        return False
    if not current_platform().is_consumer_blackwell:
        return False
    return _is_bf16_mxfp4(x, w, precision_config)


@contextmanager
def _maybe_sm12x_moe_tuning(x, w, precision_config, num_rows, which="GEMM1"):
    if not _sm12x_moe_tuning_should_apply(x, w, precision_config, num_rows):
        yield
        return
    with scoped_opt_flags_constraints(_sm12x_moe_constraints(which)):
        yield


def _routing(
    logits: torch.Tensor,
    n_expts_act: int,
    sm_first: bool = False,
    dtype: torch.dtype | None = None,
) -> tuple[RaggedTensorMetadata, torch.Tensor, torch.Tensor, torch.Tensor]:
    if dtype is None:
        dtype = logits.dtype

    assert logits.ndim == 2, "router_logits must be (n_tokens, n_expts_tot)"
    n_tokens, n_expts_tot = logits.shape

    assert sm_first is False, "sm_first=True not supported for triton_kernels routing"
    # triton_kernels' fused topk AND its bitmatrix-metadata kernel hard-require a
    # power-of-2 selection width (tl.arange/tl.topk/tl.sort over k, and tl.sort
    # over 32*k inside make_bitmatrix_metadata). DeepSeek-V4 routes top-6, so pad
    # the selection width up to the next power of two. This is exact for the
    # DeepSeek-V4 path because the router logits are packed log-weights (-1e20 at
    # every non-selected expert, log(w_i) at the selected experts whose weights
    # already sum to 1): softmax over the padded top-k recovers each real expert's
    # weight w_i and assigns every pad slot exp(-1e20) == 0, so the padded experts
    # are inert and no renormalization is needed. No-op when k is already a power
    # of two (e.g. gpt-oss top-4).
    k_eff = min(1 << (n_expts_act - 1).bit_length(), n_expts_tot)
    sparse = topk(logits, k_eff, apply_softmax=not sm_first)
    mask_metadata = sparse.mask_metadata

    col_sorted = mask_metadata.col_sorted_indx
    gather_indx = col_sorted // k_eff
    scatter_indx = col_sorted

    vals_flat = sparse.vals.reshape(-1)
    if dtype is not None and vals_flat.dtype != dtype:
        vals_flat = vals_flat.to(dtype)
    gate_scal = vals_flat[scatter_indx]

    n_total_rows = n_tokens * k_eff
    ragged_metadata = make_ragged_tensor_metadata(mask_metadata.col_sum, n_total_rows)

    return ragged_metadata, gather_indx, scatter_indx, gate_scal


def _swizzle_mxfp4(quant_tensor, scale, num_warps):
    """Weight swizzle for mxfp4 MoE, used for OAI mxfp4 kernel."""

    value_layout = layout.make_default_matmul_mxfp4_w_layout(mx_axis=-2)
    scale_layout = layout.make_default_matmul_mxfp4_w_scale_layout(
        mx_axis=-2, num_warps=num_warps
    )
    # Generic Triton matmul hints (persistent + epilogue subtiling), not a
    # datacenter-only path — consumer Blackwell (sm_120/sm_121) wants them too.
    if platform.is_blackwell_plus:
        constraints = {
            "is_persistent": True,
            "epilogue_subtile": 1,
        }
        opt_flags.update_opt_flags_constraints(constraints)
    elif platform.is_hopper:
        constraints = {
            "split_k": 1,
        }
        opt_flags.update_opt_flags_constraints(constraints)
    elif platform.is_amd:
        # Fix block_k=256 to support scale swizzling.
        constraints = {
            "block_k": 256,
        }
        opt_flags.update_opt_flags_constraints(constraints)
    # transpose the tensor so that the quantization axis is on dim1
    quant_tensor = quant_tensor.transpose(-2, -1)
    scale = scale.transpose(-2, -1)
    quant_tensor = convert_layout(
        wrap_torch_tensor(quant_tensor, dtype=FP4), value_layout
    )
    scale = convert_layout(wrap_torch_tensor(scale), scale_layout)
    return quant_tensor, InFlexData(), scale


def _strided_mxfp4(quant_tensor, scale, num_warps):
    """Zero-copy StridedLayout wrap of a raw row-major packed-FP4 weight.

    Twin of :func:`_swizzle_mxfp4` that keeps the weight BYTES in place: it wraps
    a transposed view of the caller's buffer (so the quantization axis lands on
    ``dim(-2)`` with ``stride(-2) == 1``) as a triton_kernels ``StridedLayout``
    FP4 tensor — no value copy. Only the E8M0 block scales are materialized into
    the Blackwell MX scale layout (the same ~1/16-sized copy that
    ``_swizzle_mxfp4`` makes). ``matmul`` selects the ``SWIZZLE_MX_VALUE="STRIDED"``
    epilogue for this layout on cap>=10; the numeric result is bit-identical to
    the ``_swizzle_mxfp4`` Blackwell value layout at DeepSeek-V4 decode shapes
    (verified on GB10/SM121), the two differing only in on-chip value tiling.

    Sharing one physical residency lets the sm12x_hybrid MoE dispatch the same
    weight bytes to either the FlashInfer CUTLASS path (raw buffer) or this
    Triton path (this strided view) without duplicating the ~44 GB/rank experts.
    """
    scale_layout = layout.make_default_matmul_mxfp4_w_scale_layout(
        mx_axis=-2, num_warps=num_warps
    )
    # Mirror the generic matmul hints _swizzle_mxfp4 sets so the strided value
    # layout shares opt_flags with the swizzled one (persistent + epilogue
    # subtiling on Blackwell); required for tile-for-tile numeric agreement.
    if platform.is_blackwell_plus:
        opt_flags.update_opt_flags_constraints(
            {"is_persistent": True, "epilogue_subtile": 1}
        )
    elif platform.is_hopper:
        opt_flags.update_opt_flags_constraints({"split_k": 1})
    elif platform.is_amd:
        opt_flags.update_opt_flags_constraints({"block_k": 256})
    # Transpose so the quantization axis is on dim1 (a view; no copy).
    quant_tensor = quant_tensor.transpose(-2, -1)
    if quant_tensor.stride(-2) != 1:
        raise ValueError(
            "strided mxfp4 wrap requires a row-major packed weight; expected "
            f"stride(-2)==1 after transpose, got strides {tuple(quant_tensor.stride())}"
        )
    scale = scale.transpose(-2, -1)
    # layout=None -> StridedLayout (zero-copy view of the caller's buffer).
    quant_wrapped = wrap_torch_tensor(quant_tensor, dtype=FP4)
    scale_wrapped = convert_layout(wrap_torch_tensor(scale), scale_layout)
    return quant_wrapped, InFlexData(), scale_wrapped


def _routing_from_topk(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_experts: int,
    dtype: torch.dtype | None = None,
) -> tuple[RaggedTensorMetadata, torch.Tensor, torch.Tensor, torch.Tensor]:
    if topk_ids.ndim != 2:
        raise ValueError(f"topk_ids must be rank-2, got {tuple(topk_ids.shape)}")
    if topk_weights.shape != topk_ids.shape:
        raise ValueError(
            "topk_weights and topk_ids must have the same shape, got "
            f"{tuple(topk_weights.shape)} and {tuple(topk_ids.shape)}"
        )
    if num_experts <= 0:
        raise ValueError(f"num_experts must be positive, got {num_experts}")

    flat_ids = topk_ids.reshape(-1).to(torch.long)
    valid = flat_ids >= 0
    safe_ids = torch.where(valid, flat_ids, flat_ids.new_zeros(()))
    sort_order = torch.argsort(safe_ids, stable=True)

    top_k = topk_ids.shape[1]
    gather_indx = (sort_order // top_k).to(torch.int32)
    scatter_indx = sort_order.to(torch.int32)
    gate_scal = topk_weights.reshape(-1)[sort_order]
    gate_scal = torch.where(valid[sort_order], gate_scal, torch.zeros_like(gate_scal))
    if dtype is not None and gate_scal.dtype != dtype:
        gate_scal = gate_scal.to(dtype)

    col_sum = torch.zeros((num_experts,), dtype=torch.int32, device=safe_ids.device)
    col_sum.scatter_add_(
        0,
        safe_ids,
        torch.ones_like(safe_ids, dtype=torch.int32),
    )
    n_total_rows = int(sort_order.numel())
    ragged_metadata = make_ragged_tensor_metadata(col_sum, n_total_rows)

    return ragged_metadata, gather_indx, scatter_indx, gate_scal


def _local_topk_for_ep(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w: torch.nn.Module,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    ep_size = int(getattr(w, "ep_size", 1))
    if ep_size <= 1:
        return topk_weights, topk_ids, int(getattr(w, "num_experts"))

    num_local_experts = int(getattr(w, "num_local_experts"))
    expert_offset = int(getattr(w, "ep_rank", 0)) * num_local_experts
    local_ids = topk_ids - expert_offset
    local_mask = (local_ids >= 0) & (local_ids < num_local_experts)
    local_weights = torch.where(
        local_mask, topk_weights, torch.zeros_like(topk_weights)
    )
    local_ids = torch.where(local_mask, local_ids, topk_ids.new_full((), -1))
    return local_weights, local_ids, num_local_experts


def triton_mxfp4_moe_weights(plan: dict, w: torch.nn.Module):
    MXFP_BLOCK_SIZE = 32

    if hasattr(w, "w13_weight_bias"):
        w.w13_weight_bias = torch.nn.Parameter(
            w.w13_weight_bias.to(torch.float32), requires_grad=False
        )
    if hasattr(w, "w2_weight_bias"):
        w.w2_weight_bias = torch.nn.Parameter(
            w.w2_weight_bias.to(torch.float32), requires_grad=False
        )

    num_warps = 8
    w13_weight, w13_flex, w13_scale = _swizzle_mxfp4(
        w.w13_weight, w.w13_weight_scale, num_warps
    )
    w2_weight, w2_flex, w2_scale = _swizzle_mxfp4(
        w.w2_weight, w.w2_weight_scale, num_warps
    )

    if hasattr(w, "w13_input_scale") and hasattr(w, "w2_input_scale"):
        # Collapse per-expert input scales to a single per-tensor scale
        # per GEMM. Quark exports a constant value across experts for
        # static ``per_tensor`` quantisation; ``max`` is a safe reduction
        # in case individual experts reach slightly different values.
        w13_in_scale = (
            w.w13_input_scale.data.to(torch.float32)
            .max()
            .reshape(1)
            .to(w.w13_input_scale.device)
            .contiguous()
        )
        w2_in_scale = (
            w.w2_input_scale.data.to(torch.float32)
            .max()
            .reshape(1)
            .to(w.w2_input_scale.device)
            .contiguous()
        )
        w.w13_act_scale = w13_in_scale
        w.w2_act_scale = w2_in_scale

        fp8_dtype = current_platform().fp8e4m3fn.dtype
        w13_lhs = InFlexData(dtype=fp8_dtype, scale=w13_in_scale)
        w2_lhs = InFlexData(dtype=fp8_dtype, scale=w2_in_scale)
        # Force bf16 output so the swiglu / down-proj results stay in a
        # standard floating dtype; without this, ``triton_kernels.matmul``
        # defaults ``out_dtype`` to the input dtype (fp8) which would
        # make the subsequent reductions / re-quantisation blow up.
        out_dtype = torch.bfloat16
    else:
        w13_lhs = InFlexData()
        w2_lhs = InFlexData()
        out_dtype = torch.bfloat16 if _uses_dynamic_mxfp4_activations(w) else None

    w.w13_precision_config = PrecisionConfig(
        flex_ctx=FlexCtx(lhs_data=w13_lhs, rhs_data=w13_flex),
        b_mx_scale=w13_scale,
        b_microblock_size=MXFP_BLOCK_SIZE,
        out_dtype=out_dtype,
    )
    w.w2_precision_config = PrecisionConfig(
        flex_ctx=FlexCtx(lhs_data=w2_lhs, rhs_data=w2_flex),
        b_mx_scale=w2_scale,
        b_microblock_size=MXFP_BLOCK_SIZE,
        out_dtype=out_dtype,
    )

    w.w13_weight_triton_tensor = w13_weight
    w.w2_weight_triton_tensor = w2_weight
    # Free original weights and scales (replaced by the swizzled versions;
    # the matmul reads only PrecisionConfig.b_mx_scale). Keeping the raw
    # scales alive costs ~100 MB/layer/rank -- ~4.4 GB/rank on DSv4-Flash.
    _release_parameter(w, "w13_weight")
    _release_parameter(w, "w2_weight")
    _release_parameter(w, "w13_weight_scale")
    _release_parameter(w, "w2_weight_scale")
    torch.cuda.empty_cache()


@register_kernel(
    "moe",
    "apply",
    name="triton_mxfp4_precomputed_moe_apply",
    solution="triton",
    weight_preprocessor=triton_mxfp4_moe_weights,
    capability=CapabilityRequirement(vendors=frozenset({"amd"})),
    signatures=format_signatures(
        "x",
        "dense",
        {torch.float16, torch.bfloat16},
    ),
    traits={
        "weight_dtype": frozenset({"mxfp4"}),
        "activation": frozenset({"silu"}),
        "routing_mode": frozenset({"precomputed_topk"}),
        "supports_deferred_finalize": frozenset({False}),
        "supports_ep": frozenset({False}),
        "supports_all_to_all_ep": frozenset({False}),
        "ispp_alignment": frozenset({1}),
        "internal_activation_dtype": frozenset({"fp8", "input"}),
        "supports_bias": frozenset({True}),
    },
    priority=Priority.SPECIALIZED + 2,
)
@register_kernel(
    "moe",
    "apply",
    name="triton_mxfp4_ep_precomputed_moe_apply",
    solution="triton",
    weight_preprocessor=triton_mxfp4_moe_weights,
    capability=CapabilityRequirement(vendors=frozenset({"amd"})),
    signatures=format_signatures(
        "x",
        "dense",
        {torch.float16, torch.bfloat16},
    ),
    traits={
        "weight_dtype": frozenset({"mxfp4"}),
        "activation": frozenset({"silu"}),
        "routing_mode": frozenset({"precomputed_topk"}),
        "supports_deferred_finalize": frozenset({False}),
        "supports_ep": frozenset({True}),
        "supports_all_to_all_ep": frozenset({False}),
        "ispp_alignment": frozenset({1}),
        "internal_activation_dtype": frozenset({"fp8", "input"}),
        "supports_bias": frozenset({True}),
    },
    priority=Priority.SPECIALIZED + 1,
)
@register_kernel(
    "moe",
    "apply",
    name="triton_mxfp4_moe_apply",
    solution="triton",
    weight_preprocessor=triton_mxfp4_moe_weights,
    signatures=format_signatures(
        "x",
        "dense",
        {torch.float16, torch.bfloat16},
    ),
    traits={
        "weight_dtype": frozenset({"mxfp4"}),
        "activation": frozenset({"silu", "swiglu"}),
        "routing_mode": frozenset({"kernel_routing"}),
        "supports_deferred_finalize": frozenset({False}),
        "supports_ep": frozenset({False}),
        "supports_all_to_all_ep": frozenset({False}),
        "ispp_alignment": frozenset({1}),
        "internal_activation_dtype": frozenset({"input", "fp8"}),
        "supports_bias": frozenset({True}),
    },
    priority=Priority.PORTABLE,
)
@register_kernel(
    "moe",
    "apply",
    name="triton_mxfp4_ep_swiglu_sm12x_moe_apply",
    solution="triton",
    weight_preprocessor=triton_mxfp4_moe_weights,
    # Consumer-Blackwell (sm_120/sm_121) expert-parallel mxfp4 MoE. The shared
    # impl below already handles the precomputed-topk EP path (_local_topk_for_ep)
    # and swiglu; the existing EP registration was gated to AMD + silu-only, so
    # NVIDIA EP found no kernel. Enables --enable-expert-parallel on sm12x (shards
    # the 256 experts across nodes, matching vLLM) without deepep all-to-all.
    capability=CapabilityRequirement(
        vendors=frozenset({"nvidia"}),
        min_arch_version=ArchVersion(12, 0),
    ),
    signatures=format_signatures(
        "x",
        "dense",
        {torch.float16, torch.bfloat16},
    ),
    traits={
        "weight_dtype": frozenset({"mxfp4"}),
        "activation": frozenset({"silu", "swiglu"}),
        "routing_mode": frozenset({"precomputed_topk"}),
        "supports_deferred_finalize": frozenset({False}),
        "supports_ep": frozenset({True}),
        "supports_all_to_all_ep": frozenset({False}),
        "ispp_alignment": frozenset({1}),
        "internal_activation_dtype": frozenset({"fp8", "input"}),
        "supports_bias": frozenset({True}),
    },
    priority=Priority.SPECIALIZED + 1,
)
def triton_mxfp4_moe_apply(
    plan: dict,
    x: torch.Tensor,
    w: torch.nn.Module,
    router_logits: torch.Tensor,
    topk_weights: torch.Tensor | None = None,
    topk_ids: torch.Tensor | None = None,
    num_tokens_global: int | None = None,
    max_num_tokens_per_gpu: int | None = None,
    do_finalize: bool = True,
    enable_pdl: bool = False,
):
    del enable_pdl
    swiglu_arg = getattr(w, "swiglu_arg", None)

    top_k = getattr(w, "top_k")
    n_tokens = router_logits.shape[0]

    if topk_weights is not None or topk_ids is not None:
        if topk_weights is None or topk_ids is None:
            raise ValueError("topk_weights and topk_ids must be provided together")
        topk_weights, topk_ids, num_experts = _local_topk_for_ep(
            topk_weights,
            topk_ids,
            w,
        )
        ragged_metadata, gather_indx, scatter_indx, gate_scal = _routing_from_topk(
            topk_weights,
            topk_ids,
            num_experts=num_experts,
            dtype=router_logits.dtype,
        )
    else:
        ragged_metadata, gather_indx, scatter_indx, gate_scal = _routing(
            router_logits,
            top_k,
            sm_first=False,
            dtype=router_logits.dtype,
        )

    w13_weight = w.w13_weight_triton_tensor
    w2_weight = w.w2_weight_triton_tensor
    w13_bias = getattr(w, "w13_weight_bias", None)
    w2_bias = getattr(w, "w2_weight_bias", None)
    w13_pc = getattr(w, "w13_precision_config", None)
    w2_pc = getattr(w, "w2_precision_config", None)

    act = None
    if swiglu_arg is not None and (
        getattr(w, "w13_input_layout", "concatenated") == "interleaved"
    ):
        # The triton_kernels fused SwiGLU consumes interleaved [g,u,g,u,...]
        # gate/up pairs and hardwires the gpt-oss form silu(alpha*gate)*(up+1)
        # (and requires a numeric alpha). Only use it for that interleaved
        # layout. Standard SwiGLU models (e.g. DeepSeek-V4: concatenated
        # [gate|up], silu(gate)*up with no "(up+1)" and alpha=1.0) fall through
        # to the unfused _silu_gate_up path below, which is byte-correct for them.
        act = FusedActivation(
            FnSpecs("swiglu", swiglu_fn, ("alpha", "limit"), reduction_n=2),
            (swiglu_arg.alpha, swiglu_arg.limit),
        )

    use_dynamic_mxfp4 = _uses_dynamic_mxfp4_activations(w)
    if hasattr(w, "w13_act_scale"):
        gemm1_input = fp8_quantize(x, scale=w.w13_act_scale)
    elif use_dynamic_mxfp4:
        gemm1_input, gemm1_scale = _quantize_mxfp4_activation(x)
        w13_pc = _with_activation_mx_scale(
            w13_pc,
            gemm1_scale,
        )
    else:
        gemm1_input = x

    num_ragged_rows = n_tokens * top_k
    with _maybe_lds_guard(gemm1_input, w13_weight, w13_pc), _maybe_sm12x_moe_tuning(
        gemm1_input, w13_weight, w13_pc, num_ragged_rows, which="GEMM1"
    ):
        intermediate_cache = matmul(
            gemm1_input,
            w13_weight,
            w13_bias,
            a_ragged_metadata=ragged_metadata,
            gather_indx=gather_indx,
            precision_config=w13_pc,
            fused_activation=act,
        )
    if act is None:
        intermediate_cache = _silu_gate_up(
            intermediate_cache,
            output_dtype=x.dtype,
            # sm12x_hybrid backs this path with the CUTLASS [up|gate] residency;
            # unset (default False) for the native concatenated [gate|up] path.
            gate_second=getattr(w, "hybrid_gate_up_swapped", False),
        )

    if hasattr(w, "w2_act_scale"):
        gemm2_input = fp8_quantize(intermediate_cache, scale=w.w2_act_scale)
    elif use_dynamic_mxfp4:
        gemm2_input, gemm2_scale = _quantize_mxfp4_activation(intermediate_cache)
        w2_pc = _with_activation_mx_scale(
            w2_pc,
            gemm2_scale,
        )
    else:
        gemm2_input = intermediate_cache

    with _maybe_lds_guard(gemm2_input, w2_weight, w2_pc), _maybe_sm12x_moe_tuning(
        gemm2_input, w2_weight, w2_pc, num_ragged_rows, which="GEMM2"
    ):
        output = matmul(
            gemm2_input,
            w2_weight,
            w2_bias,
            a_ragged_metadata=ragged_metadata,
            precision_config=w2_pc,
            scatter_indx=scatter_indx,
            gammas=gate_scal,
        )
    if top_k > 1:
        # _routing may pad the selection width to a power of two (e.g. top-6 -> 8)
        # for the triton_kernels topk/bitmatrix kernels, so the scattered output
        # carries that padded number of expert rows per token. Derive it from the
        # output rather than top_k. Padded slots carry gate weight 0, so summing
        # over the padded width is exact.
        # 0-token forwards (attention-DP idle ranks) must not divide by zero.
        effective_top_k = output.shape[0] // n_tokens if n_tokens else top_k
        return output.view(n_tokens, effective_top_k, output.shape[-1]).sum(dim=1)
    return output
