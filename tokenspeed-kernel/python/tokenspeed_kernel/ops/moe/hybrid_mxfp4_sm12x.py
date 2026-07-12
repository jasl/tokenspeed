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

"""Hybrid MXFP4 MoE for consumer Blackwell (sm_120/sm_121).

The FlashInfer CUTLASS MXFP4 path wins on prefill (large token count, ~+14 %
ctx_pp) while the Triton MXFP4 path wins on decode (small token count, higher
tg). This solution keeps a SINGLE physical expert-weight residency (~44 GB/rank
cannot be duplicated) and dispatches per call by token count ``M``:

    M >= TOKENSPEED_MOE_SM12X_HYBRID_MIN_TOKENS  ->  CUTLASS body (prefill)
    M <  TOKENSPEED_MOE_SM12X_HYBRID_MIN_TOKENS  ->  Triton body  (decode)

The weight preprocessor builds the CUTLASS residency (raw row-major packed FP4
weights + CUTLASS 128x4 swizzled scales) and then adds, over the SAME weight
byte buffers, a zero-copy Triton ``StridedLayout`` wrap plus small Blackwell MX
scale copies (~1/16 of the weight size). Only scales are duplicated. Because the
CUTLASS layout stores w13 as [w3|w1] == [up|gate], the Triton GEMM1 emits
[up|gate]; the Triton branch runs its ``gate_second`` SiLU variant, which is
byte-identical to the native concatenated [gate|up] activation.
"""

from __future__ import annotations

import os

import torch
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

# Triton-layer helpers are always importable in this tree (triton/__init__ imports
# them unconditionally); reuse them so all triton_kernels knowledge stays there.
from tokenspeed_kernel.ops.moe.triton.mxfp4 import (
    MXFP4_BLOCK,
    FlexCtx,
    InFlexData,
    PrecisionConfig,
    _strided_mxfp4,
    triton_mxfp4_moe_apply,
)

platform = current_platform()

_MIN_TOKENS_ENV = "TOKENSPEED_MOE_SM12X_HYBRID_MIN_TOKENS"
_DEFAULT_MIN_TOKENS = 1024
_TRITON_NUM_WARPS = 8


def _hybrid_min_tokens() -> int:
    """Prefill/decode split point in tokens (env-overridable per deploy)."""
    raw = os.environ.get(_MIN_TOKENS_ENV, "")
    if raw.strip():
        try:
            value = int(raw)
            if value >= 0:
                return value
        except ValueError:
            pass
    return _DEFAULT_MIN_TOKENS


if platform.is_nvidia:
    from tokenspeed_kernel.ops.moe.flashinfer.cutlass_mxfp4 import (
        _reorder_w13,
        flashinfer_cutlass_mxfp4_moe_apply,
        flashinfer_cutlass_mxfp4_moe_weights,
    )

    def sm12x_hybrid_mxfp4_moe_weights(plan: dict, w: torch.nn.Module) -> None:
        """Build both residencies over one shared weight buffer.

        1. CUTLASS residency via the stock preprocessor: reorders w13 to
           [w3|w1]==[up|gate] raw bytes, swizzles scales to CUTLASS 128x4, sets
           global scales / swiglu params / sizes. This is what the prefill body
           reads and it owns ``w.w13_weight`` / ``w.w2_weight``.
        2. Triton residency: a zero-copy ``StridedLayout`` wrap of the SAME raw
           weight buffers plus Blackwell MX scale copies + PrecisionConfigs. This
           is what the decode body reads. The raw weights are NOT freed.
        """
        w13_layout = getattr(w, "w13_input_layout", "concatenated")
        if w13_layout != "concatenated":
            # Interleaved [g,u,g,u] routes through the fused-swiglu path, which
            # owns its own gate/up ordering; the gate_second swap would be wrong.
            raise NotImplementedError(
                "sm12x_hybrid MoE supports the concatenated w13 layout only, got "
                f"{w13_layout!r}"
            )
        # A real (non-None) expert bias would be reordered to [up|gate] by the
        # CUTLASS preprocessor while the Triton branch expects [gate|up] — refuse
        # that case. A None/absent bias (DeepSeek-V4 MoE, with_bias=False) is fine:
        # both bodies read it via getattr(..., None). `hasattr` alone is wrong
        # because the module may carry a None bias placeholder.
        if (
            getattr(w, "w13_weight_bias", None) is not None
            or getattr(w, "w2_weight_bias", None) is not None
        ):
            raise NotImplementedError(
                "sm12x_hybrid MoE does not support non-None expert biases"
            )

        # Capture the reordered LINEAR (unswizzled) scales in [up|gate] order
        # before the CUTLASS preprocessor overwrites them with the 128x4 swizzle.
        w13_scale_linear = _reorder_w13(w.w13_weight_scale.data, w13_layout, 1)
        w2_scale_linear = w.w2_weight_scale.data.contiguous()

        # --- CUTLASS residency (prefill) --------------------------------------
        # Reorders w.w13_weight to [up|gate] raw, swizzles scales, sets sizes.
        flashinfer_cutlass_mxfp4_moe_weights(plan=plan, w=w)

        # --- Triton residency (decode), zero-copy over the SAME raw buffers ----
        w13_tri, w13_flex, w13_scale_tri = _strided_mxfp4(
            w.w13_weight.data, w13_scale_linear, _TRITON_NUM_WARPS
        )
        w2_tri, w2_flex, w2_scale_tri = _strided_mxfp4(
            w.w2_weight.data, w2_scale_linear, _TRITON_NUM_WARPS
        )

        w.w13_precision_config = PrecisionConfig(
            flex_ctx=FlexCtx(lhs_data=InFlexData(), rhs_data=w13_flex),
            b_mx_scale=w13_scale_tri,
            b_microblock_size=MXFP4_BLOCK,
            out_dtype=None,
        )
        w.w2_precision_config = PrecisionConfig(
            flex_ctx=FlexCtx(lhs_data=InFlexData(), rhs_data=w2_flex),
            b_mx_scale=w2_scale_tri,
            b_microblock_size=MXFP4_BLOCK,
            out_dtype=None,
        )
        w.w13_weight_triton_tensor = w13_tri
        w.w2_weight_triton_tensor = w2_tri
        # CUTLASS w13 is [up|gate] -> Triton GEMM1 emits [up|gate] -> gate_second.
        w.hybrid_gate_up_swapped = True

    @register_kernel(
        "moe",
        "apply",
        name="sm12x_hybrid_mxfp4_moe_apply",
        solution="sm12x_hybrid",
        weight_preprocessor=sm12x_hybrid_mxfp4_moe_weights,
        capability=CapabilityRequirement(
            vendors=frozenset({"nvidia"}),
            min_arch_version=ArchVersion(12, 0),
            max_arch_version=ArchVersion(12, 1),
        ),
        signatures=format_signatures(
            "x",
            "dense",
            {torch.float16, torch.bfloat16},
        ),
        traits={
            "weight_dtype": frozenset({"mxfp4"}),
            "activation": frozenset({"swiglu"}),
            "routing_mode": frozenset({"precomputed_topk"}),
            "supports_deferred_finalize": frozenset({False}),
            "supports_ep": frozenset({True}),
            "supports_all_to_all_ep": frozenset({False}),
            # Large-M calls run the CUTLASS body, which needs 128-aligned ispp.
            "ispp_alignment": frozenset({128}),
            "internal_activation_dtype": frozenset({"input"}),
        },
        # Opt-in only (below CUTLASS PERFORMANT and the sm12x Triton SPECIALIZED
        # registrations); selected by forcing solution="sm12x_hybrid".
        priority=Priority.PORTABLE,
    )
    def sm12x_hybrid_mxfp4_moe_apply(
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
    ) -> torch.Tensor:
        """Dispatch by token count to the CUTLASS (prefill) or Triton (decode) body."""
        if x.shape[0] >= _hybrid_min_tokens():
            return flashinfer_cutlass_mxfp4_moe_apply(
                plan,
                x,
                w,
                router_logits,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                num_tokens_global=num_tokens_global,
                max_num_tokens_per_gpu=max_num_tokens_per_gpu,
                do_finalize=do_finalize,
                enable_pdl=enable_pdl,
            )
        return triton_mxfp4_moe_apply(
            plan,
            x,
            w,
            router_logits,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            num_tokens_global=num_tokens_global,
            max_num_tokens_per_gpu=max_num_tokens_per_gpu,
            do_finalize=do_finalize,
            enable_pdl=enable_pdl,
        )
