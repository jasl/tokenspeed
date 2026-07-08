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

"""Communication ops for distributed communication.

All ops require explicit group (tuple of ranks) and rank parameters.
Groups are looked up from pg_manager internally via comm_backend.
"""

from dataclasses import dataclass
from enum import IntEnum

import torch
import torch.distributed
from tokenspeed_kernel.ops.communication.trtllm import (
    allgather_dual_rmsnorm,
    allreduce_residual_rmsnorm,
    reducescatter_residual_rmsnorm,
)

from tokenspeed.runtime.distributed.comm_backend import (
    CommBackend,
    Group,
    get_global_backend,
)
from tokenspeed.runtime.distributed.process_group_manager import (
    process_group_manager as pg_manager,
)
from tokenspeed.runtime.utils.pdl import pdl_enabled


def _get_process_group(group: Group):
    return pg_manager.get_process_group("nccl", group)


# ---------------------------------------------------------------------------
# Fusion parameters
# ---------------------------------------------------------------------------


class FusionOp(IntEnum):
    """What post-communication fusion to apply."""

    NONE = 0
    # all_reduce + residual_add + RMSNorm
    RESIDUAL_RMS_NORM = 1
    # reduce_scatter + residual_add + RMSNorm
    RS_RESIDUAL_RMS_NORM = 2
    # all_gather + dual RMSNorm (for MLA)
    AG_DUAL_RMS_NORM = 3


@dataclass
class FusionParams:
    """Optional fusion context passed to fused comm_ops functions.

    Not all fields are used by every ``FusionOp``. Only the relevant
    subset is accessed.
    """

    fusion_op: FusionOp = FusionOp.NONE

    # --- For RESIDUAL_RMS_NORM / RS_RESIDUAL_RMS_NORM ---
    residual: torch.Tensor | None = None
    norm_weight: torch.Tensor | None = None
    eps: float = 1e-6

    # --- For AG_DUAL_RMS_NORM ---
    norm_weight_2: torch.Tensor | None = None
    eps_2: float = 1e-6

    # --- For reduce-scatter fusion ---
    add_in: torch.Tensor | None = None
    residual_reduce_scattered: bool = False
    has_partial_norm_out: bool = False

    # --- Shared by RESIDUAL_RMS_NORM / RS_RESIDUAL_RMS_NORM / AG_DUAL_RMS_NORM ---
    max_token_num: int = 0

    # --- For FP8 block quantization ---
    block_quant_fp8: bool = False

    # --- General ---
    total_num_tokens: int = 0
    trigger_completion_at_end: bool = False
    fp32_acc: bool = False
    max_sm_to_use: int | None = None


# ---------------------------------------------------------------------------
# Basic primitives
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Piecewise decode CUDA graph seam
# ---------------------------------------------------------------------------
#
# When the decode forward is captured as a piecewise CUDA graph (compute ->
# comm -> compute), collectives must NOT be recorded into the graph. During
# segmented capture the active collective-mode BreakableCapture splits the
# graph here: the eager collective closure is recorded (and run once, at
# capture) via add_eager to re-run between graph replays, and the persistent
# buffer the next segment reads is returned.
#
# _NO_BREAK means "not captured / no collective-mode capture -> run the
# collective normally as today", so the flag-off path (and the breakable
# PREFILL graph, which captures its collectives) is unchanged.

_NO_BREAK = object()


def _active_collective_mode_capture():
    """The in-flight collective-mode BreakableCapture, or None.

    The import is local to avoid a module-load circular import
    (cuda_graph_wrapper imports the sampling backend, which pulls in comm)."""
    from tokenspeed.runtime.execution.breakable_cuda_graph import BreakableCapture
    from tokenspeed.runtime.execution.cuda_graph_wrapper import get_is_capture_mode

    if not get_is_capture_mode():
        return None
    cap = BreakableCapture.current()
    if cap is not None and cap._capturing and cap.break_at_collectives:
        return cap
    return None


def _maybe_record_piecewise_break(op_name, eager_fn, output_tensor):
    """If a piecewise decode graph is being captured, record ``eager_fn`` as a
    segment break and return ``output_tensor`` (the persistent buffer the next
    segment reads). Otherwise return ``_NO_BREAK`` so the caller runs the
    collective normally."""
    cap = _active_collective_mode_capture()
    if cap is not None:
        cap.add_eager(eager_fn)
        return output_tensor
    return _NO_BREAK


def all_reduce(
    tensor: torch.Tensor,
    group: Group,
    backend: CommBackend | None = None,
    op: torch.distributed.ReduceOp = torch.distributed.ReduceOp.SUM,
) -> torch.Tensor:
    """All-reduce the tensor across the given communication group."""
    if backend is None:
        backend = get_global_backend()
    # Piecewise decode: run the all-reduce eagerly between graph segments.
    # in-place -> output buffer is the same tensor the next segment reads.
    rec = _maybe_record_piecewise_break(
        "AR",
        lambda t=tensor, g=group, o=op: backend.all_reduce(t, g, op=o),
        tensor,
    )
    if rec is not _NO_BREAK:
        return rec
    return backend.all_reduce(tensor, group, op=op)


def all_gather(
    tensor: torch.Tensor,
    group: Group,
    dim: int = -1,
    backend: CommBackend | None = None,
) -> torch.Tensor:
    """All-gather the tensor across the given communication group."""
    if backend is None:
        backend = get_global_backend()
    return backend.all_gather(tensor, group, dim)


def all_gather_into_tensor(
    output: torch.Tensor,
    input: torch.Tensor,
    group: Group,
    backend: CommBackend | None = None,
) -> None:
    """All-gather input into a pre-allocated output buffer."""
    if backend is None:
        backend = get_global_backend()
    # Piecewise decode: writes into the caller-owned ``output`` buffer, which
    # the next graph segment reads. Void return, so just return after
    # recording the eager collective.
    rec = _maybe_record_piecewise_break(
        "AGT",
        lambda o=output, i=input, g=group: backend.all_gather_into_tensor(o, i, g),
        output,
    )
    if rec is not _NO_BREAK:
        return
    backend.all_gather_into_tensor(output, input, group)


def reduce_scatter(
    tensor: torch.Tensor,
    group: Group,
    backend: CommBackend | None = None,
) -> torch.Tensor:
    """Reduce-scatter the tensor across the given communication group."""
    if backend is None:
        backend = get_global_backend()
    return backend.reduce_scatter(tensor, group)


def all_to_all_single(
    output: torch.Tensor,
    input: torch.Tensor,
    group: Group,
    backend: CommBackend | None = None,
) -> None:
    """Even-split all_to_all into a pre-allocated output buffer."""
    if backend is None:
        backend = get_global_backend()
    backend.all_to_all_single(output, input, group)


# ---------------------------------------------------------------------------
# Fused ops (comm + residual + norm)
# ---------------------------------------------------------------------------


def fused_all_reduce(
    tensor: torch.Tensor,
    rank: int,
    group: Group,
    backend: CommBackend | None = None,
    fusion_params: FusionParams | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """All-reduce with optional fused residual + RMSNorm."""
    if backend is None:
        backend = get_global_backend()

    if fusion_params is None or fusion_params.fusion_op == FusionOp.NONE:
        return backend.all_reduce(tensor, group)

    if fusion_params.fusion_op == FusionOp.RESIDUAL_RMS_NORM:
        return allreduce_residual_rmsnorm(
            input_tensor=tensor,
            residual=fusion_params.residual,
            weight=fusion_params.norm_weight,
            rank=rank,
            group=_get_process_group(group),
            eps=fusion_params.eps,
            fp32_acc=fusion_params.fp32_acc,
            block_quant_fp8=fusion_params.block_quant_fp8,
            residual_reduce_scattered=fusion_params.residual_reduce_scattered,
            has_partial_norm_out=fusion_params.has_partial_norm_out,
            trigger_completion_at_end=fusion_params.trigger_completion_at_end,
            max_sm_to_use=fusion_params.max_sm_to_use,
            launch_with_pdl=pdl_enabled(),
        )

    raise ValueError(
        f"Unsupported fusion_op {fusion_params.fusion_op} for fused_all_reduce"
    )


def fused_reduce_scatter(
    tensor: torch.Tensor,
    rank: int,
    group: Group,
    backend: CommBackend | None = None,
    fusion_params: FusionParams | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Reduce-scatter with optional fused residual + RMSNorm."""
    if backend is None:
        backend = get_global_backend()

    if fusion_params is None or fusion_params.fusion_op == FusionOp.NONE:
        return backend.reduce_scatter(tensor, group)

    if fusion_params.fusion_op == FusionOp.RS_RESIDUAL_RMS_NORM:
        return reducescatter_residual_rmsnorm(
            input_tensor=tensor,
            weight=fusion_params.norm_weight,
            residual=fusion_params.residual,
            eps=fusion_params.eps,
            rank=rank,
            group=_get_process_group(group),
            add_in=fusion_params.add_in,
            fp32_acc=fusion_params.fp32_acc,
            block_quant_fp8=fusion_params.block_quant_fp8,
            max_token_num=fusion_params.max_token_num or tensor.shape[0],
            launch_with_pdl=pdl_enabled(),
        )

    raise ValueError(
        f"Unsupported fusion_op {fusion_params.fusion_op} for fused_reduce_scatter"
    )


def fused_all_gather(
    tensor: torch.Tensor,
    rank: int,
    group: Group,
    dim: int = -1,
    backend: CommBackend | None = None,
    fusion_params: FusionParams | None = None,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """All-gather with optional fused dual-RMSNorm."""
    if backend is None:
        backend = get_global_backend()

    if fusion_params is None or fusion_params.fusion_op == FusionOp.NONE:
        return backend.all_gather(tensor, group, dim)

    if fusion_params.fusion_op == FusionOp.AG_DUAL_RMS_NORM:
        return allgather_dual_rmsnorm(
            qkv=tensor,
            weight_q_a=fusion_params.norm_weight,
            eps_q=fusion_params.eps,
            weight_kv_a=fusion_params.norm_weight_2,
            eps_kv=fusion_params.eps_2,
            rank=rank,
            group=_get_process_group(group),
            total_num_tokens=fusion_params.total_num_tokens,
            max_token_num=fusion_params.max_token_num
            or max(tensor.shape[0], fusion_params.total_num_tokens),
            fp32_acc=fusion_params.fp32_acc,
            block_quant_fp8=fusion_params.block_quant_fp8,
            launch_with_pdl=pdl_enabled(),
        )

    raise ValueError(
        f"Unsupported fusion_op {fusion_params.fusion_op} for fused_all_gather"
    )


# ---------------------------------------------------------------------------
# Token-aware ops (uneven token distribution via TritonRSAG)
# ---------------------------------------------------------------------------


def _reject_allocating_collective_under_piecewise(name: str) -> None:
    """Allocating token-aware collectives (TritonRSAG) return a freshly
    allocated buffer, so the next graph segment has no stable buffer to read.
    On the supported piecewise config (TP=2 / EP off / DP=1) use_all_reduce is
    True everywhere and these never fire; if one does, fail LOUD rather than
    silently hang or corrupt."""
    if _active_collective_mode_capture() is not None:
        raise RuntimeError(
            f"piecewise decode graph: unexpected allocating collective {name} "
            "— not supported; disable --piecewise-decode-cudagraph or file a bug"
        )


def token_all_gather(
    tensor: torch.Tensor,
    group: Group,
    scattered_num_tokens: list[int],
    backend=None,
) -> torch.Tensor:
    """All-gather with token-aware distribution (TritonRSAG).

    Args:
        scattered_num_tokens: Number of tokens on each rank in the group,
            e.g. [50, 50, 51, 49] for 4 ranks with 200 total tokens.
    """
    if backend is None:
        backend = get_global_backend()
    _reject_allocating_collective_under_piecewise("token_all_gather")
    return backend.token_all_gather(tensor, group, scattered_num_tokens)


def token_reduce_scatter(
    tensor: torch.Tensor,
    group: Group,
    scattered_num_tokens: list[int],
    backend=None,
) -> torch.Tensor:
    """Reduce-scatter with token-aware distribution (TritonRSAG).

    Args:
        scattered_num_tokens: Number of tokens on each rank in the group,
            e.g. [50, 50, 51, 49] for 4 ranks with 200 total tokens.
    """
    if backend is None:
        backend = get_global_backend()
    _reject_allocating_collective_under_piecewise("token_reduce_scatter")
    return backend.token_reduce_scatter(tensor, group, scattered_num_tokens)
