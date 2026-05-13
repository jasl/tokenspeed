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

from typing import TYPE_CHECKING

import torch
from tokenspeed_kernel.torch_compile import get_compiler_backend

from tokenspeed.runtime.utils import crash_on_warnings, get_colorful_logger

if TYPE_CHECKING:
    from tokenspeed.runtime.layers.logits_processor import LogitsProcessorOutput


logger = get_colorful_logger(__name__)


# Smallest positive value per dtype, used as the lower bound for `uniform_`
# draws that feed rejection-sampling kernels. A coin of exact 0 silently
# accepts a zero-probability draft in `chain_speculative_sampling_target_only`
# (the kernel condition `coin <= target_prob / threshold_acc` reduces to
# `0 <= 0`), so the coin must be strictly positive.
COIN_EPS = {
    torch.float32: torch.finfo(torch.float32).tiny,
    torch.bfloat16: torch.finfo(torch.bfloat16).tiny,
}


def coin_eps(dtype: torch.dtype) -> float:
    """Lower bound for uniform coin draws of the given dtype. See COIN_EPS."""
    return COIN_EPS[dtype]


def nan_guard_logits(
    logits: torch.Tensor,
    enable_nan_detection: bool,
) -> torch.Tensor:
    """Replace NaNs with -1e5 and optionally crash; no-op when detection is disabled."""
    if not enable_nan_detection:
        return logits.nan_to_num_()

    if not torch.any(torch.isnan(logits)):
        return logits

    logger.warning("Detected errors during sampling! NaN in the logits.")
    logits = torch.where(torch.isnan(logits), torch.full_like(logits, -1e5), logits)
    if crash_on_warnings():
        raise ValueError("Detected errors during sampling! NaN in the logits.")
    return logits


def write_output_logprobs(
    logits_output: LogitsProcessorOutput,
    logits: torch.Tensor,
    tokens: torch.Tensor,
    top_logprobs_nums: list[int] | None = None,
) -> None:
    """Fill output logprob tensors; callers gate on the enable flag."""
    raw_logprobs = torch.log_softmax(logits, dim=-1)
    logits_output.next_token_logprobs = raw_logprobs.gather(
        -1, tokens.long().unsqueeze(-1)
    ).squeeze(-1)
    write_output_top_logprobs(logits_output, raw_logprobs, top_logprobs_nums)


def write_output_top_logprobs(
    logits_output: LogitsProcessorOutput,
    raw_logprobs: torch.Tensor,
    top_logprobs_nums: list[int] | None,
) -> None:
    """Attach top-k logprobs from already-normalized output logprobs."""
    if not top_logprobs_nums:
        return

    num_rows = raw_logprobs.shape[0]
    if len(top_logprobs_nums) == num_rows:
        expanded_top_logprobs_nums = top_logprobs_nums
    elif num_rows % len(top_logprobs_nums) == 0:
        repeat = num_rows // len(top_logprobs_nums)
        expanded_top_logprobs_nums = [
            k for k in top_logprobs_nums for _ in range(repeat)
        ]
    else:
        raise ValueError(
            f"Cannot align top_logprobs_nums={top_logprobs_nums} "
            f"to {num_rows} output logprob rows."
        )

    max_k = max(expanded_top_logprobs_nums)
    if max_k <= 0:
        return

    top = raw_logprobs.topk(max_k, dim=-1)
    logits_output.next_token_top_logprobs_val = top.values
    logits_output.next_token_top_logprobs_idx = top.indices
    logits_output.next_token_top_logprobs_nums = expanded_top_logprobs_nums


@torch.compile(dynamic=True, backend=get_compiler_backend())
def top_p_normalize_probs_torch(
    probs: torch.Tensor,
    top_ps: torch.Tensor,
) -> torch.Tensor:
    """Pure-torch nucleus renorm — used by the prefill-logprob path."""
    probs_sort, probs_idx = probs.sort(dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    probs_sort[(probs_sum - probs_sort) > top_ps.view(-1, 1)] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    return torch.zeros_like(probs_sort).scatter_(-1, probs_idx, probs_sort)
