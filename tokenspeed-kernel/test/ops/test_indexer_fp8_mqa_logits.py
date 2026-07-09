"""Tests for the FP8 indexer MQA-logits Triton kernel (sm12x).

Reference: fp32 torch over the dequantized inputs, same compact layout as
``indexer_mqa_logits_sm12x`` (the MXFP4 sibling / torch reference).
"""

import pytest
import torch

from tokenspeed_kernel.ops.attention.triton.indexer_fp8_mqa_logits_sm12x import (
    indexer_fp8_mqa_logits_sm12x_triton,
)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)

HEAD_DIM = 128


def _make_inputs(num_q, num_heads, num_kv, *, seed, device):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(num_q, num_heads, HEAD_DIM, generator=gen) * 2.0
    k = torch.randn(num_kv, HEAD_DIM, generator=gen)

    q_amax = q.abs().amax(dim=-1).clamp(min=1e-6)  # [num_q, H]
    q_scales = (q_amax / 448.0).float()
    q_fp8 = (q / q_scales[..., None]).to(torch.float8_e4m3fn)

    k_amax = k.abs().amax(dim=-1).clamp(min=1e-6)  # [num_kv]
    k_scales = (k_amax / 448.0).float()
    k_fp8 = (k / k_scales[:, None]).to(torch.float8_e4m3fn)

    weights = torch.rand(num_q, num_heads, generator=gen).float() + 0.1
    return (
        q_fp8.to(device),
        q_scales.to(device),
        k_fp8.to(device),
        k_scales.to(device),
        weights.to(device),
    )


def _reference(q_fp8, q_scales, k_fp8, k_scales, weights, ks, ke, max_len):
    # Dequantized fp32 reference in the same compact layout.
    num_q = q_fp8.shape[0]
    q = q_fp8.float() * q_scales[..., None]  # [num_q, H, D]
    k = k_fp8.float() * k_scales[:, None]  # [num_kv, D]
    out = torch.full(
        (num_q, max_len), float("-inf"), dtype=torch.float32, device=q.device
    )
    for m in range(num_q):
        s, e = int(ks[m]), int(ke[m])
        length = min(e - s, max_len)
        if length <= 0:
            continue
        scores = torch.einsum("hd,nd->hn", q[m], k[s : s + length])  # [H, L]
        out[m, :length] = (scores.clamp(min=0.0) * weights[m][:, None]).sum(dim=0)
    return out


def _windows(num_q, num_kv, *, kind, device):
    if kind == "full":
        ks = torch.zeros(num_q, dtype=torch.int32)
        ke = torch.full((num_q,), num_kv, dtype=torch.int32)
    elif kind == "causal":
        pos = torch.linspace(1, num_kv, num_q).round().to(torch.int32)
        ks = torch.zeros(num_q, dtype=torch.int32)
        ke = pos
    else:  # sliding
        pos = torch.linspace(1, num_kv, num_q).round().to(torch.int32)
        ks = (pos - 96).clamp(min=0)
        ke = pos
    return ks.to(device), ke.to(device)


@requires_cuda
@pytest.mark.parametrize("kind", ["full", "causal", "sliding"])
def test_matches_fp32_reference(kind):
    device = torch.device("cuda")
    num_q, num_heads, num_kv = 65, 64, 517  # ragged vs BLOCK_N=128
    q_fp8, q_scales, k_fp8, k_scales, weights = _make_inputs(
        num_q, num_heads, num_kv, seed=1234, device=device
    )
    ks, ke = _windows(num_q, num_kv, kind=kind, device=device)
    max_len = int((ke - ks).max())

    got = indexer_fp8_mqa_logits_sm12x_triton(
        q_fp8, q_scales, k_fp8, k_scales, weights, ks, ke, max_len
    )
    ref = _reference(q_fp8, q_scales, k_fp8, k_scales, weights, ks, ke, max_len)

    inf_mask = torch.isinf(ref)
    assert torch.equal(torch.isinf(got), inf_mask)
    # tf32 dot vs fp32 reference over D=128 contraction.
    torch.testing.assert_close(
        got[~inf_mask], ref[~inf_mask], rtol=2e-2, atol=2e-2
    )


@requires_cuda
def test_prefolded_q_scales_equivalent():
    device = torch.device("cuda")
    q_fp8, q_scales, k_fp8, k_scales, weights = _make_inputs(
        33, 64, 260, seed=99, device=device
    )
    ks, ke = _windows(33, 260, kind="causal", device=device)
    max_len = int((ke - ks).max())

    in_kernel = indexer_fp8_mqa_logits_sm12x_triton(
        q_fp8, q_scales, k_fp8, k_scales, weights, ks, ke, max_len
    )
    prefolded = indexer_fp8_mqa_logits_sm12x_triton(
        q_fp8, None, k_fp8, k_scales, weights * q_scales, ks, ke, max_len
    )
    torch.testing.assert_close(in_kernel, prefolded, rtol=1e-6, atol=1e-6)


@requires_cuda
def test_empty_and_degenerate_windows():
    device = torch.device("cuda")
    q_fp8, q_scales, k_fp8, k_scales, weights = _make_inputs(
        4, 64, 64, seed=7, device=device
    )
    # Row 0: empty window (ks == ke); rows 1-3 normal.
    ks = torch.tensor([10, 0, 5, 60], dtype=torch.int32, device=device)
    ke = torch.tensor([10, 30, 6, 64], dtype=torch.int32, device=device)
    max_len = 30

    got = indexer_fp8_mqa_logits_sm12x_triton(
        q_fp8, q_scales, k_fp8, k_scales, weights, ks, ke, max_len
    )
    ref = _reference(q_fp8, q_scales, k_fp8, k_scales, weights, ks, ke, max_len)
    assert torch.isinf(got[0]).all()
    inf_mask = torch.isinf(ref)
    assert torch.equal(torch.isinf(got), inf_mask)
    torch.testing.assert_close(got[~inf_mask], ref[~inf_mask], rtol=2e-2, atol=2e-2)

    # Zero queries / zero kv early-outs.
    empty = indexer_fp8_mqa_logits_sm12x_triton(
        q_fp8[:0], None, k_fp8, k_scales, weights[:0], ks[:0], ke[:0], 8
    )
    assert empty.shape == (0, 8)


@requires_cuda
def test_topk_selection_matches_reference():
    # The kernel exists to feed top-k selection: verify selected sets, not
    # just logits, on a long-ish causal layout (tf32 jitter must not perturb
    # top-512 membership beyond boundary ties).
    device = torch.device("cuda")
    num_q, num_kv, topk = 16, 4096, 512
    q_fp8, q_scales, k_fp8, k_scales, weights = _make_inputs(
        num_q, 64, num_kv, seed=42, device=device
    )
    ks = torch.zeros(num_q, dtype=torch.int32, device=device)
    ke = torch.full((num_q,), num_kv, dtype=torch.int32, device=device)

    got = indexer_fp8_mqa_logits_sm12x_triton(
        q_fp8, q_scales, k_fp8, k_scales, weights, ks, ke, num_kv
    )
    ref = _reference(q_fp8, q_scales, k_fp8, k_scales, weights, ks, ke, num_kv)

    got_idx = got.topk(topk, dim=-1).indices
    ref_idx = ref.topk(topk, dim=-1).indices
    overlaps = []
    for m in range(num_q):
        inter = len(set(got_idx[m].tolist()) & set(ref_idx[m].tolist()))
        overlaps.append(inter / topk)
    overlap = min(overlaps)
    # Allow a small boundary-tie band; selection must be essentially identical.
    assert overlap > 0.98, f"worst top-{topk} set overlap {overlap:.4f}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
