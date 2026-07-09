import os
import sys
import unittest

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, suite="runtime-1gpu")

import torch

from tokenspeed.runtime.layers.deepseek_v4_mhc import _tf32_hc_prenorm_gemm_triton


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestTf32HcPrenormGemmTriton(unittest.TestCase):
    """Fused split-K prenorm GEMM + sum-of-squares vs the fp32 torch reference.

    The GEMM accumulates tf32 products (10-bit mantissa) into fp32, so it gets
    a tf32-appropriate tolerance; the sum-of-squares path is pure fp32 and is
    held tight.
    """

    def _run(self, num_tokens: int, n_splits: int) -> None:
        torch.manual_seed(0x5EED + num_tokens + n_splits)
        hc_mult, hidden_size = 4, 2560
        hc_hidden = hc_mult * hidden_size
        hc_mult3 = hc_mult * 2 + hc_mult * hc_mult
        device = torch.device("cuda")

        x = torch.randn(num_tokens, hc_hidden, dtype=torch.bfloat16, device=device)
        fn = torch.randn(hc_mult3, hc_hidden, dtype=torch.float32, device=device)
        out = torch.empty(
            n_splits, num_tokens, hc_mult3, dtype=torch.float32, device=device
        )
        sqrsum = torch.empty(n_splits, num_tokens, dtype=torch.float32, device=device)

        _tf32_hc_prenorm_gemm_triton(x, fn, out, sqrsum, n_splits)

        x32 = x.float()
        ref_out = x32 @ fn.t()
        ref_sq = (x32 * x32).sum(dim=1)

        got_out = out.sum(dim=0)
        got_sq = sqrsum.sum(dim=0)

        torch.testing.assert_close(got_out, ref_out, rtol=2e-2, atol=2e-1)
        torch.testing.assert_close(got_sq, ref_sq, rtol=1e-3, atol=1e-1)

    def test_single_split(self):
        self._run(num_tokens=333, n_splits=1)

    def test_multi_split(self):
        self._run(num_tokens=37, n_splits=3)

    def test_split_not_dividing_k_blocks(self):
        # K = 10240, split_k = ceil(K/7) exercises the ragged split tail.
        self._run(num_tokens=64, n_splits=7)

    def test_empty_tokens_noop(self):
        device = torch.device("cuda")
        x = torch.empty(0, 10240, dtype=torch.bfloat16, device=device)
        fn = torch.randn(24, 10240, dtype=torch.float32, device=device)
        out = torch.empty(1, 0, 24, dtype=torch.float32, device=device)
        sqrsum = torch.empty(1, 0, dtype=torch.float32, device=device)
        _tf32_hc_prenorm_gemm_triton(x, fn, out, sqrsum, 1)


if __name__ == "__main__":
    unittest.main()
