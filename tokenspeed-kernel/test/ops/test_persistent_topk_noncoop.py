"""Non-cooperative persistent_topk on <128KB-smem parts (GB10 / consumer Blackwell).

The cooperative-radix "large" path (seq_len > RADIX_THRESHOLD) needs all
ctas_per_group CTAs co-resident for a spin-wait barrier; past num_sms*occupancy
it falls back to FilteredTopK, which needs >=128KB smem/block -- a hard fail at
~400K context on ~99KB-smem parts. The non-cooperative port routes every row
through a single barrier-free CTA (2-pass finer-histogram top-k with a
streaming-radix overflow fallback), which must be EXACT (identical value
multiset to torch.topk; ties at the pivot broken arbitrarily) at depths well
past the old ceiling.

On >=128KB-smem devices (Hopper, datacenter Blackwell) the noncoop gate never
fires and this test simply exercises the unchanged cooperative path.
"""

import unittest

import torch


def _has_kernel() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        from tokenspeed_kernel.thirdparty.cuda.deepseek_v4_attention import (
            has_persistent_topk,
        )

        return has_persistent_topk()
    except Exception:
        return False


@unittest.skipUnless(_has_kernel(), "requires CUDA + prebuilt persistent_topk")
class TestPersistentTopKNoncoop(unittest.TestCase):
    def _run_case(self, rows: int, seq: int, topk: int) -> None:
        from tokenspeed_kernel.thirdparty.cuda.deepseek_v4_attention import (
            persistent_topk,
        )

        torch.manual_seed(seq * 31 + topk)
        logits = torch.randn(rows, seq, device="cuda", dtype=torch.float32)
        lengths = torch.full((rows,), seq, device="cuda", dtype=torch.int32)
        out = torch.full((rows, topk), -1, device="cuda", dtype=torch.int32)
        ws = torch.zeros(4 * 1024 * 1024, device="cuda", dtype=torch.uint8)
        persistent_topk(logits, lengths, out, ws, topk, seq)
        torch.cuda.synchronize()
        ref = torch.topk(logits, topk, dim=1).indices
        for r in range(rows):
            got_vals = logits[r, out[r].long()].sort(descending=True).values
            ref_vals = logits[r, ref[r]].sort(descending=True).values
            torch.testing.assert_close(
                got_vals,
                ref_vals,
                atol=0.0,
                rtol=0.0,
                msg=f"row {r} seq={seq} topk={topk}: value multiset differs",
            )

    def test_long_context_past_the_cooperative_ceiling(self):
        """450K/655K previously hard-failed on GB10 (total_ctas > num_sms*occ,
        FilteredTopK fallback unallocatable at ~99KB smem)."""
        for seq in (450_560, 655_360):
            for topk in (512, 2048):
                self._run_case(rows=2, seq=seq, topk=topk)

    def test_mid_context_exactness(self):
        """131K/262K: the noncoop selector's 2-pass path vs torch.topk."""
        for seq in (131_072, 262_144):
            for topk in (512, 2048):
                self._run_case(rows=2, seq=seq, topk=topk)

    def test_short_rows_unchanged(self):
        """<= RADIX_THRESHOLD rows take the pre-existing selectors."""
        for seq in (4_096, 30_000):
            self._run_case(rows=4, seq=seq, topk=512)
