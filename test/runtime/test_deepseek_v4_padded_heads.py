import os
import sys
import unittest
from unittest.mock import patch

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, suite="runtime-1gpu")

import tokenspeed.runtime.layers.attention.backends.deepseek_v4 as ds4_backend
from tokenspeed.runtime.layers.attention.backends.deepseek_v4 import (
    _fi_prefill_tile_heads,
)
from tokenspeed.runtime.models.deepseek_v4 import _deepseek_v4_padded_heads


class TestDeepseekV4PaddedHeads(unittest.TestCase):
    """Head-padding policy for the sparse-MLA query.

    Model-level ``padded_heads`` keeps the upstream FlashMLA tile rule
    (pad to 64/128) -- it sizes ``attn_sink`` and the DECODE path, where the
    FlashInfer split-K decode at reduced head counts IMAs under concurrent
    ragged batches (2026-07-09 arthur bisect). Only the sm12x FI PREFILL call
    drops to the nearest supported tile in {16, 32, 64, 128} via
    ``_fi_prefill_tile_heads``.
    """

    def test_model_padded_heads_keeps_flashmla_rule(self):
        self.assertEqual(_deepseek_v4_padded_heads(16), 64)
        self.assertEqual(_deepseek_v4_padded_heads(32), 64)
        self.assertEqual(_deepseek_v4_padded_heads(64), 64)
        self.assertEqual(_deepseek_v4_padded_heads(65), 128)
        self.assertEqual(_deepseek_v4_padded_heads(128), 128)
        with self.assertRaises(ValueError):
            _deepseek_v4_padded_heads(129)

    def test_fi_prefill_tile_ladder_when_route_active(self):
        with patch.object(
            ds4_backend, "_use_flashinfer_sparse_mla_prefill_sm12x", return_value=True
        ):
            self.assertEqual(_fi_prefill_tile_heads(8, 64), 16)
            self.assertEqual(_fi_prefill_tile_heads(16, 64), 16)
            self.assertEqual(_fi_prefill_tile_heads(17, 64), 32)
            self.assertEqual(_fi_prefill_tile_heads(32, 64), 32)
            self.assertEqual(_fi_prefill_tile_heads(33, 64), 64)
            self.assertEqual(_fi_prefill_tile_heads(64, 64), 64)
            self.assertEqual(_fi_prefill_tile_heads(96, 128), 128)
            # Never exceeds the model-level pad.
            self.assertEqual(_fi_prefill_tile_heads(96, 64), 64)

    def test_fi_prefill_tile_passthrough_when_route_inactive(self):
        with patch.object(
            ds4_backend, "_use_flashinfer_sparse_mla_prefill_sm12x", return_value=False
        ):
            self.assertEqual(_fi_prefill_tile_heads(32, 64), 64)
            self.assertEqual(_fi_prefill_tile_heads(96, 128), 128)


if __name__ == "__main__":
    unittest.main()
