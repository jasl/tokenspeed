import os
import sys
import unittest
from unittest.mock import patch

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, suite="runtime-1gpu")

import tokenspeed.runtime.models.deepseek_v4 as ds4_models
from tokenspeed.runtime.models.deepseek_v4 import _deepseek_v4_padded_heads


class TestDeepseekV4PaddedHeads(unittest.TestCase):
    """Padded-head tile selection for the sparse-MLA query.

    FlashMLA tiles need 64 padded heads; FlashInfer's SM120 packed sparse-MLA
    dispatch accepts {16, 32, 64, 128}, so consumer Blackwell pads to the
    nearest supported tile instead (TP=2 keeps its 32 local heads on the
    32-head tile rather than zero-padding to 64).
    """

    def test_flashmla_route_pads_to_64_minimum(self):
        with patch.object(
            ds4_models, "_use_flashinfer_sparse_mla_sm120", return_value=False
        ):
            self.assertEqual(_deepseek_v4_padded_heads(16), 64)
            self.assertEqual(_deepseek_v4_padded_heads(32), 64)
            self.assertEqual(_deepseek_v4_padded_heads(64), 64)
            self.assertEqual(_deepseek_v4_padded_heads(65), 128)
            self.assertEqual(_deepseek_v4_padded_heads(128), 128)
            with self.assertRaises(ValueError):
                _deepseek_v4_padded_heads(129)

    def test_fi_sm120_route_uses_supported_tile_ladder(self):
        with patch.object(
            ds4_models, "_use_flashinfer_sparse_mla_sm120", return_value=True
        ):
            self.assertEqual(_deepseek_v4_padded_heads(8), 16)
            self.assertEqual(_deepseek_v4_padded_heads(16), 16)
            self.assertEqual(_deepseek_v4_padded_heads(17), 32)
            self.assertEqual(_deepseek_v4_padded_heads(32), 32)
            self.assertEqual(_deepseek_v4_padded_heads(33), 64)
            self.assertEqual(_deepseek_v4_padded_heads(64), 64)
            self.assertEqual(_deepseek_v4_padded_heads(65), 128)
            self.assertEqual(_deepseek_v4_padded_heads(128), 128)
            with self.assertRaises(ValueError):
                _deepseek_v4_padded_heads(129)


if __name__ == "__main__":
    unittest.main()
