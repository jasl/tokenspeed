# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.layers.moe import utils as moe_utils
from tokenspeed.runtime.layers.moe.utils import MoeBackend
from tokenspeed.runtime.models.deepseek_v4 import (
    DeepseekV4MegaMoEExperts,
    DeepseekV4MoE,
    DeepseekV4Sm12xMoEExperts,
)
from tokenspeed.runtime.utils.env import global_server_args_dict


def _ep2_mapping() -> Mapping:
    return Mapping(
        rank=0,
        world_size=2,
        attn_tp_size=2,
        attn_cp_size=1,
        attn_dp_size=1,
        dense_tp_size=1,
        dense_dp_size=2,
        moe_tp_size=1,
        moe_ep_size=2,
        moe_dp_size=1,
    )


def _moe_config() -> SimpleNamespace:
    return SimpleNamespace(
        n_shared_experts=None,
        routed_scaling_factor=1.0,
        scoring_func="sqrtsoftplus",
        num_hash_layers=0,
        vocab_size=16,
        n_routed_experts=4,
        hidden_size=128,
        moe_intermediate_size=128,
        hidden_act="swiglu",
        num_experts_per_tok=2,
        norm_topk_prob=True,
        topk_method="noaux_tc",
        swiglu_limit=7.0,
    )


class TestDeepseekV4MegaMoE(unittest.TestCase):
    def test_weight_loader_places_expert_shards(self):
        experts = DeepseekV4MegaMoEExperts(
            num_experts=4,
            num_local_experts=2,
            top_k=2,
            hidden_size=128,
            intermediate_size=128,
            mapping=None,
            prefix="layers.0.ffn.experts",
        )

        w1 = torch.full((128, 64), 1, dtype=torch.uint8)
        w3 = torch.full((128, 64), 3, dtype=torch.uint8)
        w2 = torch.full((128, 64), 2, dtype=torch.uint8)
        s1 = torch.full((128, 4), 11, dtype=torch.uint8)
        s3 = torch.full((128, 4), 13, dtype=torch.uint8)
        s2 = torch.full((128, 4), 12, dtype=torch.uint8)

        experts.weight_loader(experts.w13_weight, w1, "w1", local_expert_id=1)
        experts.weight_loader(experts.w13_weight, w3, "w3", local_expert_id=1)
        experts.weight_loader(experts.w2_weight, w2, "w2", local_expert_id=1)
        experts.weight_loader(experts.w13_weight_scale, s1, "w1", local_expert_id=1)
        experts.weight_loader(experts.w13_weight_scale, s3, "w3", local_expert_id=1)
        experts.weight_loader(experts.w2_weight_scale, s2, "w2", local_expert_id=1)

        torch.testing.assert_close(experts.w13_weight[1, :128], w1)
        torch.testing.assert_close(experts.w13_weight[1, 128:], w3)
        torch.testing.assert_close(experts.w2_weight[1], w2)
        torch.testing.assert_close(experts.w13_weight_scale[1, :128], s1)
        torch.testing.assert_close(experts.w13_weight_scale[1, 128:], s3)
        torch.testing.assert_close(experts.w2_weight_scale[1], s2)


class TestDeepseekV4Sm12xMoE(unittest.TestCase):
    def test_weight_loader_places_expert_shards_and_biases(self):
        experts = DeepseekV4Sm12xMoEExperts(
            num_experts=4,
            num_local_experts=2,
            top_k=2,
            hidden_size=128,
            intermediate_size=128,
            mapping=_ep2_mapping(),
            prefix="layers.0.ffn.experts",
        )

        w1 = torch.full((128, 64), 1, dtype=torch.uint8)
        w3 = torch.full((128, 64), 3, dtype=torch.uint8)
        w2 = torch.full((128, 64), 2, dtype=torch.uint8)
        s1 = torch.full((128, 4), 11, dtype=torch.uint8)
        s3 = torch.full((128, 4), 13, dtype=torch.uint8)
        s2 = torch.full((128, 4), 12, dtype=torch.uint8)
        b1 = torch.full((128,), 0.25, dtype=torch.bfloat16)
        b3 = torch.full((128,), 0.75, dtype=torch.bfloat16)
        b2 = torch.full((128,), 0.5, dtype=torch.bfloat16)

        experts.weight_loader(experts.w13_weight, w1, "w1", local_expert_id=1)
        experts.weight_loader(experts.w13_weight, w3, "w3", local_expert_id=1)
        experts.weight_loader(experts.w2_weight, w2, "w2", local_expert_id=1)
        experts.weight_loader(experts.w13_weight_scale, s1, "w1", local_expert_id=1)
        experts.weight_loader(experts.w13_weight_scale, s3, "w3", local_expert_id=1)
        experts.weight_loader(experts.w2_weight_scale, s2, "w2", local_expert_id=1)
        experts.weight_loader(experts.w13_weight_bias, b1, "w1", local_expert_id=1)
        experts.weight_loader(experts.w13_weight_bias, b3, "w3", local_expert_id=1)
        experts.weight_loader(experts.w2_weight_bias, b2, "w2", local_expert_id=1)

        torch.testing.assert_close(experts.w13_weight[1, :128], w1)
        torch.testing.assert_close(experts.w13_weight[1, 128:], w3)
        torch.testing.assert_close(experts.w2_weight[1], w2)
        torch.testing.assert_close(experts.w13_weight_scale[1, :128], s1)
        torch.testing.assert_close(experts.w13_weight_scale[1, 128:], s3)
        torch.testing.assert_close(experts.w2_weight_scale[1], s2)
        torch.testing.assert_close(experts.w13_weight_bias[1, :128], b1)
        torch.testing.assert_close(experts.w13_weight_bias[1, 128:], b3)
        torch.testing.assert_close(experts.w2_weight_bias[1], b2)

        experts.finalize_weights()

        self.assertEqual(experts.w13_weight_bias.dtype, torch.float32)
        self.assertEqual(experts.w2_weight_bias.dtype, torch.float32)

    def test_deepseek_v4_moe_selects_sm12x_model_level_experts(self):
        previous_backend = moe_utils.MOE_BACKEND
        server_args_snapshot = dict(global_server_args_dict)
        moe_utils.MOE_BACKEND = MoeBackend.SM12X_MXFP4
        global_server_args_dict["ep_num_redundant_experts"] = 0
        try:
            with patch(
                "tokenspeed.runtime.models.deepseek_v4._platform",
                SimpleNamespace(is_sm12x=True),
            ):
                moe = DeepseekV4MoE(
                    _moe_config(),
                    _ep2_mapping(),
                    quant_config=None,
                    layer_index=0,
                    prefix="layers.0.ffn",
                )
        finally:
            moe_utils.MOE_BACKEND = previous_backend
            global_server_args_dict.clear()
            global_server_args_dict.update(server_args_snapshot)

        self.assertFalse(moe.use_mega_moe)
        self.assertTrue(moe.use_sm12x_mxfp4)
        self.assertIsInstance(moe.experts, DeepseekV4Sm12xMoEExperts)
        self.assertIsNone(moe.topk)
        self.assertEqual(moe.hash_indices_dtype, torch.int32)

    def test_forward_uses_sm12x_kernel_contract(self):
        import sys

        experts = DeepseekV4Sm12xMoEExperts(
            num_experts=4,
            num_local_experts=2,
            top_k=2,
            hidden_size=128,
            intermediate_size=128,
            mapping=_ep2_mapping(),
            prefix="layers.0.ffn.experts",
        )
        experts.finalize_weights()

        captured = {}

        def fake_forward(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            return torch.full_like(args[0], 3)

        fake_module = SimpleNamespace(sm12x_mxfp4_moe_forward=fake_forward)
        hidden_states = torch.ones(3, 128, dtype=torch.bfloat16)
        topk_weights = torch.full((3, 2), 0.5, dtype=torch.float32)
        topk_ids = torch.tensor([[0, 1], [1, 3], [0, 2]], dtype=torch.int64)

        with patch.dict(
            sys.modules,
            {"tokenspeed_kernel.ops.moe.sm12x_mxfp4": fake_module},
        ):
            output = experts(
                hidden_states,
                topk_weights,
                topk_ids,
                activation_clamp=5.0,
            )

        torch.testing.assert_close(output, torch.full_like(hidden_states, 3))
        self.assertIs(captured["args"][0], hidden_states)
        self.assertIs(captured["args"][1], topk_weights)
        self.assertEqual(captured["args"][2].dtype, torch.int32)
        self.assertIs(captured["args"][3], experts.w13_weight)
        self.assertIs(captured["args"][4], experts.w13_weight_scale)
        self.assertIs(captured["args"][5], experts.w2_weight)
        self.assertIs(captured["args"][6], experts.w2_weight_scale)
        self.assertIs(captured["kwargs"]["w13_bias"], experts.w13_weight_bias)
        # ``w2_weight_bias`` is left at the zero initializer in this test (no
        # checkpoint with a ``w2`` bias shard was loaded), so the SM12x MoE
        # forward sees ``w2_bias=None`` -- the ``auto`` dispatcher inside
        # sm12x_mxfp4_moe_forward needs that signal to be allowed to route
        # to the tensorcore kernel (which silently drops ``w2_bias``).
        self.assertTrue(experts._w2_bias_is_zero)
        self.assertIsNone(captured["kwargs"]["w2_bias"])
        self.assertEqual(captured["kwargs"]["activation"], "swiglu")
        self.assertEqual(captured["kwargs"]["swiglu_limit"], 5.0)
        self.assertEqual(captured["kwargs"]["ep_rank"], 0)
        self.assertEqual(captured["kwargs"]["ep_size"], 2)

    def test_forward_passes_w2_bias_when_loaded_nonzero(self):
        """When the checkpoint loaded a non-zero ``w2_bias`` shard, the
        finalize-time detection records it, and the forward keeps passing
        the bias tensor through to the kernel -- never silently routes to
        a path that would drop it.
        """
        import sys

        experts = DeepseekV4Sm12xMoEExperts(
            num_experts=4,
            num_local_experts=2,
            top_k=2,
            hidden_size=128,
            intermediate_size=128,
            mapping=_ep2_mapping(),
            prefix="layers.0.ffn.experts",
        )
        b2 = torch.full((128,), 0.125, dtype=torch.bfloat16)
        experts.weight_loader(experts.w2_weight_bias, b2, "w2", local_expert_id=1)
        experts.finalize_weights()
        self.assertFalse(experts._w2_bias_is_zero)

        captured = {}

        def fake_forward(*args, **kwargs):
            captured["kwargs"] = kwargs
            return torch.full_like(args[0], 3)

        fake_module = SimpleNamespace(sm12x_mxfp4_moe_forward=fake_forward)
        hidden_states = torch.ones(3, 128, dtype=torch.bfloat16)
        topk_weights = torch.full((3, 2), 0.5, dtype=torch.float32)
        topk_ids = torch.tensor([[0, 1], [1, 3], [0, 2]], dtype=torch.int64)
        with patch.dict(
            sys.modules,
            {"tokenspeed_kernel.ops.moe.sm12x_mxfp4": fake_module},
        ):
            experts(
                hidden_states,
                topk_weights,
                topk_ids,
                activation_clamp=5.0,
            )
        self.assertIs(captured["kwargs"]["w2_bias"], experts.w2_weight_bias)

    def test_forward_reuses_sm12x_work_buffers(self):
        import sys

        experts = DeepseekV4Sm12xMoEExperts(
            num_experts=4,
            num_local_experts=2,
            top_k=2,
            hidden_size=128,
            intermediate_size=128,
            mapping=_ep2_mapping(),
            prefix="layers.0.ffn.experts",
        )
        experts.finalize_weights()

        buffer_ptrs = []

        def fake_forward(*args, **kwargs):
            output = kwargs["output"]
            intermediate = kwargs["intermediate"]
            buffer_ptrs.append((output.data_ptr(), intermediate.data_ptr()))
            self.assertEqual(output.shape, (2, 128))
            self.assertEqual(intermediate.shape, (2, 2, 128))
            output.fill_(4)
            return output

        fake_module = SimpleNamespace(sm12x_mxfp4_moe_forward=fake_forward)
        hidden_states = torch.ones(2, 128, dtype=torch.bfloat16)
        topk_weights = torch.full((2, 2), 0.5, dtype=torch.float32)
        topk_ids = torch.tensor([[0, 1], [1, 3]], dtype=torch.int32)

        with patch.dict(
            sys.modules,
            {"tokenspeed_kernel.ops.moe.sm12x_mxfp4": fake_module},
        ):
            first = experts(
                hidden_states,
                topk_weights,
                topk_ids,
                activation_clamp=None,
            )
            second = experts(
                hidden_states,
                topk_weights,
                topk_ids,
                activation_clamp=None,
            )

        torch.testing.assert_close(first, torch.full_like(hidden_states, 4))
        torch.testing.assert_close(second, torch.full_like(hidden_states, 4))
        self.assertEqual(buffer_ptrs[0], buffer_ptrs[1])


if __name__ == "__main__":
    unittest.main()
