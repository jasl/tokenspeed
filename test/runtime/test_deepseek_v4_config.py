import argparse
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from tokenspeed.runtime.configs.deepseek_v4_config import DeepseekV4Config
from tokenspeed.runtime.configs.model_config import (
    AttentionArch,
    ModelConfig,
    configure_deepseek_v4_attention,
    is_deepseek_v4,
)
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.deepseek_v4 import (
    DeepseekV4AttentionBackend,
)
from tokenspeed.runtime.layers.attention.deepseek_v4_ops import (
    DEEPSEEK_V4_INDEXER_DIM,
    DeepseekV4AttentionOpUnavailable,
    deepseek_v4_indexer_topk_reference,
    fused_qnorm_rope_kv_insert,
    has_fused_qnorm_rope_kv_insert,
)
from tokenspeed.runtime.layers.attention.kv_cache.deepseek_v4 import (
    DeepseekV4ForwardMetadata,
    DeepseekV4TokenToKVPool,
    _group_slot_mapping_from_raw,
    deepseek_v4_cache_layout_from_config,
)
from tokenspeed.runtime.layers.layernorm import FusedRMSNorm, RMSNorm
from tokenspeed.runtime.layers.moe.backends.mxfp4.flashinfer import (
    _get_flashinfer_mxfp4_device_permute_indices,
    _reorder_w1w3_to_w3w1,
)
from tokenspeed.runtime.layers.quantization import QUANTIZATION_METHODS
from tokenspeed.runtime.models.deepseek_v4 import (
    DeepseekV4Indexer,
    DeepseekV4MoEGate,
    _deepseek_v4_deepgemm_fp4_indexer_available,
    _deepseek_v4_deepgemm_fp4_indexer_enabled_for_platform,
    _deepseek_v4_fast_mhc_enabled_for_platform,
    _deepseek_v4_fast_topk_enabled_for_platform,
    _deepseek_v4_fused_select_experts,
    _deepseek_v4_indexer_decode_max_len,
    _deepseek_v4_indexer_prefill_topk_chunks,
    _deepseek_v4_indexer_topk_from_cache_batched,
    _deepseek_v4_indexer_topk_from_logits,
    _deepseek_v4_reorder_c4_ape_2604,
    _deepseek_v4_trtllm_fp8_quant_enabled_for_platform,
    _DeepseekV4TopKBuffer,
    _fp8_act_quant_dequant,
    _fp8_linear,
    _mhc_post_reference,
    _mhc_pre_reference,
    deepseek_v4_attention_layout,
    deepseek_v4_rope_config,
    deepseek_v4_select_experts,
    hc_head,
    mhc_post,
    mhc_pre,
    pack_topk_as_router_logits,
)
from tokenspeed.runtime.utils.env import global_server_args_dict
from tokenspeed.runtime.utils.hf_transformers_utils import (
    _CONFIG_REGISTRY,
    _wrap_deepseek_v4_tokenizer,
    get_tokenizer,
    prefers_deepseek_v4_tokenizer,
)


class TestDeepseekV4Config(unittest.TestCase):
    quant_config = {
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "scale_fmt": "ue8m0",
    }

    def test_config_registry(self):
        self.assertEqual(DeepseekV4Config.model_type, "deepseek_v4")
        self.assertIs(_CONFIG_REGISTRY["deepseek_v4"], DeepseekV4Config)

    def test_deepseek_v4_tokenizer_wrapper_uses_model_encoder(self):
        calls = []

        class DummyTokenizer:
            vocab_size = 5

            def __call__(self, text, add_special_tokens=False, **kwargs):
                self.last_call = (text, add_special_tokens, kwargs)
                return {"input_ids": [len(text)]}

            def encode(self, text, add_special_tokens=False, **kwargs):
                return [len(text)]

            def get_added_vocab(self):
                return {"<extra>": 5}

        def encode_messages(messages, **kwargs):
            calls.append((messages, kwargs))
            return "<encoded>"

        tokenizer = _wrap_deepseek_v4_tokenizer(DummyTokenizer(), encode_messages)

        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": "hi"}],
            tokenize=False,
            enable_thinking=True,
            reasoning_effort="medium",
        )
        token_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": "hi"}],
            truncation=True,
            max_length=16,
        )

        self.assertEqual(prompt, "<encoded>")
        self.assertEqual(token_ids, [9])
        self.assertEqual(len(tokenizer), 6)
        self.assertEqual(calls[0][1]["thinking_mode"], "thinking")
        self.assertIsNone(calls[0][1]["reasoning_effort"])
        self.assertEqual(calls[1][1]["thinking_mode"], "chat")
        self.assertEqual(
            tokenizer.last_call,
            ("<encoded>", False, {"truncation": True, "max_length": 16}),
        )

    def test_deepseek_v4_tokenizer_is_auto_selected_by_architecture(self):
        self.assertTrue(prefers_deepseek_v4_tokenizer(["DeepseekV4ForCausalLM"]))
        self.assertFalse(prefers_deepseek_v4_tokenizer(["KimiK2ForCausalLM"]))
        self.assertFalse(prefers_deepseek_v4_tokenizer(None))

    def test_auto_tokenizer_mode_wraps_deepseek_v4_architecture(self):
        class DummyTokenizer:
            vocab_size = 5

            def __call__(self, text, add_special_tokens=False, **kwargs):
                return {"input_ids": [len(text)]}

            def encode(self, text, add_special_tokens=False, **kwargs):
                return [len(text)]

            def get_added_vocab(self):
                return {}

        def encode_messages(messages, **kwargs):
            return "<encoded>"

        with (
            patch(
                "tokenspeed.runtime.utils.hf_transformers_utils.AutoTokenizer.from_pretrained",
                return_value=DummyTokenizer(),
            ),
            patch(
                "tokenspeed.runtime.utils.hf_transformers_utils._load_deepseek_v4_encode_messages",
                return_value=encode_messages,
            ),
        ):
            tokenizer = get_tokenizer(
                "deepseek-ai/DeepSeek-V4-Flash",
                tokenizer_mode="auto",
                architectures=["DeepseekV4ForCausalLM"],
            )

        self.assertEqual(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": "hi"}],
            ),
            [9],
        )

    def test_deepseek_v4_server_args_cli_flags_round_trip(self):
        from tokenspeed.runtime.utils.env import (
            global_server_args_dict,
            global_server_args_dict_update,
        )
        from tokenspeed.runtime.utils.server_args import ServerArgs

        # Defaults match dataclass declaration
        self.assertFalse(ServerArgs.disable_deepseek_v4_fast_mhc)
        self.assertEqual(ServerArgs.deepseek_v4_mega_moe_max_num_tokens, 0)
        self.assertEqual(ServerArgs.deepseek_v4_indexer_prefill_max_logits_mb, 512)

        # CLI flags parse
        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        ns = parser.parse_args(
            [
                "--model=stub",
                "--disable-deepseek-v4-fast-mhc",
                "--deepseek-v4-mega-moe-max-num-tokens=128",
                "--deepseek-v4-indexer-prefill-max-logits-mb=256",
            ]
        )
        args = ServerArgs.from_cli_args(ns)
        self.assertTrue(args.disable_deepseek_v4_fast_mhc)
        self.assertEqual(args.deepseek_v4_mega_moe_max_num_tokens, 128)
        self.assertEqual(args.deepseek_v4_indexer_prefill_max_logits_mb, 256)

        # Propagation into global_server_args_dict
        snapshot = dict(global_server_args_dict)
        try:
            global_server_args_dict_update(args)
            self.assertTrue(global_server_args_dict["disable_deepseek_v4_fast_mhc"])
            self.assertEqual(
                global_server_args_dict["deepseek_v4_mega_moe_max_num_tokens"], 128
            )
            self.assertEqual(
                global_server_args_dict["deepseek_v4_indexer_prefill_max_logits_mb"],
                256,
            )
        finally:
            global_server_args_dict.clear()
            global_server_args_dict.update(snapshot)

    def test_fp8_quantization_config(self):
        quantization = QUANTIZATION_METHODS["fp8"]

        config = quantization.from_config(self.quant_config)

        self.assertEqual(quantization.get_name(), "fp8")
        self.assertIsNone(
            quantization.override_quantization_method(self.quant_config, None)
        )
        self.assertEqual(config.activation_scheme, "dynamic")
        self.assertTrue(config.is_checkpoint_fp8_serialized)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
    def test_deepseek_v4_fused_qkv_rmsnorm_matches_separate(self):
        torch.manual_seed(0)
        q = torch.randn(8, 1536, device="cuda", dtype=torch.bfloat16)
        kv = torch.randn(8, 512, device="cuda", dtype=torch.bfloat16)
        q_norm = RMSNorm(1536, eps=1e-6).cuda().to(torch.bfloat16)
        kv_norm = RMSNorm(512, eps=1e-6).cuda().to(torch.bfloat16)
        fused_norm = FusedRMSNorm(q_norm, kv_norm)

        q_out = torch.empty_like(q)
        kv_out = torch.empty_like(kv)
        try:
            fused_norm(q, kv, output_q_a=q_out, output_kv_a=kv_out)
        except RuntimeError as exc:
            self.skipTest(str(exc))

        torch.cuda.synchronize()
        self.assertTrue(torch.equal(q_out, q_norm(q)))
        self.assertTrue(torch.equal(kv_out, kv_norm(kv)))

    def test_model_config_maps_deepseek_v4_to_standard_fp8(self):
        model_config = object.__new__(ModelConfig)
        model_config.hf_config = SimpleNamespace(
            model_type="deepseek_v4", quantization_config=self.quant_config
        )
        model_config.quantization = None

        model_config._verify_quantization()

        self.assertEqual(model_config.quantization, "fp8")

    def test_model_config_overrides_default_block_size_for_deepseek_v4(self):
        def make_hf_config():
            return SimpleNamespace(
                architectures=["DeepseekV4ForCausalLM"],
                model_type="deepseek_v4",
                head_dim=512,
                qk_rope_head_dim=64,
                index_head_dim=128,
                rope_scaling=None,
                hidden_size=4096,
                num_attention_heads=8,
                num_key_value_heads=8,
                num_hidden_layers=1,
                vocab_size=32000,
                quantization_config=None,
            )

        def build(block_size):
            server_args = SimpleNamespace(
                mapping=None,
                block_size=block_size,
                load_format="auto",
                ext_yaml=None,
            )
            hf_config = make_hf_config()
            with (
                patch(
                    "tokenspeed.runtime.configs.model_config.get_config",
                    return_value=hf_config,
                ),
                patch(
                    "tokenspeed.runtime.configs.model_config.get_generation_config",
                    return_value=SimpleNamespace(eos_token_id=None),
                ),
                patch(
                    "tokenspeed.runtime.configs.model_config.get_hf_text_config",
                    return_value=hf_config,
                ),
                patch(
                    "tokenspeed.runtime.configs.model_config.get_context_length",
                    return_value=4096,
                ),
                patch.object(ModelConfig, "_verify_quantization"),
            ):
                ModelConfig(
                    "stub",
                    model_override_args="{}",
                    server_args=server_args,
                )
            return server_args

        self.assertEqual(build(64).block_size, 256)
        self.assertEqual(build(128).block_size, 128)

    def test_model_config_keeps_incompatible_user_quantization_error(self):
        model_config = object.__new__(ModelConfig)
        model_config.hf_config = SimpleNamespace(
            model_type="deepseek_v4", quantization_config=self.quant_config
        )
        model_config.quantization = "mxfp4"

        with self.assertRaisesRegex(ValueError, "does not match"):
            model_config._verify_quantization()

    def test_deepseek_v4_attention_op_boundary_fails_loudly_when_missing(self):
        if has_fused_qnorm_rope_kv_insert():
            self.skipTest("DeepSeek V4 fused attention op is available in this build")

        q = torch.empty(1, 1, 512)
        kv = torch.empty(1, 512)
        cache = torch.empty(1, 584, dtype=torch.uint8)
        slots = torch.zeros(1, dtype=torch.int32)
        positions = torch.zeros(1, dtype=torch.int32)
        cos_sin = torch.empty(1, 128)

        with self.assertRaisesRegex(
            DeepseekV4AttentionOpUnavailable,
            "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert",
        ):
            fused_qnorm_rope_kv_insert(
                q, kv, cache, slots, positions, cos_sin, 1e-6, 256
            )

    def test_deepseek_v4_model_config_uses_mla_runtime_metadata(self):
        model_config = object.__new__(ModelConfig)
        model_config.hf_config = SimpleNamespace(
            architectures=["DeepseekV4ForCausalLM"],
            head_dim=512,
            qk_rope_head_dim=64,
            index_head_dim=128,
            rope_scaling=None,
        )

        self.assertTrue(is_deepseek_v4(model_config.hf_config))

        configure_deepseek_v4_attention(model_config)

        self.assertEqual(model_config.attention_arch, AttentionArch.MLA)
        self.assertEqual(model_config.head_dim, 512)
        self.assertEqual(model_config.kv_lora_rank, 512)
        self.assertEqual(model_config.qk_rope_head_dim, 64)
        self.assertEqual(model_config.qk_nope_head_dim, 448)
        self.assertEqual(model_config.v_head_dim, 512)
        self.assertEqual(model_config.index_head_dim, 128)
        self.assertAlmostEqual(model_config.scaling, 512**-0.5)

    def test_deepseek_v4_attention_layout_matches_compressed_cache_contract(self):
        config = SimpleNamespace(
            compress_ratios=[0, 4, 128],
            num_attention_heads=64,
            head_dim=512,
            qk_rope_head_dim=64,
            sliding_window=128,
            index_head_dim=128,
        )

        swa = deepseek_v4_attention_layout(config, 0, attn_tp_size=4)
        csa = deepseek_v4_attention_layout(config, 1, attn_tp_size=4)
        csa_fp4 = deepseek_v4_attention_layout(
            config, 1, attn_tp_size=4, use_fp4_indexer_cache=True
        )
        hca = deepseek_v4_attention_layout(config, 2, attn_tp_size=4)

        self.assertEqual(swa.kind, "swa")
        self.assertEqual(swa.compress_ratio, 1)
        self.assertEqual(swa.num_local_heads, 16)
        self.assertEqual(swa.padded_heads, 64)
        self.assertEqual(swa.nope_head_dim, 448)
        self.assertEqual(swa.swa_head_bytes, 584)
        self.assertFalse(swa.needs_compressed_cache)
        self.assertFalse(swa.needs_indexer)

        self.assertEqual(csa.kind, "csa")
        self.assertEqual(csa.compress_ratio, 4)
        self.assertTrue(csa.needs_compressed_cache)
        self.assertTrue(csa.needs_indexer)
        self.assertEqual(csa.compressed_cache_alignment, 576)
        self.assertEqual(csa.indexer_cache_head_bytes, 132)
        self.assertEqual(csa_fp4.indexer_cache_head_bytes, 68)

        self.assertEqual(hca.kind, "hca")
        self.assertEqual(hca.compress_ratio, 128)
        self.assertTrue(hca.needs_compressed_cache)
        self.assertFalse(hca.needs_indexer)

    def test_deepseek_v4_attention_layout_rejects_unknown_ratio(self):
        config = SimpleNamespace(
            compress_ratios=[8],
            num_attention_heads=64,
            head_dim=512,
            qk_rope_head_dim=64,
            sliding_window=128,
            index_head_dim=128,
        )

        with self.assertRaisesRegex(ValueError, "compress_ratio=8"):
            deepseek_v4_attention_layout(config, 0)

    def test_deepseek_v4_rope_config_matches_layer_type(self):
        config = SimpleNamespace(
            rope_theta=10000,
            compress_rope_theta=160000,
            rope_scaling={
                "type": "yarn",
                "factor": 16,
                "original_max_position_embeddings": 65536,
                "beta_fast": 32,
                "beta_slow": 1,
            },
        )

        swa_base, swa_scaling = deepseek_v4_rope_config(config, compress_ratio=1)
        csa_base, csa_scaling = deepseek_v4_rope_config(config, compress_ratio=4)

        self.assertEqual(swa_base, 10000.0)
        self.assertIsNone(swa_scaling)
        self.assertEqual(csa_base, 160000.0)
        self.assertIsNot(csa_scaling, config.rope_scaling)
        self.assertEqual(csa_scaling["rope_type"], "deepseek_yarn")
        self.assertEqual(csa_scaling["factor"], 16)
        self.assertEqual(csa_scaling["mscale"], 0)
        self.assertEqual(csa_scaling["mscale_all_dim"], 0)

    def test_deepseek_v4_kv_pool_allocates_v4_cache_families(self):
        config = SimpleNamespace(
            compress_ratios=[1, 4, 128],
            head_dim=512,
            index_head_dim=128,
            sliding_window=128,
        )
        layout = deepseek_v4_cache_layout_from_config(
            config,
            page_size=64,
            use_fp4_indexer_cache=True,
        )

        self.assertEqual(layout.cache_cell_size(3), 16771)

        pool = DeepseekV4TokenToKVPool(
            size=128,
            model_dtype=torch.bfloat16,
            layout=layout,
            layer_num=3,
            device="cpu",
            enable_memory_saver=False,
            max_batch_size=2,
            max_context_len=128,
            page_size=64,
            rank=0,
            hf_config=config,
            max_scheduled_tokens=1,
        )

        self.assertEqual(tuple(pool.get_swa_kv_buffer(0).shape), (7, 37440))
        self.assertIsNone(pool.compressed_kv_buffer[0])
        self.assertEqual(tuple(pool.get_compressed_kv_buffer_2d(1).shape), (4, 37440))
        self.assertEqual(tuple(pool.get_compressor_state_buffer(1).shape), (7, 4, 2048))
        self.assertEqual(
            tuple(pool.get_compressor_state_buffer(2).shape), (35, 8, 1024)
        )
        self.assertEqual(pool.get_compressor_state_buffer(1).dtype, torch.float32)
        self.assertEqual(pool.get_compressor_state_buffer(2).dtype, torch.float32)
        self.assertEqual(tuple(pool.get_indexer_kv_buffer_2d(1).shape), (4, 64 * 68))
        self.assertEqual(tuple(pool.get_indexer_state_buffer(1).shape), (7, 4, 512))
        self.assertEqual(pool.get_indexer_state_buffer(1).dtype, torch.float32)

    def test_deepseek_v4_kv_pool_uses_compressed_storage_blocks_for_page256(self):
        config = SimpleNamespace(
            compress_ratios=[1, 4, 128],
            head_dim=512,
            index_head_dim=128,
            sliding_window=128,
        )
        layout = deepseek_v4_cache_layout_from_config(
            config,
            page_size=256,
            use_fp4_indexer_cache=True,
        )
        pool = DeepseekV4TokenToKVPool(
            size=512,
            model_dtype=torch.bfloat16,
            layout=layout,
            layer_num=3,
            device="cpu",
            enable_memory_saver=False,
            max_batch_size=2,
            max_context_len=512,
            page_size=256,
            rank=0,
            hf_config=config,
            max_scheduled_tokens=1,
        )

        self.assertEqual(pool.swa_block_size, 64)
        self.assertEqual(pool.get_compressed_block_size(1), 64)
        self.assertEqual(pool.get_compressed_block_size(2), 2)
        self.assertEqual(tuple(pool.get_compressed_kv_buffer_2d(1).shape), (5, 37440))
        self.assertEqual(tuple(pool.get_indexer_kv_buffer_2d(1).shape), (5, 64 * 68))

    def test_deepseek_v4_kv_pool_rejects_nonpositive_size(self):
        config = SimpleNamespace(
            compress_ratios=[1],
            head_dim=512,
            index_head_dim=128,
            sliding_window=128,
        )
        layout = deepseek_v4_cache_layout_from_config(
            config,
            page_size=64,
            use_fp4_indexer_cache=True,
        )

        with self.assertRaisesRegex(ValueError, "must be positive"):
            DeepseekV4TokenToKVPool(
                size=0,
                model_dtype=torch.bfloat16,
                layout=layout,
                layer_num=1,
                device="cpu",
                enable_memory_saver=False,
                max_batch_size=2,
                max_context_len=128,
                page_size=64,
                rank=0,
                hf_config=config,
                max_scheduled_tokens=1,
            )

    def test_deepseek_v4_group_slot_mapping_consumes_compact_base_offsets(self):
        slots = _group_slot_mapping_from_raw(
            positions=torch.tensor([128, 129, 192, 64], dtype=torch.int64),
            req_indices=torch.tensor([0, 0, 1, 1], dtype=torch.int32),
            block_table=torch.tensor([[10, 11], [20, 21]], dtype=torch.int32),
            rows_per_page=64,
            base_offsets=torch.tensor([2, 1], dtype=torch.int32),
        )

        self.assertTrue(torch.equal(slots, torch.tensor([640, 641, -1, 1280])))

    def test_deepseek_v4_backend_preserves_compact_paged_cache_contract(self):
        backend = DeepseekV4AttentionBackend(
            SimpleNamespace(
                page_size=64,
                device="cpu",
                num_attention_heads=64,
                num_kv_heads=1,
                attn_tp_size=1,
                dtype=torch.bfloat16,
                head_dim=512,
                context_len=4096,
            )
        )
        compact = torch.tensor([[10, 11], [20, -1]], dtype=torch.int32)
        base = torch.tensor([2, 1], dtype=torch.int32)

        backend.init_forward_metadata(
            bs=2,
            num_tokens=2,
            req_pool_indices=torch.tensor([0, 1], dtype=torch.int64),
            seq_lens=torch.tensor([200, 80], dtype=torch.int32),
            forward_mode=ForwardMode.DECODE,
            req_to_page=torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.int32),
            paged_cache_block_tables={"v4.swa_kv": compact},
            paged_cache_block_table_base_offsets={"v4.swa_kv": base},
        )

        metadata = backend.forward_metadata
        self.assertIsNotNone(metadata)
        assert metadata is not None
        self.assertTrue(torch.equal(metadata.swa_block_table, compact))
        self.assertTrue(torch.equal(metadata.swa_base_logical_page, base))

    def test_deepseek_v4_cuda_graph_refresh_keeps_compact_table_columns(self):
        backend = DeepseekV4AttentionBackend(
            SimpleNamespace(
                page_size=64,
                device="cpu",
                num_attention_heads=64,
                num_kv_heads=1,
                attn_tp_size=1,
                dtype=torch.bfloat16,
                head_dim=512,
                context_len=4096,
            )
        )
        backend.init_cuda_graph_state(
            2,
            paged_cache_group_specs=(
                SimpleNamespace(
                    group_id="v4.swa_kv",
                    retention="sliding_window",
                    rows_per_page=64,
                    entry_stride_tokens=1,
                    sliding_window_tokens=128,
                ),
            ),
            max_tokens_per_req=1,
        )
        compact = torch.tensor([[10, 11], [20, -1]], dtype=torch.int32)
        refreshed = backend._refresh_cuda_graph_paged_cache_block_tables(
            2,
            {"v4.swa_kv": compact},
            pad_value=-1,
            paged_cache_block_table_base_offsets={
                "v4.swa_kv": torch.tensor([2, 1], dtype=torch.int32)
            },
        )

        table = refreshed["v4.swa_kv"]
        self.assertTrue(torch.equal(table[:, :2], compact))
        self.assertTrue(torch.equal(table[:, 2:], torch.full_like(table[:, 2:], -1)))

    def test_deepseek_v4_metadata_splits_named_cache_groups(self):
        backend = DeepseekV4AttentionBackend(
            SimpleNamespace(
                page_size=64,
                device="cpu",
                num_attention_heads=64,
                num_kv_heads=1,
                attn_tp_size=1,
                dtype=torch.bfloat16,
                head_dim=512,
                context_len=4096,
            )
        )
        swa = torch.tensor([[10, 11], [20, -1]], dtype=torch.int32)
        c4_state = torch.tensor([[30], [40]], dtype=torch.int32)
        c128_state = torch.tensor([[50], [60]], dtype=torch.int32)
        indexer_state = torch.tensor([[70], [80]], dtype=torch.int32)
        c4_state_base = torch.tensor([3, 4], dtype=torch.int32)
        c128_state_base = torch.tensor([5, 6], dtype=torch.int32)
        indexer_state_base = torch.tensor([7, 8], dtype=torch.int32)

        backend.init_forward_metadata(
            bs=2,
            num_tokens=2,
            req_pool_indices=torch.tensor([0, 1], dtype=torch.int64),
            seq_lens=torch.tensor([200, 80], dtype=torch.int32),
            forward_mode=ForwardMode.DECODE,
            req_to_page=torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.int32),
            paged_cache_block_tables={
                "v4.swa_kv": swa,
                "v4.c4a.compressor_state": c4_state,
                "v4.c128a.compressor_state": c128_state,
                "v4.c4a.indexer_compressor_state": indexer_state,
            },
            paged_cache_block_table_base_offsets={
                "v4.c4a.compressor_state": c4_state_base,
                "v4.c128a.compressor_state": c128_state_base,
                "v4.c4a.indexer_compressor_state": indexer_state_base,
            },
        )

        metadata = backend.forward_metadata
        self.assertIsNotNone(metadata)
        assert metadata is not None
        self.assertTrue(torch.equal(metadata.swa_block_table, swa))
        self.assertTrue(
            torch.equal(metadata.compressor_state_block_tables[4], c4_state)
        )
        self.assertTrue(
            torch.equal(metadata.compressor_state_block_tables[128], c128_state)
        )
        self.assertTrue(torch.equal(metadata.indexer_state_block_table, indexer_state))
        self.assertTrue(
            torch.equal(metadata.compressor_state_base_logical_pages[4], c4_state_base)
        )
        self.assertTrue(
            torch.equal(
                metadata.compressor_state_base_logical_pages[128],
                c128_state_base,
            )
        )
        self.assertTrue(
            torch.equal(metadata.indexer_state_base_logical_page, indexer_state_base)
        )

    def test_deepseek_v4_metadata_maps_compressed_slots(self):
        compressed_table = torch.tensor([[10, 11], [20, 21]], dtype=torch.int32)
        metadata = DeepseekV4ForwardMetadata(
            page_size=64,
            req_pool_indices=torch.tensor([0, 1], dtype=torch.int32),
            block_table=torch.tensor([[0, 1], [3, 4]], dtype=torch.int32),
            seq_lens=torch.tensor([70, 5], dtype=torch.int32),
            query_lens=torch.tensor([3, 5], dtype=torch.int32),
            query_start_loc=torch.tensor([0, 3, 8], dtype=torch.int32),
            token_to_req_indices=torch.tensor(
                [0, 0, 0, 1, 1, 1, 1, 1],
                dtype=torch.int32,
            ),
            paged_cache_block_tables={"v4.c4a.compressed_kv": compressed_table},
        )

        self.assertTrue(
            torch.equal(
                metadata.token_to_req_indices,
                torch.tensor([0, 0, 0, 1, 1, 1, 1, 1], dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(metadata.compressed_block_table(4), compressed_table)
        )
        self.assertTrue(
            torch.equal(metadata.compressed_block_table(128), metadata.block_table)
        )
        slots = metadata.compressed_slot_mapping(
            torch.tensor([3, 7, 127], dtype=torch.int64),
            compress_ratio=4,
        )
        self.assertTrue(torch.equal(slots, torch.tensor([640, 641, 671])))

        page256_metadata = DeepseekV4ForwardMetadata(
            page_size=256,
            req_pool_indices=torch.tensor([0], dtype=torch.int32),
            block_table=torch.tensor([[5, 6]], dtype=torch.int32),
            seq_lens=torch.tensor([300], dtype=torch.int32),
            query_lens=torch.tensor([3], dtype=torch.int32),
            query_start_loc=torch.tensor([0, 3], dtype=torch.int32),
            token_to_req_indices=torch.tensor([0, 0, 0], dtype=torch.int32),
        )
        slots = page256_metadata.compressed_slot_mapping(
            torch.tensor([255, 256, 511], dtype=torch.int64),
            compress_ratio=4,
            kv_cache_block_size=64,
        )
        self.assertTrue(torch.equal(slots, torch.tensor([383, 384, 447])))

    def test_deepseek_v4_decode_backend_maps_compressed_slots_batched(self):
        backend = DeepseekV4AttentionBackend(
            SimpleNamespace(
                page_size=64,
                device="cpu",
                num_attention_heads=64,
                num_kv_heads=1,
                attn_tp_size=1,
                dtype=torch.bfloat16,
                head_dim=512,
                context_len=4096,
            )
        )
        seq_lens = torch.tensor([70, 3], dtype=torch.int32)
        backend.init_forward_metadata(
            bs=2,
            num_tokens=2,
            req_pool_indices=torch.tensor([0, 1], dtype=torch.int64),
            seq_lens=seq_lens,
            forward_mode=ForwardMode.DECODE,
            req_to_page=torch.tensor([[10, 11], [20, 21]], dtype=torch.int32),
        )
        positions = seq_lens.to(torch.int64) - 1

        topk_indices = torch.tensor(
            [[1, 65, 3, -1], [0, -1, -1, -1]],
            dtype=torch.int32,
        )
        indices, lens = backend._decode_compressed_indices_and_lens(
            positions,
            compress_ratio=4,
            block_size=64,
            topk_indices=topk_indices,
        )
        self.assertTrue(torch.equal(lens, torch.tensor([3, 1], dtype=torch.int32)))
        self.assertTrue(
            torch.equal(
                indices[:, 0, :4],
                torch.tensor(
                    [[641, 705, 643, -1], [1280, -1, -1, -1]],
                    dtype=torch.int32,
                ),
            )
        )

        seq_lens = torch.tensor([256, 129], dtype=torch.int32)
        backend.init_forward_metadata(
            bs=2,
            num_tokens=2,
            req_pool_indices=torch.tensor([0, 1], dtype=torch.int64),
            seq_lens=seq_lens,
            forward_mode=ForwardMode.DECODE,
            req_to_page=torch.tensor(
                [[10, 11, 12, 13], [20, 21, 22, 23]],
                dtype=torch.int32,
            ),
        )
        indices, lens = backend._decode_compressed_indices_and_lens(
            seq_lens.to(torch.int64) - 1,
            compress_ratio=128,
            block_size=64,
            topk_indices=None,
        )
        self.assertTrue(torch.equal(lens, torch.tensor([2, 1], dtype=torch.int32)))
        self.assertTrue(
            torch.equal(
                indices[:, 0, :2],
                torch.tensor([[640, 641], [1280, -1]], dtype=torch.int32),
            )
        )

    def test_deepseek_v4_indexer_decode_batches_cache_reads(self):
        torch.manual_seed(0)
        positions = torch.tensor([15, 7, 3], dtype=torch.int64)
        token_to_req_indices = torch.tensor([0, 1, 2], dtype=torch.int32)
        block_table = torch.tensor([[0], [1], [2]], dtype=torch.int32)
        cache = torch.randn(12, 128, dtype=torch.float32)
        index_q = torch.randn(3, 2, 128, dtype=torch.float32)
        weights = torch.randn(3, 2, dtype=torch.float32)

        def cache_reader(cache_2d, slot_mapping, block_size):
            del block_size
            return cache_2d[slot_mapping.long()]

        actual = _deepseek_v4_indexer_topk_from_cache_batched(
            cache_reader=cache_reader,
            cache_2d=cache,
            positions=positions,
            token_to_req_indices=token_to_req_indices,
            block_table=block_table,
            cache_block_size=4,
            index_q=index_q,
            weights=weights,
            compress_ratio=4,
            topk_tokens=3,
        )

        expected = torch.full((3, 3), -1, dtype=torch.int32)
        for token_idx, position in enumerate(positions.tolist()):
            num_compressed = (position + 1) // 4
            local = torch.arange(num_compressed, dtype=torch.int64)
            req_idx = int(token_to_req_indices[token_idx].item())
            pages = torch.div(local, 4, rounding_mode="floor")
            offsets = local % 4
            page_ids = block_table[req_idx, pages.long()].to(torch.int64)
            slots = page_ids * 4 + offsets
            selected = min(num_compressed, expected.shape[1])
            expected[token_idx, :selected] = deepseek_v4_indexer_topk_reference(
                index_q[token_idx : token_idx + 1],
                cache_reader(cache, slots, 4),
                weights[token_idx : token_idx + 1],
                top_k=selected,
            )[0]

        self.assertTrue(torch.equal(actual, expected))

    def test_deepseek_v4_indexer_decode_all_candidate_shortcut_skips_cache_read(self):
        def cache_reader(cache_2d, slot_mapping, block_size):
            del cache_2d, slot_mapping, block_size
            raise AssertionError("all-candidate shortcut should not read cache")

        topk = _deepseek_v4_indexer_topk_from_cache_batched(
            cache_reader=cache_reader,
            cache_2d=torch.empty(0),
            positions=torch.tensor([0, 3, 7, 11], dtype=torch.int64),
            token_to_req_indices=torch.zeros(4, dtype=torch.int32),
            block_table=torch.arange(4, dtype=torch.int32).view(1, 4),
            cache_block_size=1,
            index_q=torch.empty(4, 1, 4),
            weights=torch.empty(4, 1),
            compress_ratio=4,
            topk_tokens=512,
        )

        expected = torch.full((4, 512), -1, dtype=torch.int32)
        expected[1, 0] = 0
        expected[2, :2] = torch.tensor([0, 1], dtype=torch.int32)
        expected[3, :3] = torch.tensor([0, 1, 2], dtype=torch.int32)
        self.assertTrue(torch.equal(topk, expected))

    def test_deepseek_v4_indexer_decode_all_candidate_shortcut_skips_q_prepare(self):
        indexer = object.__new__(DeepseekV4Indexer)
        torch.nn.Module.__init__(indexer)
        indexer.use_fp4_cache = True
        indexer.compress_ratio = 4
        indexer.n_head = 2
        indexer.head_dim = DEEPSEEK_V4_INDEXER_DIM
        indexer.topk_tokens = 512
        indexer.topk_buffer = None
        indexer.softmax_scale = DEEPSEEK_V4_INDEXER_DIM**-0.5

        calls: dict[str, int] = {"compressor": 0}

        class FakeCompressor:
            norm = SimpleNamespace(
                weight=torch.ones(DEEPSEEK_V4_INDEXER_DIM),
                variance_epsilon=1.0e-6,
            )

            def __call__(_self, **kwargs):
                self.assertFalse(kwargs["write_compressed_cache"])
                calls["compressor"] += 1

        def unexpected_q_prepare(*_args, **_kwargs):
            raise AssertionError("all-candidate decode should skip Q preparation")

        indexer.compressor = FakeCompressor()
        indexer.wq_b = unexpected_q_prepare
        indexer.weights_proj = unexpected_q_prepare

        positions = torch.tensor([0, 3, 7, 11], dtype=torch.int64)
        metadata = SimpleNamespace(
            token_to_req_indices=torch.zeros(4, dtype=torch.int32),
            block_table=torch.arange(4, dtype=torch.int32).view(1, 4),
        )
        metadata.compressed_slot_mapping = lambda *args, **kwargs: torch.full(
            (4,), -1, dtype=torch.int64
        )
        metadata.compressed_block_table = lambda *args, **kwargs: metadata.block_table
        pool = SimpleNamespace(
            state_block_size=4,
            get_indexer_state_buffer=lambda layer_index: torch.empty(1),
            get_indexer_block_size=lambda layer_index: 1,
            get_indexer_kv_buffer_2d=lambda layer_index: torch.empty(0),
        )
        ctx = SimpleNamespace(
            token_to_kv_pool=pool,
            attn_backend=SimpleNamespace(forward_metadata=metadata),
            forward_mode=ForwardMode.DECODE,
        )

        with (
            patch(
                "tokenspeed.runtime.models.deepseek_v4.deepseek_v4_csa_indexer_cache_insert"
            ) as cache_insert,
            patch(
                "tokenspeed.runtime.models.deepseek_v4.deepseek_v4_prepare_indexer_q_mxfp4",
                side_effect=unexpected_q_prepare,
            ),
            patch(
                "tokenspeed.runtime.models.deepseek_v4.deepseek_v4_prepare_indexer_q_reference",
                side_effect=unexpected_q_prepare,
            ),
        ):
            topk = DeepseekV4Indexer.forward(
                indexer,
                hidden_states=torch.empty(4, 8),
                qr=torch.empty(4, 8),
                positions=positions,
                ctx=ctx,
                out_cache_loc=torch.zeros(4, dtype=torch.int64),
                layer_index=0,
                cos_sin_cache=torch.empty(16, 64),
            )

        self.assertEqual(calls["compressor"], 1)
        cache_insert.assert_called_once()
        expected = torch.full((4, 512), -1, dtype=torch.int32)
        expected[1, 0] = 0
        expected[2, :2] = torch.tensor([0, 1], dtype=torch.int32)
        expected[3, :3] = torch.tensor([0, 1, 2], dtype=torch.int32)
        self.assertTrue(torch.equal(topk, expected))

    def test_deepseek_v4_indexer_decode_max_len_uses_context_or_cache_window(self):
        block_table = torch.zeros((2, 257), dtype=torch.int32)

        with patch.dict(global_server_args_dict, {"max_model_len": 4096}):
            self.assertEqual(
                _deepseek_v4_indexer_decode_max_len(
                    block_table,
                    cache_block_size=64,
                    compress_ratio=4,
                ),
                1024,
            )

        with patch.dict(global_server_args_dict, {"max_model_len": None}):
            self.assertEqual(
                _deepseek_v4_indexer_decode_max_len(
                    block_table,
                    cache_block_size=64,
                    compress_ratio=4,
                ),
                4112,
            )

    def test_deepseek_v4_indexer_topk_reuses_output_buffer(self):
        logits = torch.tensor(
            [
                [0.0, 3.0, 1.0, -float("inf")],
                [4.0, 1.0, 2.0, 3.0],
            ],
            dtype=torch.float32,
        )
        lengths = torch.tensor([3, 4], dtype=torch.int32)
        out = torch.empty((2, 2), dtype=torch.int32)

        actual = _deepseek_v4_indexer_topk_from_logits(
            logits,
            lengths,
            topk_tokens=2,
            out=out,
        )

        self.assertEqual(actual.data_ptr(), out.data_ptr())
        self.assertTrue(torch.equal(actual[0].sort().values, torch.tensor([1, 2])))
        self.assertTrue(torch.equal(actual[1].sort().values, torch.tensor([0, 3])))

    def test_deepseek_v4_topk_buffer_grows_and_reuses(self):
        buffer = _DeepseekV4TopKBuffer(topk_tokens=3)

        first = buffer.get(2, torch.device("cpu"))
        second = buffer.get(1, torch.device("cpu"))
        third = buffer.get(4, torch.device("cpu"))

        self.assertEqual(first.shape, (2, 3))
        self.assertEqual(second.shape, (1, 3))
        self.assertEqual(first.data_ptr(), second.data_ptr())
        self.assertEqual(third.shape, (4, 3))
        self.assertGreaterEqual(buffer.buffer.shape[0], 4)

    def test_deepseek_v4_indexer_prefill_topk_chunks_cap_logits_bytes(self):
        positions = torch.tensor([3, 7, 11, 15], dtype=torch.int64)

        self.assertEqual(
            _deepseek_v4_indexer_prefill_topk_chunks(
                positions,
                compress_ratio=4,
                max_logits_bytes=32,
            ),
            [(0, 2), (2, 4)],
        )
        self.assertEqual(
            _deepseek_v4_indexer_prefill_topk_chunks(
                positions,
                compress_ratio=4,
                max_logits_bytes=64,
            ),
            [(0, 4)],
        )
        self.assertEqual(
            _deepseek_v4_indexer_prefill_topk_chunks(
                torch.tensor([39], dtype=torch.int64),
                compress_ratio=4,
                max_logits_bytes=16,
            ),
            [(0, 1)],
        )

    def test_hidden_compression_helpers_preserve_expected_shapes(self):
        import torch

        torch.manual_seed(0)
        tokens, hc_mult, hidden = 3, 4, 5
        mix_hc = (2 + hc_mult) * hc_mult
        residual = torch.randn(tokens, hc_mult, hidden, dtype=torch.float32)
        fn = torch.randn(mix_hc, hc_mult * hidden, dtype=torch.float32)
        scale = torch.ones(3, dtype=torch.float32)
        base = torch.zeros(mix_hc, dtype=torch.float32)

        layer_input, post, comb = mhc_pre(
            residual,
            fn,
            scale,
            base,
            rms_eps=1e-6,
            hc_eps=1e-6,
            sinkhorn_iters=2,
        )
        updated = mhc_post(layer_input, residual, post, comb)

        self.assertEqual(tuple(layer_input.shape), (tokens, hidden))
        self.assertEqual(tuple(post.shape), (tokens, hc_mult, 1))
        self.assertEqual(tuple(comb.shape), (tokens, hc_mult, hc_mult))
        self.assertEqual(tuple(updated.shape), tuple(residual.shape))

    def test_fast_mhc_is_disabled_by_default_on_sm12x(self):
        with patch(
            "tokenspeed.runtime.models.deepseek_v4._platform",
            SimpleNamespace(is_sm12x=True),
        ):
            self.assertFalse(_deepseek_v4_fast_mhc_enabled_for_platform())

    def test_fast_mhc_remains_enabled_on_non_sm12x(self):
        with patch(
            "tokenspeed.runtime.models.deepseek_v4._platform",
            SimpleNamespace(is_sm12x=False),
        ):
            self.assertTrue(_deepseek_v4_fast_mhc_enabled_for_platform())

    def test_mhc_pre_uses_sm12x_native_path_when_available(self):
        residual = SimpleNamespace(is_cuda=True)
        fn = torch.randn(24, 32, dtype=torch.float32)
        scale = torch.ones(3, dtype=torch.float32)
        base = torch.zeros(24, dtype=torch.float32)
        expected = (
            torch.empty(2, 8, dtype=torch.bfloat16),
            torch.empty(2, 4, 1, dtype=torch.float32),
            torch.empty(2, 4, 4, dtype=torch.float32),
        )

        with (
            patch(
                "tokenspeed.runtime.models.deepseek_v4._platform",
                SimpleNamespace(is_sm12x=True),
            ),
            patch(
                "tokenspeed.runtime.models.deepseek_v4._DEEPSEEK_V4_SM12X_MHC_UNAVAILABLE",
                False,
            ),
            patch(
                "tokenspeed.runtime.models.deepseek_v4._deepseek_v4_sm12x_mhc_input_supported",
                return_value=True,
            ),
            patch(
                "tokenspeed.runtime.layers.deepseek_v4_mhc.sm12x_mhc_pre",
                return_value=expected,
                create=True,
            ) as sm12x_mhc_pre,
            patch(
                "tokenspeed.runtime.models.deepseek_v4._mhc_pre_reference",
                side_effect=AssertionError("SM12x native mHC pre should be used"),
            ),
        ):
            actual = mhc_pre(
                residual,
                fn,
                scale,
                base,
                rms_eps=1e-6,
                hc_eps=1e-6,
                sinkhorn_iters=2,
            )

        self.assertIs(actual, expected)
        sm12x_mhc_pre.assert_called_once_with(
            residual,
            fn,
            scale,
            base,
            1e-6,
            1e-6,
            2,
        )

    def test_mhc_pre_cpu_reference_path_does_not_poison_sm12x_native(self):
        from tokenspeed.runtime.models import deepseek_v4 as deepseek_v4_model

        residual = torch.randn(2, 4, 8, dtype=torch.float32)
        fn = torch.randn(24, 32, dtype=torch.float32)
        scale = torch.ones(3, dtype=torch.float32)
        base = torch.zeros(24, dtype=torch.float32)

        with (
            patch(
                "tokenspeed.runtime.models.deepseek_v4._platform",
                SimpleNamespace(is_sm12x=True),
            ),
            patch(
                "tokenspeed.runtime.models.deepseek_v4._DEEPSEEK_V4_SM12X_MHC_UNAVAILABLE",
                False,
            ),
            patch(
                "tokenspeed.runtime.layers.deepseek_v4_mhc.sm12x_mhc_pre",
                side_effect=AssertionError("CPU tensors must use the reference path"),
                create=True,
            ) as sm12x_mhc_pre,
        ):
            mhc_pre(
                residual,
                fn,
                scale,
                base,
                rms_eps=1e-6,
                hc_eps=1e-6,
                sinkhorn_iters=2,
            )

            sm12x_mhc_pre.assert_not_called()
            self.assertFalse(deepseek_v4_model._DEEPSEEK_V4_SM12X_MHC_UNAVAILABLE)

    def test_mhc_post_uses_sm12x_native_path_when_available(self):
        hidden_states = SimpleNamespace(is_cuda=True)
        residual = SimpleNamespace(is_cuda=True)
        post = torch.empty(2, 4, 1, dtype=torch.float32)
        comb = torch.empty(2, 4, 4, dtype=torch.float32)
        expected = torch.empty(2, 4, 8, dtype=torch.bfloat16)

        with (
            patch(
                "tokenspeed.runtime.models.deepseek_v4._platform",
                SimpleNamespace(is_sm12x=True),
            ),
            patch(
                "tokenspeed.runtime.models.deepseek_v4._DEEPSEEK_V4_SM12X_MHC_UNAVAILABLE",
                False,
            ),
            patch(
                "tokenspeed.runtime.models.deepseek_v4._deepseek_v4_sm12x_mhc_input_supported",
                return_value=True,
            ),
            patch(
                "tokenspeed.runtime.layers.deepseek_v4_mhc.sm12x_mhc_post",
                return_value=expected,
                create=True,
            ) as sm12x_mhc_post,
            patch(
                "tokenspeed.runtime.models.deepseek_v4._mhc_post_reference",
                side_effect=AssertionError("SM12x native mHC post should be used"),
            ),
        ):
            actual = mhc_post(hidden_states, residual, post, comb)

        self.assertIs(actual, expected)
        sm12x_mhc_post.assert_called_once_with(hidden_states, residual, post, comb)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_sm12x_native_mhc_pre_matches_reference(self):
        major, _minor = torch.cuda.get_device_capability()
        if major != 12:
            self.skipTest("SM12x native mHC pre contract only applies to CUDA major 12")

        from tokenspeed.runtime.layers import deepseek_v4_mhc

        torch.manual_seed(20260509)
        tokens, hc_mult, hidden = 2, 4, 128
        mix_hc = (2 + hc_mult) * hc_mult
        residual = torch.randn(
            tokens, hc_mult, hidden, device="cuda", dtype=torch.bfloat16
        )
        fn = (
            torch.randn(mix_hc, hc_mult * hidden, device="cuda", dtype=torch.float32)
            * 0.01
        )
        scale = torch.tensor([0.7, 1.1, 0.5], device="cuda", dtype=torch.float32)
        base = torch.randn(mix_hc, device="cuda", dtype=torch.float32) * 0.01

        actual = deepseek_v4_mhc.sm12x_mhc_pre(
            residual,
            fn,
            scale,
            base,
            1.0e-6,
            1.0e-6,
            2,
        )
        expected = _mhc_pre_reference(
            residual,
            fn,
            scale,
            base,
            1.0e-6,
            1.0e-6,
            2,
        )
        torch.cuda.synchronize()

        torch.testing.assert_close(
            actual[0].float(), expected[0].float(), atol=2.0e-2, rtol=2.0e-2
        )
        torch.testing.assert_close(actual[1], expected[1], atol=2.0e-4, rtol=2.0e-4)
        torch.testing.assert_close(actual[2], expected[2], atol=2.0e-4, rtol=2.0e-4)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_sm12x_native_mhc_pre_split_matches_reference(self):
        major, _minor = torch.cuda.get_device_capability()
        if major != 12:
            self.skipTest("SM12x native mHC pre contract only applies to CUDA major 12")

        from tokenspeed.runtime.layers import deepseek_v4_mhc

        torch.manual_seed(20260511)
        tokens, hc_mult, hidden = 1, 4, 1024
        mix_hc = (2 + hc_mult) * hc_mult
        residual = torch.randn(
            tokens, hc_mult, hidden, device="cuda", dtype=torch.bfloat16
        )
        fn = (
            torch.randn(mix_hc, hc_mult * hidden, device="cuda", dtype=torch.float32)
            * 0.01
        )
        scale = torch.tensor([0.7, 1.1, 0.5], device="cuda", dtype=torch.float32)
        base = torch.randn(mix_hc, device="cuda", dtype=torch.float32) * 0.01

        actual = deepseek_v4_mhc.sm12x_mhc_pre(
            residual,
            fn,
            scale,
            base,
            1.0e-6,
            1.0e-6,
            2,
        )
        expected = _mhc_pre_reference(
            residual,
            fn,
            scale,
            base,
            1.0e-6,
            1.0e-6,
            2,
        )
        torch.cuda.synchronize()

        torch.testing.assert_close(
            actual[0].float(), expected[0].float(), atol=2.0e-2, rtol=2.0e-2
        )
        torch.testing.assert_close(actual[1], expected[1], atol=3.0e-4, rtol=3.0e-4)
        torch.testing.assert_close(actual[2], expected[2], atol=3.0e-4, rtol=3.0e-4)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_sm12x_native_mhc_pre_deepseek_v4_shape_matches_reference(self):
        major, _minor = torch.cuda.get_device_capability()
        if major != 12:
            self.skipTest("SM12x native mHC pre contract only applies to CUDA major 12")

        from tokenspeed.runtime.layers import deepseek_v4_mhc

        torch.manual_seed(20260512)
        tokens, hc_mult, hidden = 1, 4, 4096
        mix_hc = (2 + hc_mult) * hc_mult
        residual = torch.randn(
            tokens, hc_mult, hidden, device="cuda", dtype=torch.bfloat16
        )
        fn = (
            torch.randn(mix_hc, hc_mult * hidden, device="cuda", dtype=torch.float32)
            * 0.01
        )
        scale = torch.tensor([0.7, 1.1, 0.5], device="cuda", dtype=torch.float32)
        base = torch.randn(mix_hc, device="cuda", dtype=torch.float32) * 0.01

        actual = deepseek_v4_mhc.sm12x_mhc_pre(
            residual,
            fn,
            scale,
            base,
            1.0e-6,
            1.0e-6,
            2,
        )
        expected = _mhc_pre_reference(
            residual,
            fn,
            scale,
            base,
            1.0e-6,
            1.0e-6,
            2,
        )
        torch.cuda.synchronize()

        torch.testing.assert_close(
            actual[0].float(), expected[0].float(), atol=2.0e-2, rtol=2.0e-2
        )
        torch.testing.assert_close(actual[1], expected[1], atol=3.0e-4, rtol=3.0e-4)
        torch.testing.assert_close(actual[2], expected[2], atol=3.0e-4, rtol=3.0e-4)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_sm12x_native_mhc_post_matches_reference(self):
        major, _minor = torch.cuda.get_device_capability()
        if major != 12:
            self.skipTest(
                "SM12x native mHC post contract only applies to CUDA major 12"
            )

        from tokenspeed.runtime.layers import deepseek_v4_mhc

        torch.manual_seed(20260510)
        tokens, hc_mult, hidden = 2, 4, 128
        hidden_states = torch.randn(
            tokens,
            hidden,
            device="cuda",
            dtype=torch.bfloat16,
        )
        residual = torch.randn(
            tokens,
            hc_mult,
            hidden,
            device="cuda",
            dtype=torch.bfloat16,
        )
        post = torch.randn(tokens, hc_mult, 1, device="cuda", dtype=torch.float32)
        comb = torch.randn(tokens, hc_mult, hc_mult, device="cuda", dtype=torch.float32)

        expected = _mhc_post_reference(hidden_states, residual, post, comb)
        actual = deepseek_v4_mhc.sm12x_mhc_post(hidden_states, residual, post, comb)
        torch.cuda.synchronize()

        torch.testing.assert_close(
            actual.float(),
            expected.float(),
            atol=2.0e-2,
            rtol=2.0e-2,
        )

    def test_trtllm_fp8_activation_quant_is_disabled_by_default_on_sm12x(self):
        with patch(
            "tokenspeed.runtime.models.deepseek_v4._platform",
            SimpleNamespace(is_sm12x=True),
        ):
            self.assertFalse(_deepseek_v4_trtllm_fp8_quant_enabled_for_platform())

    def test_trtllm_fp8_activation_quant_remains_enabled_on_non_sm12x(self):
        with patch(
            "tokenspeed.runtime.models.deepseek_v4._platform",
            SimpleNamespace(is_sm12x=False),
        ):
            self.assertTrue(_deepseek_v4_trtllm_fp8_quant_enabled_for_platform())

    def test_deepgemm_fp4_indexer_is_disabled_by_default_on_sm12x(self):
        with patch(
            "tokenspeed.runtime.models.deepseek_v4._platform",
            SimpleNamespace(is_sm12x=True),
        ):
            self.assertFalse(_deepseek_v4_deepgemm_fp4_indexer_enabled_for_platform())

    def test_deepgemm_fp4_indexer_remains_enabled_on_non_sm12x(self):
        with patch(
            "tokenspeed.runtime.models.deepseek_v4._platform",
            SimpleNamespace(is_sm12x=False),
        ):
            self.assertTrue(_deepseek_v4_deepgemm_fp4_indexer_enabled_for_platform())

    def test_fast_topk_is_disabled_by_default_on_sm12x(self):
        with patch(
            "tokenspeed.runtime.models.deepseek_v4._platform",
            SimpleNamespace(is_sm12x=True),
        ):
            self.assertFalse(_deepseek_v4_fast_topk_enabled_for_platform())

    def test_fast_topk_remains_enabled_on_non_sm12x(self):
        with patch(
            "tokenspeed.runtime.models.deepseek_v4._platform",
            SimpleNamespace(is_sm12x=False),
        ):
            self.assertTrue(_deepseek_v4_fast_topk_enabled_for_platform())

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
    def test_indexer_topk_batched_uses_torch_topk_when_fast_topk_is_disabled(self):
        cache = torch.eye(4, device="cuda", dtype=torch.bfloat16)

        def cache_reader(cache_2d, slot_mapping, block_size):
            del block_size
            return cache_2d[slot_mapping.long()]

        with (
            patch(
                "tokenspeed.runtime.models.deepseek_v4._deepseek_v4_fast_topk_enabled_for_platform",
                return_value=False,
            ),
            patch(
                "tokenspeed_kernel.thirdparty.trtllm.fast_topk_v2",
                side_effect=AssertionError("SM12x must not call fast_topk_v2"),
            ),
        ):
            topk = _deepseek_v4_indexer_topk_from_cache_batched(
                cache_reader=cache_reader,
                cache_2d=cache,
                positions=torch.tensor([3], device="cuda", dtype=torch.int64),
                token_to_req_indices=torch.tensor(
                    [0], device="cuda", dtype=torch.int32
                ),
                block_table=torch.arange(4, device="cuda", dtype=torch.int32).view(
                    1, 4
                ),
                cache_block_size=1,
                index_q=torch.ones(1, 1, 4, device="cuda", dtype=torch.bfloat16),
                weights=torch.ones(1, 1, device="cuda", dtype=torch.float32),
                compress_ratio=1,
                topk_tokens=512,
            )

        self.assertEqual(tuple(topk.shape), (1, 512))
        self.assertTrue(torch.all(topk[0, :4] >= 0))
        self.assertTrue(torch.all(topk[0, 4:] == -1))

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
    def test_indexer_topk_batched_is_cuda_graph_capturable(self):
        torch.manual_seed(1234)
        device = torch.device("cuda")
        positions = torch.tensor([15, 7], device=device, dtype=torch.int64)
        token_to_req_indices = torch.tensor([0, 1], device=device, dtype=torch.int32)
        block_table = torch.tensor(
            [[0, 1, 2, 3], [4, 5, 6, 7]],
            device=device,
            dtype=torch.int32,
        )
        cache = torch.randn(8, 4, device=device, dtype=torch.bfloat16)
        index_q = torch.randn(2, 2, 4, device=device, dtype=torch.bfloat16)
        weights = torch.randn(2, 2, device=device, dtype=torch.float32)
        out = torch.empty(2, 4, device=device, dtype=torch.int32)

        def cache_reader(cache_2d, slot_mapping, block_size):
            del block_size
            return cache_2d[slot_mapping.long()]

        with patch.dict(global_server_args_dict, {"max_model_len": 16}, clear=False):
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                for _ in range(2):
                    _deepseek_v4_indexer_topk_from_cache_batched(
                        cache_reader=cache_reader,
                        cache_2d=cache,
                        positions=positions,
                        token_to_req_indices=token_to_req_indices,
                        block_table=block_table,
                        cache_block_size=1,
                        index_q=index_q,
                        weights=weights,
                        compress_ratio=4,
                        topk_tokens=4,
                        out=out,
                    )
            stream.synchronize()

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                _deepseek_v4_indexer_topk_from_cache_batched(
                    cache_reader=cache_reader,
                    cache_2d=cache,
                    positions=positions,
                    token_to_req_indices=token_to_req_indices,
                    block_table=block_table,
                    cache_block_size=1,
                    index_q=index_q,
                    weights=weights,
                    compress_ratio=4,
                    topk_tokens=4,
                    out=out,
                )
            graph.replay()

        self.assertEqual(tuple(out.shape), (2, 4))

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
    def test_deepgemm_fp4_indexer_available_respects_sm12x_gate(self):
        fake_deep_gemm = SimpleNamespace(
            fp8_fp4_mqa_logits=object(),
            fp8_fp4_paged_mqa_logits=object(),
            get_paged_mqa_logits_metadata=object(),
        )
        index_q = torch.empty(
            (1, 32, DEEPSEEK_V4_INDEXER_DIM // 2),
            device="cuda",
            dtype=torch.uint8,
        )

        with (
            patch(
                "tokenspeed.runtime.models.deepseek_v4._platform",
                SimpleNamespace(is_sm12x=True),
            ),
            patch("tokenspeed.runtime.models.deepseek_v4.deep_gemm", fake_deep_gemm),
        ):
            self.assertFalse(_deepseek_v4_deepgemm_fp4_indexer_available(index_q))

        with (
            patch(
                "tokenspeed.runtime.models.deepseek_v4._platform",
                SimpleNamespace(is_sm12x=False),
            ),
            patch("tokenspeed.runtime.models.deepseek_v4.deep_gemm", fake_deep_gemm),
        ):
            self.assertTrue(_deepseek_v4_deepgemm_fp4_indexer_available(index_q))

    def test_hidden_compression_pre_matches_reference_math(self):
        import torch
        import torch.nn.functional as F

        torch.manual_seed(1)
        tokens, hc_mult, hidden = 2, 3, 4
        mix_hc = (2 + hc_mult) * hc_mult
        residual = torch.randn(tokens, hc_mult, hidden, dtype=torch.bfloat16)
        fn = torch.randn(mix_hc, hc_mult * hidden, dtype=torch.float32)
        scale = torch.tensor([0.7, 1.1, 0.5], dtype=torch.float32)
        base = torch.randn(mix_hc, dtype=torch.float32)
        eps = 1e-5

        layer_input, post, comb = mhc_pre(
            residual, fn, scale, base, rms_eps=1e-6, hc_eps=eps, sinkhorn_iters=3
        )

        x = residual.flatten(1).float()
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)
        mixes = F.linear(x, fn) * rsqrt
        pre_raw, post_raw, comb_raw = torch.split(
            mixes, [hc_mult, hc_mult, hc_mult * hc_mult], dim=-1
        )
        pre_base, post_base, comb_base = torch.split(
            base, [hc_mult, hc_mult, hc_mult * hc_mult], dim=-1
        )
        expected_pre = torch.sigmoid(pre_raw * scale[0] + pre_base) + eps
        expected_post = (
            torch.sigmoid(post_raw * scale[1] + post_base) * 2.0
        ).unsqueeze(-1)
        expected_comb = (
            F.softmax(
                comb_raw.reshape(tokens, hc_mult, hc_mult) * scale[2]
                + comb_base.reshape(1, hc_mult, hc_mult),
                dim=-1,
            )
            + eps
        )
        expected_comb = expected_comb / (expected_comb.sum(dim=-2, keepdim=True) + eps)
        for _ in range(2):
            expected_comb = expected_comb / (
                expected_comb.sum(dim=-1, keepdim=True) + eps
            )
            expected_comb = expected_comb / (
                expected_comb.sum(dim=-2, keepdim=True) + eps
            )
        expected_layer_input = torch.sum(
            expected_pre.unsqueeze(-1) * residual.float(), dim=1
        ).to(residual.dtype)

        self.assertTrue(torch.allclose(layer_input, expected_layer_input))
        self.assertTrue(torch.allclose(post, expected_post))
        self.assertTrue(torch.allclose(comb, expected_comb))

    def test_hidden_compression_post_matches_lane_orientation(self):
        import torch

        hidden_states = torch.tensor([[10.0, 20.0]], dtype=torch.float32)
        residual = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.float32)
        post = torch.tensor([[[0.5], [0.25]]], dtype=torch.float32)
        comb = torch.tensor([[[0.1, 0.2], [0.3, 0.4]]], dtype=torch.float32)

        updated = mhc_post(hidden_states, residual, post, comb)

        expected = torch.empty_like(residual)
        expected[:, 0] = (
            comb[:, 0, 0:1] * residual[:, 0]
            + comb[:, 1, 0:1] * residual[:, 1]
            + post[:, 0] * hidden_states
        )
        expected[:, 1] = (
            comb[:, 0, 1:2] * residual[:, 0]
            + comb[:, 1, 1:2] * residual[:, 1]
            + post[:, 1] * hidden_states
        )
        self.assertTrue(torch.allclose(updated, expected))

    def test_hc_head_matches_shape_contract(self):
        import torch

        tokens, hc_mult, hidden = 2, 4, 6
        x = torch.randn(tokens, hc_mult, hidden)
        fn = torch.randn(hc_mult, hc_mult * hidden)
        scale = torch.ones(1)
        base = torch.zeros(hc_mult)

        y = hc_head(x, fn, scale, base, rms_norm_eps=1e-6, hc_eps=1e-6)

        self.assertEqual(tuple(y.shape), (tokens, hidden))

    def test_deepseek_v4_router_matches_noaux_bias_semantics(self):
        import torch
        import torch.nn.functional as F

        logits = torch.tensor(
            [
                [0.2, 1.0, -0.5, 0.7],
                [1.5, -0.3, 0.8, 0.0],
            ],
            dtype=torch.float32,
        )
        bias = torch.tensor([0.0, -0.4, 0.6, 0.0], dtype=torch.float32)

        topk_weights, topk_ids, scores = deepseek_v4_select_experts(
            logits,
            top_k=2,
            renormalize=True,
            correction_bias=bias,
        )

        expected_scores = F.softplus(logits).sqrt()
        expected_ids = torch.topk(expected_scores + bias, k=2, dim=-1, sorted=False)[1]
        expected_weights = expected_scores.gather(1, expected_ids)
        expected_weights = expected_weights / expected_weights.sum(dim=-1, keepdim=True)

        self.assertTrue(torch.allclose(scores, expected_scores))
        self.assertTrue(torch.equal(topk_ids, expected_ids.to(torch.int32)))
        self.assertTrue(torch.allclose(topk_weights, expected_weights))

    def test_deepseek_v4_hash_router_uses_table_ids_and_gate_scores(self):
        import torch
        import torch.nn.functional as F

        logits = torch.tensor(
            [
                [0.5, 1.0, -0.5, 0.1],
                [-0.2, 0.3, 1.4, 0.0],
            ],
            dtype=torch.float32,
        )
        input_ids = torch.tensor([3, 1], dtype=torch.long)
        table = torch.tensor(
            [
                [0, 1],
                [2, 3],
                [1, 0],
                [3, 1],
            ],
            dtype=torch.int32,
        )

        topk_weights, topk_ids, _ = deepseek_v4_select_experts(
            logits,
            top_k=2,
            renormalize=True,
            hash_indices_table=table,
            input_ids=input_ids,
        )

        expected_ids = torch.tensor([[3, 1], [2, 3]], dtype=torch.int32)
        expected_scores = F.softplus(logits).sqrt()
        expected_weights = expected_scores.gather(1, expected_ids.long())
        expected_weights = expected_weights / expected_weights.sum(dim=-1, keepdim=True)

        self.assertTrue(torch.equal(topk_ids, expected_ids))
        self.assertTrue(torch.allclose(topk_weights, expected_weights))

    def test_deepseek_v4_gate_fallback_returns_fp32_logits(self):
        import torch
        import torch.nn.functional as F

        config = SimpleNamespace(
            n_routed_experts=4,
            hidden_size=8,
            num_hash_layers=0,
            topk_method=None,
        )
        gate = DeepseekV4MoEGate(config, layer_index=1)
        hidden_states = torch.randn(3, config.hidden_size)

        logits = gate(hidden_states)
        expected = F.linear(hidden_states, gate.weight, None).float()

        self.assertEqual(logits.dtype, torch.float32)
        self.assertTrue(torch.allclose(logits, expected))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_deepseek_v4_gate_dsv3_router_gemm_shape(self):
        import torch

        major, _ = torch.cuda.get_device_capability()
        if major < 9:
            self.skipTest("DSV3 router GEMM requires SM90+")

        config = SimpleNamespace(
            n_routed_experts=256,
            hidden_size=4096,
            num_hash_layers=0,
            topk_method=None,
        )
        gate = DeepseekV4MoEGate(config, layer_index=1).cuda().to(torch.bfloat16)
        hidden_states = torch.randn(
            2, config.hidden_size, device="cuda", dtype=torch.bfloat16
        )

        try:
            logits = gate(hidden_states)
        except RuntimeError as exc:
            if "dsv3_gemm library not found" not in str(exc):
                raise
            self.skipTest(str(exc))
        torch.cuda.synchronize()

        self.assertEqual(tuple(logits.shape), (2, config.n_routed_experts))
        self.assertEqual(logits.dtype, torch.float32)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_deepseek_v4_fused_softplus_sqrt_topk_matches_reference(self):
        import torch
        import torch.nn.functional as F
        from tokenspeed_kernel.thirdparty.cuda.routing import (
            softplus_sqrt_topk_flash,
        )

        logits = torch.linspace(
            -3.0, 3.0, 256, device="cuda", dtype=torch.float32
        ).repeat(3, 1)
        bias = torch.linspace(0.25, -0.25, 256, device="cuda", dtype=torch.float32)
        topk_weights = torch.empty(3, 6, device="cuda", dtype=torch.float32)
        topk_ids = torch.empty(3, 6, device="cuda", dtype=torch.int32)

        try:
            softplus_sqrt_topk_flash(logits, bias, topk_ids, topk_weights, 1.0, True)
        except (AttributeError, RuntimeError) as exc:
            self.skipTest(f"fused DeepSeek V4 router op unavailable: {exc}")
        torch.cuda.synchronize()

        scores = F.softplus(logits).sqrt()
        expected_ids = torch.topk(scores + bias, k=6, dim=-1, sorted=True)[1]
        expected_weights = scores.gather(1, expected_ids)
        expected_weights = expected_weights / expected_weights.sum(dim=-1, keepdim=True)

        self.assertTrue(torch.equal(topk_ids, expected_ids.to(torch.int32)))
        self.assertTrue(torch.allclose(topk_weights, expected_weights, atol=1e-6))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_deepseek_v4_fused_select_experts_returns_scores(self):
        import torch
        import torch.nn.functional as F

        logits = torch.linspace(
            -3.0, 3.0, 256, device="cuda", dtype=torch.float32
        ).repeat(2, 1)
        bias = torch.linspace(0.25, -0.25, 256, device="cuda", dtype=torch.float32)

        topk_weights, topk_ids, scores = deepseek_v4_select_experts(
            logits,
            top_k=6,
            renormalize=True,
            correction_bias=bias,
        )

        expected_scores = F.softplus(logits).sqrt()
        expected_ids = torch.topk(expected_scores + bias, k=6, dim=-1, sorted=True)[1]
        expected_weights = expected_scores.gather(1, expected_ids)
        expected_weights = expected_weights / expected_weights.sum(dim=-1, keepdim=True)

        self.assertTrue(torch.allclose(scores, expected_scores))
        self.assertTrue(torch.equal(topk_ids, expected_ids.to(torch.int32)))
        self.assertTrue(torch.allclose(topk_weights, expected_weights, atol=1e-6))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_deepseek_v4_bias_fused_router_runs_by_default(self):
        import torch

        logits = torch.zeros(2, 256, device="cuda", dtype=torch.float32)
        bias = torch.linspace(0.25, -0.25, 256, device="cuda", dtype=torch.float32)

        out = _deepseek_v4_fused_select_experts(
            logits, top_k=6, renormalize=True, correction_bias=bias
        )

        if out is None:
            self.skipTest("fused DeepSeek V4 router op unavailable")
        topk_weights, topk_ids = out
        self.assertEqual(tuple(topk_weights.shape), (2, 6))
        self.assertEqual(tuple(topk_ids.shape), (2, 6))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_deepseek_v4_fused_hash_topk_matches_reference(self):
        import torch
        import torch.nn.functional as F
        from tokenspeed_kernel.thirdparty.cuda.routing import (
            hash_softplus_sqrt_topk_flash,
        )

        logits = torch.linspace(
            -2.0, 2.0, 256, device="cuda", dtype=torch.float32
        ).repeat(3, 1)
        input_ids = torch.tensor([1, 0, 1], device="cuda", dtype=torch.long)
        table = torch.tensor(
            [[5, 7, 11, 13, 17, 19], [23, 29, 31, 37, 41, 43]],
            device="cuda",
            dtype=torch.int32,
        )
        topk_weights = torch.empty(3, 6, device="cuda", dtype=torch.float32)
        topk_ids = torch.empty(3, 6, device="cuda", dtype=torch.int32)

        try:
            hash_softplus_sqrt_topk_flash(
                logits, input_ids, table, topk_ids, topk_weights, 1.0, True
            )
        except (AttributeError, RuntimeError) as exc:
            self.skipTest(f"fused DeepSeek V4 hash router op unavailable: {exc}")
        torch.cuda.synchronize()

        expected_ids = table[input_ids]
        scores = F.softplus(logits).sqrt()
        expected_weights = scores.gather(1, expected_ids.long())
        expected_weights = expected_weights / expected_weights.sum(dim=-1, keepdim=True)

        self.assertTrue(torch.equal(topk_ids, expected_ids))
        self.assertTrue(torch.allclose(topk_weights, expected_weights, atol=1e-6))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_deepseek_v4_fp8_activation_quant_matches_reference(self):
        import torch

        x = torch.randn(5, 256, device="cuda", dtype=torch.bfloat16) * 3.0

        actual = _fp8_act_quant_dequant(x, 128)

        x_blocks = x.float().reshape(-1, x.shape[-1]).unflatten(-1, (-1, 128))
        amax = x_blocks.abs().amax(dim=-1).clamp_min(1.0e-4)
        scale = torch.pow(2.0, torch.ceil(torch.log2(amax / 448.0)))
        scale = scale.to(torch.float8_e8m0fnu).float()
        quantized = (
            (x_blocks / scale.unsqueeze(-1))
            .clamp(-448.0, 448.0)
            .to(torch.float8_e4m3fn)
        )
        expected = (quantized.float() * scale.unsqueeze(-1)).flatten(-2).reshape_as(x)

        self.assertTrue(torch.equal(actual, expected))

    def test_deepseek_v4_fp8_activation_quant_uses_sm12x_native_path(self):
        sentinel = object()
        x = SimpleNamespace(shape=(1, 128), is_cuda=True, dtype=torch.bfloat16)
        with (
            patch(
                "tokenspeed.runtime.models.deepseek_v4._platform",
                SimpleNamespace(is_sm12x=True),
            ),
            patch(
                "tokenspeed_kernel.ops.gemm.sm12x_fp8."
                "sm12x_mxfp8_block128_quant_dequant_ue8m0",
                return_value=sentinel,
            ) as native_quant_dequant,
        ):
            actual = _fp8_act_quant_dequant(x, 128)

        self.assertIs(actual, sentinel)
        native_quant_dequant.assert_called_once_with(x)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_deepseek_v4_fp8_linear_uses_sm12x_weight_gemv(self):
        major, _minor = torch.cuda.get_device_capability()
        if major != 12:
            self.skipTest("SM12x FP8 weight GEMV only applies to CUDA major 12")

        torch.manual_seed(20260513)
        rows, hidden, out_dim = 1, 256, 384
        x = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16) * 2.0
        weight = (
            torch.randn(out_dim, hidden, device="cuda", dtype=torch.bfloat16) * 0.25
        ).to(torch.float8_e4m3fn)
        raw_scales = torch.randn(
            (out_dim + 127) // 128,
            hidden // 128,
            device="cuda",
            dtype=torch.float32,
        ).abs()
        scales = torch.pow(
            2.0,
            torch.ceil(torch.log2(raw_scales.clamp_min(1.0e-4))),
        ).to(torch.float8_e8m0fnu)
        layer = SimpleNamespace(
            weight=weight,
            weight_scale_inv=scales,
            quant_config=SimpleNamespace(weight_block_size=(128, 128)),
        )

        from tokenspeed_kernel.ops.gemm import sm12x_fp8

        calls = []
        real_gemv = sm12x_fp8.sm12x_fp8_weight_gemv_ue8m0

        def wrapped_gemv(*args, **kwargs):
            calls.append(args)
            return real_gemv(*args, **kwargs)

        with (
            patch(
                "tokenspeed.runtime.models.deepseek_v4._platform",
                SimpleNamespace(is_sm12x=True),
            ),
            patch(
                "tokenspeed_kernel.ops.gemm.sm12x_fp8." "sm12x_fp8_weight_gemv_ue8m0",
                side_effect=wrapped_gemv,
            ),
        ):
            actual = _fp8_linear(layer, x, (out_dim, hidden))

        x_eff = _fp8_act_quant_dequant(x, 128)
        expanded_scales = scales.float().repeat_interleave(128, dim=0)
        expanded_scales = expanded_scales.repeat_interleave(128, dim=1)[
            :out_dim, :hidden
        ]
        expected = (x_eff @ (weight.float() * expanded_scales).T).to(torch.bfloat16)

        torch.cuda.synchronize()
        self.assertEqual(len(calls), 1)
        torch.testing.assert_close(actual.float(), expected.float(), rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_deepseek_v4_fp8_linear_keeps_prefill_on_fp32_weight_matmul(self):
        major, _minor = torch.cuda.get_device_capability()
        if major != 12:
            self.skipTest("SM12x FP8 weight GEMV only applies to CUDA major 12")

        torch.manual_seed(20260513)
        rows, hidden, out_dim = 17, 256, 384
        x = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16) * 2.0
        weight = (
            torch.randn(out_dim, hidden, device="cuda", dtype=torch.bfloat16) * 0.25
        ).to(torch.float8_e4m3fn)
        raw_scales = torch.randn(
            (out_dim + 127) // 128,
            hidden // 128,
            device="cuda",
            dtype=torch.float32,
        ).abs()
        scales = torch.pow(
            2.0,
            torch.ceil(torch.log2(raw_scales.clamp_min(1.0e-4))),
        ).to(torch.float8_e8m0fnu)
        layer = SimpleNamespace(
            weight=weight,
            weight_scale_inv=scales,
            quant_config=SimpleNamespace(weight_block_size=(128, 128)),
        )

        from tokenspeed_kernel.ops.gemm import sm12x_fp8

        calls = []
        real_gemv = sm12x_fp8.sm12x_fp8_weight_gemv_ue8m0

        def wrapped_gemv(*args, **kwargs):
            calls.append(args)
            return real_gemv(*args, **kwargs)

        with (
            patch(
                "tokenspeed.runtime.models.deepseek_v4._platform",
                SimpleNamespace(is_sm12x=True),
            ),
            patch(
                "tokenspeed_kernel.ops.gemm.sm12x_fp8." "sm12x_fp8_weight_gemv_ue8m0",
                side_effect=wrapped_gemv,
            ),
        ):
            actual = _fp8_linear(layer, x, (out_dim, hidden))

        x_eff = _fp8_act_quant_dequant(x, 128)
        expanded_scales = scales.float().repeat_interleave(128, dim=0)
        expanded_scales = expanded_scales.repeat_interleave(128, dim=1)[
            :out_dim, :hidden
        ]
        expected = (x_eff @ (weight.float() * expanded_scales).T).to(torch.bfloat16)

        torch.cuda.synchronize()
        self.assertEqual(len(calls), 0)
        torch.testing.assert_close(actual.float(), expected.float(), rtol=0, atol=0)

    def test_packed_topk_router_logits_recover_weights_after_softmax(self):
        import torch

        topk_ids = torch.tensor([[3, 1], [2, 0]], dtype=torch.int32)
        topk_weights = torch.tensor([[0.7, 0.3], [0.55, 0.45]], dtype=torch.float32)

        packed = pack_topk_as_router_logits(topk_weights, topk_ids, num_experts=4)
        recovered = packed.softmax(dim=-1).gather(1, topk_ids.long())

        self.assertTrue(torch.allclose(recovered, topk_weights))

    def test_mxfp4_flashinfer_reorders_w1w3_halves_for_trtllm(self):
        import torch

        weight = torch.arange(4, dtype=torch.uint8).reshape(1, 4, 1)
        scale = torch.arange(8, dtype=torch.uint8).reshape(1, 4, 2)
        bias = torch.arange(4, dtype=torch.float32).reshape(1, 4)

        self.assertTrue(
            torch.equal(
                _reorder_w1w3_to_w3w1(weight, -2).flatten(),
                torch.tensor([2, 3, 0, 1], dtype=torch.uint8),
            )
        )
        self.assertTrue(
            torch.equal(
                _reorder_w1w3_to_w3w1(scale, -2).flatten(),
                torch.tensor([4, 5, 6, 7, 0, 1, 2, 3], dtype=torch.uint8),
            )
        )
        self.assertTrue(
            torch.equal(
                _reorder_w1w3_to_w3w1(bias, -1).flatten(),
                torch.tensor([2, 3, 0, 1], dtype=torch.float32),
            )
        )
        if hasattr(torch, "float8_e8m0fnu"):
            scale_f8 = torch.tensor(
                [[0.0078125, 0.015625, 0.03125, 0.0625]], dtype=torch.float32
            ).to(torch.float8_e8m0fnu)
            reordered = _reorder_w1w3_to_w3w1(scale_f8, -1)
            self.assertEqual(reordered.dtype, torch.float8_e8m0fnu)
            self.assertTrue(
                torch.equal(
                    reordered.view(torch.uint8),
                    torch.tensor([[122, 123, 120, 121]], dtype=torch.uint8),
                )
            )

    def test_mxfp4_flashinfer_uses_gated_permute_for_w13(self):
        import torch
        from tokenspeed_kernel.ops.moe.flashinfer import (
            _maybe_get_cached_w3_w1_permute_indices,
            get_w2_permute_indices_with_cache,
        )

        x = torch.empty((4096, 2048), dtype=torch.uint8)
        expected_w13 = _maybe_get_cached_w3_w1_permute_indices({}, x, 128)
        expected_w2 = get_w2_permute_indices_with_cache({}, x, 128)

        actual_w13 = _get_flashinfer_mxfp4_device_permute_indices(x, 128, kind="w13")
        actual_w2 = _get_flashinfer_mxfp4_device_permute_indices(x, 128, kind="w2")

        self.assertTrue(torch.equal(actual_w13.cpu(), expected_w13.cpu()))
        self.assertTrue(torch.equal(actual_w2.cpu(), expected_w2.cpu()))
        self.assertFalse(torch.equal(actual_w13.cpu(), actual_w2.cpu()))

    def test_c4_ape_reorder_matches_overlap_window_layout(self):
        import torch

        ape = torch.arange(4 * 8, dtype=torch.float32).reshape(4, 8)

        reordered = _deepseek_v4_reorder_c4_ape_2604(ape)
        expected = torch.tensor(
            [
                [0, 1, 2, 3, 8, 9, 10, 11],
                [16, 17, 18, 19, 24, 25, 26, 27],
                [4, 5, 6, 7, 12, 13, 14, 15],
                [20, 21, 22, 23, 28, 29, 30, 31],
            ],
            dtype=torch.float32,
        )

        self.assertTrue(torch.equal(reordered, expected))


if __name__ == "__main__":
    unittest.main()
