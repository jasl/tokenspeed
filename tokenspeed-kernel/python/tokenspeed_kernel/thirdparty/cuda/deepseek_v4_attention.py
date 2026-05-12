"""DeepSeek V4 attention CUDA kernel wrappers."""

from __future__ import annotations

import functools
from pathlib import Path

import torch
import tvm_ffi

_SM12X_MHC_PRE_SPLIT_WIDTH = 1024
_SM12X_MHC_PRE_MAX_SPLITS = 16


def _objs_dir() -> Path:
    return Path(__file__).resolve().parent / "objs"


@functools.cache
def _load_deepseek_v4_attention_module():
    so_path = _objs_dir() / "deepseek_v4_attention" / "deepseek_v4_attention.so"
    if not so_path.exists():
        raise RuntimeError(
            f"tokenspeed_kernel DeepSeek V4 attention library not found at {so_path}. "
            "Run `pip install -e tokenspeed-kernel/python/` to build."
        )
    return tvm_ffi.load_module(str(so_path))


def has_fused_qnorm_rope_kv_insert() -> bool:
    try:
        _load_deepseek_v4_attention_module()
    except Exception:
        return False
    return True


def fused_qnorm_rope_kv_insert(
    q: torch.Tensor,
    kv: torch.Tensor,
    k_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    rms_norm_eps: float,
    block_size: int,
) -> None:
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"q must be float16 or bfloat16, got {q.dtype}")
    if kv.dtype != q.dtype:
        raise TypeError(f"kv dtype {kv.dtype} must match q dtype {q.dtype}")
    if k_cache.dtype != torch.uint8:
        raise TypeError(f"k_cache must be uint8, got {k_cache.dtype}")
    if cos_sin_cache.dtype != torch.float32:
        raise TypeError(f"cos_sin_cache must be float32, got {cos_sin_cache.dtype}")
    if slot_mapping.dtype != torch.int64:
        slot_mapping = slot_mapping.to(torch.int64)
    if positions.dtype != torch.int64:
        positions = positions.to(torch.int64)

    _load_deepseek_v4_attention_module().fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
        q,
        kv,
        k_cache,
        slot_mapping.contiguous(),
        positions.contiguous(),
        cos_sin_cache.contiguous(),
        float(rms_norm_eps),
        int(block_size),
    )


def _sm12x_mhc_pre_split_count(*, hc_mult: int, hidden_size: int) -> int:
    width = int(hc_mult) * int(hidden_size)
    if width <= 0:
        return 1
    splits = (width + _SM12X_MHC_PRE_SPLIT_WIDTH - 1) // _SM12X_MHC_PRE_SPLIT_WIDTH
    return max(1, min(_SM12X_MHC_PRE_MAX_SPLITS, splits))


def sm12x_mhc_pre(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if residual.dtype != torch.bfloat16:
        raise RuntimeError(
            f"SM12x mHC pre requires bf16 residual, got {residual.dtype}"
        )
    if fn.dtype != torch.float32:
        raise RuntimeError(f"SM12x mHC pre requires fp32 fn, got {fn.dtype}")
    if hc_scale.dtype != torch.float32 or hc_base.dtype != torch.float32:
        raise RuntimeError("SM12x mHC pre requires fp32 scale/base")
    if not residual.is_cuda:
        raise RuntimeError("SM12x mHC pre requires CUDA tensors")
    if residual.dim() < 2:
        raise ValueError(f"residual must end with [hc, hidden], got {residual.shape}")
    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    outer_shape = residual.shape[:-2]
    num_tokens = residual.numel() // (hc_mult * hidden_size)
    if num_tokens > 16:
        raise RuntimeError("SM12x mHC pre native path is decode-only for <=16 tokens")

    residual_flat = residual.contiguous().reshape(num_tokens, hc_mult, hidden_size)
    layer_input = torch.empty(
        (num_tokens, hidden_size), dtype=residual.dtype, device=residual.device
    )
    post = torch.empty(
        (num_tokens, hc_mult, 1), dtype=torch.float32, device=residual.device
    )
    comb = torch.empty(
        (num_tokens, hc_mult, hc_mult), dtype=torch.float32, device=residual.device
    )
    module = _load_deepseek_v4_attention_module()
    fn = fn.contiguous()
    hc_scale = hc_scale.contiguous()
    hc_base = hc_base.contiguous()
    num_splits = _sm12x_mhc_pre_split_count(
        hc_mult=hc_mult,
        hidden_size=hidden_size,
    )
    if num_splits > 1:
        mix_hc = hc_mult * (2 + hc_mult)
        partials = torch.empty(
            (num_tokens, num_splits, mix_hc + 1),
            dtype=torch.float32,
            device=residual.device,
        )
        module.deepseek_v4_sm12x_mhc_pre_split_cuda(
            residual_flat,
            fn,
            hc_scale,
            hc_base,
            layer_input,
            post,
            comb,
            partials,
            float(rms_eps),
            float(hc_eps),
            int(sinkhorn_iters),
        )
    else:
        module.deepseek_v4_sm12x_mhc_pre_cuda(
            residual_flat,
            fn,
            hc_scale,
            hc_base,
            layer_input,
            post,
            comb,
            float(rms_eps),
            float(hc_eps),
            int(sinkhorn_iters),
        )
    return (
        layer_input.reshape(*outer_shape, hidden_size),
        post.reshape(*outer_shape, hc_mult, 1),
        comb.reshape(*outer_shape, hc_mult, hc_mult),
    )


def sm12x_mhc_post(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    if hidden_states.dtype != torch.bfloat16 or residual.dtype != torch.bfloat16:
        raise RuntimeError("SM12x mHC post requires bf16 hidden/residual tensors")
    if post.dtype != torch.float32 or comb.dtype != torch.float32:
        raise RuntimeError("SM12x mHC post requires fp32 post/comb tensors")
    if not hidden_states.is_cuda or not residual.is_cuda:
        raise RuntimeError("SM12x mHC post requires CUDA tensors")
    if residual.dim() < 2:
        raise ValueError(f"residual must end with [hc, hidden], got {residual.shape}")
    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    outer_shape = residual.shape[:-2]
    num_tokens = residual.numel() // (hc_mult * hidden_size)
    if num_tokens > 16:
        raise RuntimeError("SM12x mHC post native path is decode-only for <=16 tokens")

    hidden_flat = hidden_states.contiguous().reshape(num_tokens, hidden_size)
    residual_flat = residual.contiguous().reshape(num_tokens, hc_mult, hidden_size)
    post_flat = post
    if post_flat.dim() == len(outer_shape) + 1:
        post_flat = post_flat.unsqueeze(-1)
    post_flat = post_flat.contiguous().reshape(num_tokens, hc_mult, 1)
    comb_flat = comb.contiguous().reshape(num_tokens, hc_mult, hc_mult)
    out = torch.empty_like(residual_flat)
    _load_deepseek_v4_attention_module().deepseek_v4_sm12x_mhc_post_cuda(
        hidden_flat,
        residual_flat,
        post_flat,
        comb_flat,
        out,
    )
    return out.reshape(*outer_shape, hc_mult, hidden_size)


def _flatten_sparse_mla_kv(kv: torch.Tensor) -> torch.Tensor:
    if kv.dim() == 2:
        return kv
    if kv.dim() == 3:
        return kv.reshape(-1, kv.shape[-1])
    if kv.dim() == 4 and kv.shape[-2] == 1:
        return kv.reshape(-1, kv.shape[-1])
    raise ValueError(f"kv must be [N, D], [B, N, D], or [N, 1, D], got {kv.shape}")


def _flatten_sparse_mla_indices(indices: torch.Tensor) -> torch.Tensor:
    if indices.dim() == 2:
        return indices
    if indices.dim() == 3 and indices.shape[1] == 1:
        return indices[:, 0, :]
    raise ValueError(f"indices must be [T, K] or [T, 1, K], got {indices.shape}")


def sparse_mla_cuda(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    attn_sink: torch.Tensor | None,
    topk_length: torch.Tensor,
) -> torch.Tensor:
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"q must be float16 or bfloat16, got {q.dtype}")
    if q.dim() != 3:
        raise ValueError(f"q must be [tokens, heads, dim], got {q.shape}")
    if q.shape[-1] != 512:
        raise ValueError(f"q head_dim must be 512, got {q.shape[-1]}")
    kv_flat = _flatten_sparse_mla_kv(kv)
    if kv_flat.dtype != q.dtype:
        raise TypeError(f"kv dtype {kv_flat.dtype} must match q dtype {q.dtype}")
    if kv_flat.shape[-1] != q.shape[-1]:
        raise ValueError(f"kv dim {kv_flat.shape[-1]} != q dim {q.shape[-1]}")
    indices_2d = _flatten_sparse_mla_indices(indices)
    if indices_2d.shape[0] != q.shape[0]:
        raise ValueError(f"indices token count {indices_2d.shape[0]} != {q.shape[0]}")
    if topk_length.dim() != 1 or topk_length.shape[0] != q.shape[0]:
        raise ValueError(
            f"topk_length must be [tokens], got {topk_length.shape} for q={q.shape}"
        )
    if attn_sink is None:
        attn_sink = torch.empty(0, device=q.device, dtype=torch.float32)
    else:
        attn_sink = attn_sink.to(device=q.device, dtype=torch.float32)
    out = torch.empty_like(q)
    _load_deepseek_v4_attention_module().deepseek_v4_sparse_mla_cuda(
        q.contiguous(),
        kv_flat.contiguous(),
        indices_2d.to(device=q.device, dtype=torch.int32).contiguous(),
        topk_length.to(device=q.device, dtype=torch.int64).contiguous(),
        attn_sink.contiguous(),
        out,
        float(sm_scale),
    )
    return out


def _flatten_cache_sparse_mla_indices(indices: torch.Tensor, name: str) -> torch.Tensor:
    if indices.dim() == 2:
        return indices
    if indices.dim() == 3 and indices.shape[1] == 1:
        return indices[:, 0, :]
    raise ValueError(f"{name} must be [T, K] or [T, 1, K], got {indices.shape}")


def _flatten_fp8_cache(cache: torch.Tensor, block_size: int, name: str) -> torch.Tensor:
    if cache.dim() == 2:
        return cache
    if cache.dim() == 4 and cache.shape[1] == block_size and cache.shape[2] == 1:
        return cache.reshape(cache.shape[0], cache.shape[1] * cache.shape[3])
    raise ValueError(
        f"{name} must be [blocks, bytes] or [blocks, block, 1, row_bytes], "
        f"got {cache.shape}"
    )


def sparse_mla_fp8_cache_cuda(
    q: torch.Tensor,
    swa_cache: torch.Tensor,
    swa_indices: torch.Tensor,
    swa_lens: torch.Tensor,
    swa_block_size: int,
    compressed_cache: torch.Tensor | None,
    extra_indices: torch.Tensor | None,
    extra_lens: torch.Tensor | None,
    compressed_block_size: int,
    sm_scale: float,
    attn_sink: torch.Tensor | None,
    online_softmax: bool = False,
) -> torch.Tensor:
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"q must be float16 or bfloat16, got {q.dtype}")
    if q.dim() != 3 or q.shape[-1] != 512:
        raise ValueError(f"q must be [tokens, heads, 512], got {q.shape}")
    if swa_cache.dtype != torch.uint8:
        raise TypeError(f"swa_cache must be uint8, got {swa_cache.dtype}")
    swa_cache = _flatten_fp8_cache(swa_cache, swa_block_size, "swa_cache")
    swa_indices_2d = _flatten_cache_sparse_mla_indices(swa_indices, "swa_indices")
    if swa_indices_2d.shape[0] != q.shape[0]:
        raise ValueError(
            f"swa_indices token count {swa_indices_2d.shape[0]} != {q.shape[0]}"
        )
    if swa_lens.dim() != 1 or swa_lens.shape[0] != q.shape[0]:
        raise ValueError(f"swa_lens must be [tokens], got {swa_lens.shape}")

    if compressed_cache is None:
        compressed_cache = torch.empty((0, 0), device=q.device, dtype=torch.uint8)
    if compressed_cache.dtype != torch.uint8:
        raise TypeError(f"compressed_cache must be uint8, got {compressed_cache.dtype}")
    compressed_cache = _flatten_fp8_cache(
        compressed_cache,
        compressed_block_size,
        "compressed_cache",
    )

    if extra_indices is None:
        extra_indices_2d = torch.empty(
            (q.shape[0], 0), device=q.device, dtype=torch.int32
        )
    else:
        extra_indices_2d = _flatten_cache_sparse_mla_indices(
            extra_indices,
            "extra_indices",
        )
    if extra_indices_2d.shape[0] != q.shape[0]:
        raise ValueError(
            f"extra_indices token count {extra_indices_2d.shape[0]} != {q.shape[0]}"
        )
    if extra_lens is None:
        extra_lens = torch.zeros(q.shape[0], device=q.device, dtype=torch.int64)
    if extra_lens.dim() != 1 or extra_lens.shape[0] != q.shape[0]:
        raise ValueError(f"extra_lens must be [tokens], got {extra_lens.shape}")

    if attn_sink is None:
        attn_sink = torch.empty(0, device=q.device, dtype=torch.float32)
    else:
        attn_sink = attn_sink.to(device=q.device, dtype=torch.float32)

    q_arg = q.contiguous()
    swa_cache_arg = swa_cache.contiguous()
    swa_indices_arg = swa_indices_2d.to(device=q.device, dtype=torch.int32).contiguous()
    swa_lens_arg = swa_lens.to(device=q.device, dtype=torch.int64).contiguous()
    compressed_cache_arg = compressed_cache.contiguous()
    extra_indices_arg = extra_indices_2d.to(
        device=q.device,
        dtype=torch.int32,
    ).contiguous()
    extra_lens_arg = extra_lens.to(device=q.device, dtype=torch.int64).contiguous()
    attn_sink_arg = attn_sink.contiguous()

    out = torch.empty_like(q_arg)
    _load_deepseek_v4_attention_module().deepseek_v4_sparse_mla_fp8_cache_cuda(
        q_arg,
        swa_cache_arg,
        swa_indices_arg,
        swa_lens_arg,
        compressed_cache_arg,
        extra_indices_arg,
        extra_lens_arg,
        attn_sink_arg,
        out,
        float(sm_scale),
        int(bool(online_softmax)),
        int(swa_block_size),
        int(compressed_block_size),
    )
    return out


def csa_indexer_cache_insert_fp8_cuda(
    *,
    state_cache: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    positions: torch.Tensor,
    compressor_slot_mapping: torch.Tensor,
    block_table: torch.Tensor,
    compressor_block_size: int,
    rms_norm_weight: torch.Tensor,
    rms_norm_eps: float,
    cos_sin_cache: torch.Tensor,
    kv_cache_2d: torch.Tensor,
    kv_slot_mapping: torch.Tensor,
    kv_cache_block_size: int,
) -> None:
    if state_cache.dtype != torch.float32:
        raise TypeError(f"state_cache must be float32, got {state_cache.dtype}")
    if state_cache.dim() != 3 or state_cache.shape[-1] != 512:
        raise ValueError(
            f"state_cache must be [blocks, block, 512], got {state_cache.shape}"
        )
    if token_to_req_indices.dtype != torch.int32:
        token_to_req_indices = token_to_req_indices.to(torch.int32)
    if positions.dtype != torch.int64:
        positions = positions.to(torch.int64)
    if compressor_slot_mapping.dtype != torch.int64:
        compressor_slot_mapping = compressor_slot_mapping.to(torch.int64)
    if block_table.dtype != torch.int32:
        block_table = block_table.to(torch.int32)
    if rms_norm_weight.dtype != torch.float32:
        rms_norm_weight = rms_norm_weight.to(torch.float32)
    if cos_sin_cache.dtype != torch.float32:
        cos_sin_cache = cos_sin_cache.to(torch.float32)
    if kv_cache_2d.dtype != torch.uint8:
        raise TypeError(f"kv_cache_2d must be uint8, got {kv_cache_2d.dtype}")
    if kv_slot_mapping.dtype != torch.int64:
        kv_slot_mapping = kv_slot_mapping.to(torch.int64)
    if rms_norm_weight.dim() != 1 or rms_norm_weight.shape[0] != 128:
        raise ValueError(f"rms_norm_weight must be [128], got {rms_norm_weight.shape}")
    if cos_sin_cache.dim() != 2 or cos_sin_cache.shape[1] < 64:
        raise ValueError(
            f"cos_sin_cache must be [positions, >=64], got {cos_sin_cache.shape}"
        )
    if kv_cache_2d.dim() != 2:
        raise ValueError(
            f"kv_cache_2d must be [blocks, bytes], got {kv_cache_2d.shape}"
        )

    _load_deepseek_v4_attention_module().deepseek_v4_csa_indexer_cache_insert_fp8_cuda(
        state_cache.contiguous(),
        token_to_req_indices.contiguous(),
        positions.contiguous(),
        compressor_slot_mapping.contiguous(),
        block_table.contiguous(),
        rms_norm_weight.contiguous(),
        cos_sin_cache.contiguous(),
        kv_cache_2d,
        kv_slot_mapping.contiguous(),
        float(rms_norm_eps),
        int(compressor_block_size),
        int(kv_cache_block_size),
    )


def compressed_kv_cache_insert_cuda(
    *,
    state_cache: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    positions: torch.Tensor,
    compressor_slot_mapping: torch.Tensor,
    block_table: torch.Tensor,
    rms_norm_weight: torch.Tensor,
    rms_norm_eps: float,
    cos_sin_cache: torch.Tensor,
    kv_cache_2d: torch.Tensor,
    kv_slot_mapping: torch.Tensor,
    kv_cache_block_size: int,
    compress_ratio: int,
) -> None:
    if state_cache.dtype != torch.float32:
        raise TypeError(f"state_cache must be float32, got {state_cache.dtype}")
    if state_cache.dim() != 3:
        raise ValueError(
            f"state_cache must be [blocks, block, width], got {state_cache.shape}"
        )
    if compress_ratio not in (4, 128):
        raise ValueError(f"compress_ratio must be 4 or 128, got {compress_ratio}")
    expected_state_width = 1024 if compress_ratio == 4 else 512
    if state_cache.shape[-1] != expected_state_width * 2:
        raise ValueError(
            "state_cache last dim must be "
            f"{expected_state_width * 2}, got {state_cache.shape[-1]}"
        )
    if token_to_req_indices.dtype != torch.int32:
        token_to_req_indices = token_to_req_indices.to(torch.int32)
    if positions.dtype != torch.int64:
        positions = positions.to(torch.int64)
    if compressor_slot_mapping.dtype != torch.int64:
        compressor_slot_mapping = compressor_slot_mapping.to(torch.int64)
    if block_table.dtype != torch.int32:
        block_table = block_table.to(torch.int32)
    if rms_norm_weight.dtype != torch.float32:
        rms_norm_weight = rms_norm_weight.to(torch.float32)
    if cos_sin_cache.dtype != torch.float32:
        cos_sin_cache = cos_sin_cache.to(torch.float32)
    if kv_cache_2d.dtype != torch.uint8:
        raise TypeError(f"kv_cache_2d must be uint8, got {kv_cache_2d.dtype}")
    if kv_slot_mapping.dtype != torch.int64:
        kv_slot_mapping = kv_slot_mapping.to(torch.int64)
    if rms_norm_weight.dim() != 1 or rms_norm_weight.shape[0] != 512:
        raise ValueError(f"rms_norm_weight must be [512], got {rms_norm_weight.shape}")
    if cos_sin_cache.dim() != 2 or cos_sin_cache.shape[1] < 64:
        raise ValueError(
            f"cos_sin_cache must be [positions, >=64], got {cos_sin_cache.shape}"
        )
    if kv_cache_2d.dim() != 2:
        raise ValueError(
            f"kv_cache_2d must be [blocks, bytes], got {kv_cache_2d.shape}"
        )

    _load_deepseek_v4_attention_module().deepseek_v4_compressed_kv_cache_insert_cuda(
        state_cache.contiguous(),
        token_to_req_indices.contiguous(),
        positions.contiguous(),
        compressor_slot_mapping.contiguous(),
        block_table.contiguous(),
        rms_norm_weight.contiguous(),
        cos_sin_cache.contiguous(),
        kv_cache_2d,
        kv_slot_mapping.contiguous(),
        float(rms_norm_eps),
        int(state_cache.shape[1]),
        int(kv_cache_block_size),
        int(compress_ratio),
    )


def decode_indices_cuda(
    *,
    positions: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    block_table: torch.Tensor,
    topk_indices: torch.Tensor | None,
    window_size: int,
    swa_block_size: int,
    compress_ratio: int,
    compressed_block_size: int,
    full_candidate_max_seq_len: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    if positions.dtype != torch.int64:
        positions = positions.to(torch.int64)
    if token_to_req_indices.dtype != torch.int32:
        token_to_req_indices = token_to_req_indices.to(torch.int32)
    if block_table.dtype != torch.int32:
        block_table = block_table.to(torch.int32)
    if positions.dim() != 1:
        raise ValueError(f"positions must be [tokens], got {positions.shape}")
    if token_to_req_indices.dim() != 1:
        raise ValueError(
            f"token_to_req_indices must be [tokens], got {token_to_req_indices.shape}"
        )
    if block_table.dim() != 2:
        raise ValueError(
            f"block_table must be [requests, pages], got {block_table.shape}"
        )
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")
    if swa_block_size <= 0:
        raise ValueError(f"swa_block_size must be positive, got {swa_block_size}")
    if compress_ratio <= 0:
        raise ValueError(f"compress_ratio must be positive, got {compress_ratio}")
    if compressed_block_size <= 0:
        raise ValueError(
            f"compressed_block_size must be positive, got {compressed_block_size}"
        )

    num_tokens = positions.shape[0]
    device = positions.device
    swa_indices = torch.empty(
        (num_tokens, max(int(window_size), 1)),
        device=device,
        dtype=torch.int32,
    )
    swa_lens = torch.empty(num_tokens, device=device, dtype=torch.int32)

    full_candidate_max_len = -1
    if full_candidate_max_seq_len is not None:
        if full_candidate_max_seq_len < 0:
            raise ValueError(
                "full_candidate_max_seq_len must be non-negative, "
                f"got {full_candidate_max_seq_len}"
            )
        full_candidate_max_len = int(full_candidate_max_seq_len) // int(compress_ratio)

    if compress_ratio == 4:
        if topk_indices is None:
            if full_candidate_max_len < 0:
                raise ValueError("CSA decode requires topk_indices")
            topk_indices = torch.empty(
                (num_tokens, 0), device=device, dtype=torch.int32
            )
        if topk_indices.dtype != torch.int32:
            topk_indices = topk_indices.to(torch.int32)
        if topk_indices.dim() != 2 or topk_indices.shape[0] != num_tokens:
            raise ValueError(
                f"topk_indices must be [tokens, topk], got {topk_indices.shape}"
            )
        topk_arg = topk_indices.contiguous()
        extra_source_width = (
            full_candidate_max_len
            if full_candidate_max_len >= 0 and topk_indices.shape[1] == 0
            else topk_indices.shape[1]
        )
        extra_width = max(64, ((extra_source_width + 63) // 64) * 64)
        extra_indices = torch.empty(
            (num_tokens, extra_width),
            device=device,
            dtype=torch.int32,
        )
        extra_lens = torch.empty(num_tokens, device=device, dtype=torch.int32)
    elif compress_ratio <= 1:
        topk_arg = torch.empty((num_tokens, 0), device=device, dtype=torch.int32)
        extra_indices = torch.empty((num_tokens, 0), device=device, dtype=torch.int32)
        extra_lens = torch.empty(num_tokens, device=device, dtype=torch.int32)
    else:
        if topk_indices is not None:
            raise ValueError(
                "non-CSA compressed decode expects dense candidate generation; "
                f"got topk_indices={topk_indices.shape}"
            )
        if full_candidate_max_len < 0:
            raise ValueError(
                "non-CSA compressed decode requires full_candidate_max_seq_len"
            )
        topk_arg = torch.empty((num_tokens, 0), device=device, dtype=torch.int32)
        extra_width = max(64, ((full_candidate_max_len + 63) // 64) * 64)
        extra_indices = torch.empty(
            (num_tokens, extra_width),
            device=device,
            dtype=torch.int32,
        )
        extra_lens = torch.empty(num_tokens, device=device, dtype=torch.int32)

    _load_deepseek_v4_attention_module().deepseek_v4_decode_indices_cuda(
        positions.contiguous(),
        token_to_req_indices.contiguous(),
        block_table.contiguous(),
        topk_arg,
        swa_indices,
        swa_lens,
        extra_indices,
        extra_lens,
        int(window_size),
        int(swa_block_size),
        int(compress_ratio),
        int(compressed_block_size),
        int(full_candidate_max_len),
    )
    if compress_ratio <= 1:
        return swa_indices, swa_lens, None, None
    return swa_indices, swa_lens, extra_indices.unsqueeze(1), extra_lens


def full_candidate_topk_cuda(
    *,
    positions: torch.Tensor,
    compress_ratio: int,
    max_seq_len: int,
    topk_tokens: int,
) -> torch.Tensor | None:
    if positions.dtype != torch.int64:
        positions = positions.to(torch.int64)
    if positions.dim() != 1:
        raise ValueError(f"positions must be [tokens], got {positions.shape}")
    if compress_ratio <= 0:
        raise ValueError(f"compress_ratio must be positive, got {compress_ratio}")
    if max_seq_len < 0:
        raise ValueError(f"max_seq_len must be non-negative, got {max_seq_len}")
    if topk_tokens < 0:
        raise ValueError(f"topk_tokens must be non-negative, got {topk_tokens}")

    max_len = int(max_seq_len) // int(compress_ratio)
    if max_len > int(topk_tokens):
        return None
    topk = torch.empty(
        (positions.shape[0], max_len),
        device=positions.device,
        dtype=torch.int32,
    )
    if positions.shape[0] == 0 or max_len == 0:
        return topk

    _load_deepseek_v4_attention_module().deepseek_v4_full_candidate_topk_cuda(
        positions.contiguous(),
        topk,
        int(compress_ratio),
    )
    return topk


def save_compressor_state_cuda(
    *,
    kv: torch.Tensor,
    score: torch.Tensor,
    ape: torch.Tensor,
    state_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    positions: torch.Tensor,
    block_size: int,
    compress_ratio: int,
) -> None:
    if kv.dtype != torch.float32:
        raise TypeError(f"kv must be float32, got {kv.dtype}")
    if score.dtype != torch.float32:
        raise TypeError(f"score must be float32, got {score.dtype}")
    if ape.dtype != torch.float32:
        raise TypeError(f"ape must be float32, got {ape.dtype}")
    if state_cache.dtype != torch.float32:
        raise TypeError(f"state_cache must be float32, got {state_cache.dtype}")
    if kv.dim() != 2:
        raise ValueError(f"kv must be [tokens, width], got {kv.shape}")
    if score.shape != kv.shape:
        raise ValueError(f"score shape {score.shape} must match kv {kv.shape}")
    if ape.dim() != 2 or ape.shape != (compress_ratio, kv.shape[-1]):
        raise ValueError(
            "ape must be [compress_ratio, width], "
            f"got {ape.shape} for compress_ratio={compress_ratio} width={kv.shape[-1]}"
        )
    if state_cache.dim() != 3 or state_cache.shape[-1] != kv.shape[-1] * 2:
        raise ValueError(
            "state_cache must be [blocks, block_size, 2 * width], "
            f"got {state_cache.shape} for width={kv.shape[-1]}"
        )
    if slot_mapping.dtype != torch.int64:
        slot_mapping = slot_mapping.to(torch.int64)
    if positions.dtype != torch.int64:
        positions = positions.to(torch.int64)
    if slot_mapping.dim() != 1 or slot_mapping.shape[0] < kv.shape[0]:
        raise ValueError(f"slot_mapping must cover kv rows, got {slot_mapping.shape}")
    if positions.dim() != 1 or positions.shape[0] < kv.shape[0]:
        raise ValueError(f"positions must cover kv rows, got {positions.shape}")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if compress_ratio <= 0:
        raise ValueError(f"compress_ratio must be positive, got {compress_ratio}")

    _load_deepseek_v4_attention_module().deepseek_v4_save_compressor_state_cuda(
        kv.contiguous(),
        score.contiguous(),
        ape.contiguous(),
        state_cache,
        slot_mapping.contiguous(),
        positions.contiguous(),
        int(block_size),
        int(compress_ratio),
    )


def inv_rope_grouped_cuda(
    *,
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int,
    rope_dim: int,
) -> torch.Tensor:
    if o.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"o must be float16 or bfloat16, got {o.dtype}")
    if o.dim() != 3:
        raise ValueError(f"o must be [tokens, heads, dim], got {o.shape}")
    if positions.dtype != torch.int64:
        positions = positions.to(torch.int64)
    if positions.dim() != 1 or positions.shape[0] != o.shape[0]:
        raise ValueError(f"positions must be [tokens], got {positions.shape}")
    if cos_sin_cache.dtype != torch.float32:
        cos_sin_cache = cos_sin_cache.to(torch.float32)
    if cos_sin_cache.dim() != 2 or cos_sin_cache.shape[1] < rope_dim:
        raise ValueError(
            f"cos_sin_cache must be [positions, >= rope_dim], got {cos_sin_cache.shape}"
        )
    if n_groups <= 0:
        raise ValueError(f"n_groups must be positive, got {n_groups}")
    if heads_per_group <= 0:
        raise ValueError(f"heads_per_group must be positive, got {heads_per_group}")
    if o.shape[1] != n_groups * heads_per_group:
        raise ValueError(
            f"heads={o.shape[1]} does not match {n_groups} * {heads_per_group}"
        )
    if o.shape[2] != nope_dim + rope_dim:
        raise ValueError(f"head dim {o.shape[2]} != {nope_dim} + {rope_dim}")
    if rope_dim <= 0 or rope_dim % 2:
        raise ValueError(f"rope_dim must be positive and even, got {rope_dim}")

    o_arg = o.contiguous()
    out = torch.empty(
        (o.shape[0], int(n_groups), int(heads_per_group) * o.shape[2]),
        device=o.device,
        dtype=o.dtype,
    )
    _load_deepseek_v4_attention_module().deepseek_v4_inv_rope_grouped_cuda(
        o_arg,
        positions.contiguous(),
        cos_sin_cache.contiguous(),
        out,
        int(n_groups),
        int(heads_per_group),
        int(nope_dim),
        int(rope_dim),
    )
    return out
