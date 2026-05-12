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

from tokenspeed_kernel.ops.attention.deepseek_v4.cuda import (
    csa_indexer_cache_insert_fp8_cuda,
    deepseek_v4_compressed_kv_cache_insert_cuda,
    deepseek_v4_decode_indices_cuda,
    deepseek_v4_full_candidate_topk_cuda,
    deepseek_v4_inv_rope_grouped_cuda,
    deepseek_v4_save_compressor_state_cuda,
    deepseek_v4_sparse_mla_cuda,
    deepseek_v4_sparse_mla_fp8_cache_cuda,
)
from tokenspeed_kernel.ops.attention.deepseek_v4.reference import (
    deepseek_v4_sparse_mla_reference,
)

try:
    from tokenspeed_kernel.ops.attention.deepseek_v4.triton_projection import (
        deepseek_v4_fp8_einsum_sm12x_triton,
        deepseek_v4_fused_inv_rope_fp8_quant_triton,
    )
except ModuleNotFoundError as exc:
    if exc.name not in {"tokenspeed_triton", "triton"}:
        raise
    deepseek_v4_fp8_einsum_sm12x_triton = None
    deepseek_v4_fused_inv_rope_fp8_quant_triton = None

__all__ = [
    "csa_indexer_cache_insert_fp8_cuda",
    "deepseek_v4_compressed_kv_cache_insert_cuda",
    "deepseek_v4_decode_indices_cuda",
    "deepseek_v4_fp8_einsum_sm12x_triton",
    "deepseek_v4_full_candidate_topk_cuda",
    "deepseek_v4_fused_inv_rope_fp8_quant_triton",
    "deepseek_v4_inv_rope_grouped_cuda",
    "deepseek_v4_save_compressor_state_cuda",
    "deepseek_v4_sparse_mla_cuda",
    "deepseek_v4_sparse_mla_fp8_cache_cuda",
    "deepseek_v4_sparse_mla_reference",
]
