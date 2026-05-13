// Copyright (c) 2026 LightSeek Foundation
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "tvm_ffi_utils.h"

using tvm::ffi::TensorView;

void fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
    TensorView q,
    TensorView kv,
    TensorView k_cache,
    TensorView slot_mapping,
    TensorView positions,
    TensorView cos_sin_cache,
    double rms_norm_eps,
    int64_t cache_block_size);

void deepseek_v4_indexer_topk_prefill(TensorView logits,
                                      TensorView row_starts,
                                      TensorView row_ends,
                                      TensorView output,
                                      int64_t k);

void deepseek_v4_gather_paged_indexer_mxfp4_cache(TensorView kv_cache,
                                                  TensorView values_out,
                                                  TensorView scales_out,
                                                  TensorView block_table,
                                                  TensorView cu_seq_lens,
                                                  int64_t cache_block_size);

void deepseek_v4_sparse_mla_cuda(
    TensorView q,
    TensorView kv,
    TensorView indices,
    TensorView topk_length,
    TensorView attn_sink,
    TensorView output,
    double sm_scale);

void deepseek_v4_sparse_mla_fp8_cache_cuda(
    TensorView q,
    TensorView swa_cache,
    TensorView swa_indices,
    TensorView swa_lens,
    TensorView compressed_cache,
    TensorView extra_indices,
    TensorView extra_lens,
    TensorView attn_sink,
    TensorView output,
    TensorView partials,
    double sm_scale,
    int64_t online_softmax,
    int64_t swa_block_size,
    int64_t compressed_block_size,
    int64_t k_split);

void deepseek_v4_csa_indexer_cache_insert_fp8_cuda(
    TensorView state_cache,
    TensorView token_to_req_indices,
    TensorView positions,
    TensorView compressor_slot_mapping,
    TensorView block_table,
    TensorView rms_norm_weight,
    TensorView cos_sin_cache,
    TensorView kv_cache,
    TensorView kv_slot_mapping,
    double rms_norm_eps,
    int64_t compressor_block_size,
    int64_t kv_cache_block_size);

void deepseek_v4_compressed_kv_cache_insert_cuda(
    TensorView state_cache,
    TensorView token_to_req_indices,
    TensorView positions,
    TensorView compressor_slot_mapping,
    TensorView block_table,
    TensorView rms_norm_weight,
    TensorView cos_sin_cache,
    TensorView kv_cache,
    TensorView kv_slot_mapping,
    double rms_norm_eps,
    int64_t compressor_block_size,
    int64_t kv_cache_block_size,
    int64_t compress_ratio);

void deepseek_v4_decode_indices_cuda(
    TensorView positions,
    TensorView token_to_req_indices,
    TensorView block_table,
    TensorView topk_indices,
    TensorView swa_indices,
    TensorView swa_lens,
    TensorView extra_indices,
    TensorView extra_lens,
    int64_t window_size,
    int64_t swa_block_size,
    int64_t compress_ratio,
    int64_t compressed_block_size,
    int64_t full_candidate_max_len);

void deepseek_v4_full_candidate_topk_cuda(
    TensorView positions,
    TensorView topk,
    int64_t compress_ratio);

void deepseek_v4_save_compressor_state_cuda(
    TensorView kv,
    TensorView score,
    TensorView ape,
    TensorView state_cache,
    TensorView slot_mapping,
    TensorView positions,
    int64_t block_size,
    int64_t compress_ratio);

void deepseek_v4_sm12x_mhc_pre_cuda(
    TensorView residual,
    TensorView fn,
    TensorView hc_scale,
    TensorView hc_base,
    TensorView layer_input,
    TensorView post,
    TensorView comb,
    double rms_eps,
    double hc_eps,
    int64_t sinkhorn_iters);

void deepseek_v4_sm12x_mhc_pre_split_cuda(
    TensorView residual,
    TensorView fn,
    TensorView hc_scale,
    TensorView hc_base,
    TensorView layer_input,
    TensorView post,
    TensorView comb,
    TensorView partials,
    double rms_eps,
    double hc_eps,
    int64_t sinkhorn_iters);

void deepseek_v4_sm12x_mhc_post_cuda(
    TensorView hidden_states,
    TensorView residual,
    TensorView post,
    TensorView comb,
    TensorView output);

TVM_FFI_DLL_EXPORT_TYPED_FUNC(fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert,
                              fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(deepseek_v4_sparse_mla_cuda,
                              deepseek_v4_sparse_mla_cuda);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(deepseek_v4_sparse_mla_fp8_cache_cuda,
                              deepseek_v4_sparse_mla_fp8_cache_cuda);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(deepseek_v4_csa_indexer_cache_insert_fp8_cuda,
                              deepseek_v4_csa_indexer_cache_insert_fp8_cuda);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(deepseek_v4_compressed_kv_cache_insert_cuda,
                              deepseek_v4_compressed_kv_cache_insert_cuda);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(deepseek_v4_decode_indices_cuda,
                              deepseek_v4_decode_indices_cuda);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(deepseek_v4_full_candidate_topk_cuda,
                              deepseek_v4_full_candidate_topk_cuda);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(deepseek_v4_save_compressor_state_cuda,
                              deepseek_v4_save_compressor_state_cuda);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(deepseek_v4_sm12x_mhc_pre_cuda,
                              deepseek_v4_sm12x_mhc_pre_cuda);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(deepseek_v4_sm12x_mhc_pre_split_cuda,
                              deepseek_v4_sm12x_mhc_pre_split_cuda);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(deepseek_v4_sm12x_mhc_post_cuda,
                              deepseek_v4_sm12x_mhc_post_cuda);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(deepseek_v4_indexer_topk_prefill,
                              deepseek_v4_indexer_topk_prefill);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(deepseek_v4_gather_paged_indexer_mxfp4_cache,
                              deepseek_v4_gather_paged_indexer_mxfp4_cache);
