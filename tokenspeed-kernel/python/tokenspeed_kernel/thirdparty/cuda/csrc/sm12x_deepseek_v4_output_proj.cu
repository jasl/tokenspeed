// Portions Copyright (c) 2025 DeepSeek (MIT). Adapted from DeepGEMM
// `deep_gemm/include/deep_gemm/impls/sm120_fp8_einsum.cuh`
// (https://github.com/deepseek-ai/DeepGEMM, PR #324, branch
// codex/sm120-paged-mqa-tiled): the SM120 `bhr,hdr->bhd` FP8 einsum
// kernel and its fp8x4-vectorized dot-product helpers are reused
// verbatim and rewrapped behind tokenspeed's tvm-ffi binding.
//
// Tokenspeed adaptations (Copyright (c) 2026 LightSeek Foundation) are
// MIT-licensed; see the project LICENSE for the full text.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

// SM12x DeepSeek V4 attention output projection: per-group batched FP8 GEMV
// computing the einsum `bhr,hdr->bhd` where
//   a       : [num_tokens, num_groups, hidden]            FP8 e4m3
//   a_scale : [num_tokens, num_groups, hidden/128]        float32 (per-block)
//   b       : [num_groups, out_rank, hidden]              FP8 e4m3, contiguous
//   b_scale : [num_groups, out_rank/128, hidden/128]      UE8M0 (uint8) or float32
//   output  : [num_tokens, num_groups, out_rank]          bfloat16
//
// Schedule: one CUDA block per (token, group, d_tile) where d_tile spans
// kOutputTileD=128 consecutive out_rank columns; the block has
// kOutputTileD threads and each thread independently accumulates the full
// hidden-dimension reduction for its (token, group, d) cell. Loads of `a`
// and `b` are fp8x4-vectorized; the block-128 scales (per-token row, per
// out_rank/128 weight tile) are folded into the FMA accumulation per
// scale group.
//
// Two earlier designs were rejected and one cooperative-tile MMA design
// only reached ~0.5x of the Triton baseline (see
// docs/notes/2026-05-09-ds4-sm12x-rejected-experiments.md). The
// per-thread GEMV-style design ported here side-steps the tensor-core
// path entirely -- at decode shape (small `tokens` batch) the M axis is
// too small for MMA tiles to pay for themselves; a thread-per-output-cell
// GEMV with fp8x4 loads and L1/L2-friendly weight access wins.

#include <cmath>
#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include "tvm_ffi_utils.h"

using tvm::ffi::TensorView;

namespace {

constexpr int kBlockK = 128;             // block-128 scale group
constexpr int kOutputTileD = 128;        // out_rank cols computed per block

__device__ __forceinline__ float decode_ue8m0_scale(uint8_t encoded) {
  uint32_t bits = static_cast<uint32_t>(encoded) << 23;
  return __uint_as_float(bits);
}

__device__ __forceinline__ float load_b_scale_value(
    const void* __restrict__ scales,
    int64_t offset,
    bool scales_are_ue8m0) {
  if (scales_are_ue8m0) {
    return decode_ue8m0_scale(static_cast<const uint8_t*>(scales)[offset]);
  }
  return static_cast<const float*>(scales)[offset];
}

// Per-thread dot kernel along the hidden reduction axis. Vectorized
// fp8x4 loads (32 bits per LDG) cut the LDG count by 4x versus the
// scalar variant; tail handling supports hidden_dim that is not a
// multiple of 128.
//
// Layout assumes `a_stride_r == 1` and `b_stride_r == 1` -- i.e. the
// hidden dim is contiguous in both operands. The host-side validation
// enforces this.
__device__ __forceinline__ float sm120_fp8_einsum_dot_fp8x4(
    const __nv_fp8_e4m3* a,
    const __nv_fp8_e4m3* b,
    const float* sfa,
    int64_t sfa_stride_r,
    const void* sfb_packed,
    int64_t sfb_stride_r,
    bool sfb_is_ue8m0,
    int hidden_dim) {
  float accum = 0.0f;
  int const num_full_scale_blocks = hidden_dim / kBlockK;
  for (int scale_idx = 0; scale_idx < num_full_scale_blocks; ++scale_idx) {
    float const sa = sfa[scale_idx * sfa_stride_r];
    float const sb =
        load_b_scale_value(sfb_packed, scale_idx * sfb_stride_r, sfb_is_ue8m0);
    float const scale = sa * sb;
    int const r_start = scale_idx * kBlockK;
#pragma unroll
    for (int r_offset = 0; r_offset < kBlockK; r_offset += 4) {
      int const r_idx = r_start + r_offset;
      auto const a_values = static_cast<float4>(
          *reinterpret_cast<const __nv_fp8x4_e4m3*>(a + r_idx));
      auto const b_values = static_cast<float4>(
          *reinterpret_cast<const __nv_fp8x4_e4m3*>(b + r_idx));
      accum = fmaf(a_values.x, b_values.x * scale, accum);
      accum = fmaf(a_values.y, b_values.y * scale, accum);
      accum = fmaf(a_values.z, b_values.z * scale, accum);
      accum = fmaf(a_values.w, b_values.w * scale, accum);
    }
  }
  int const tail_start = num_full_scale_blocks * kBlockK;
  if (tail_start >= hidden_dim) {
    return accum;
  }
  float const tail_sa = sfa[num_full_scale_blocks * sfa_stride_r];
  float const tail_sb = load_b_scale_value(
      sfb_packed, num_full_scale_blocks * sfb_stride_r, sfb_is_ue8m0);
  float const tail_scale = tail_sa * tail_sb;
  for (int r_idx = tail_start; r_idx < hidden_dim; r_idx += 4) {
    auto const a_values = static_cast<float4>(
        *reinterpret_cast<const __nv_fp8x4_e4m3*>(a + r_idx));
    auto const b_values = static_cast<float4>(
        *reinterpret_cast<const __nv_fp8x4_e4m3*>(b + r_idx));
    accum = fmaf(a_values.x, b_values.x * tail_scale, accum);
    accum = fmaf(a_values.y, b_values.y * tail_scale, accum);
    accum = fmaf(a_values.z, b_values.z * tail_scale, accum);
    accum = fmaf(a_values.w, b_values.w * tail_scale, accum);
  }
  return accum;
}

__global__ void deepseek_v4_grouped_fp8_einsum_kernel(
    nv_bfloat16* __restrict__ output,
    const __nv_fp8_e4m3* __restrict__ a,
    const float* __restrict__ a_scales,
    const __nv_fp8_e4m3* __restrict__ b,
    const void* __restrict__ b_scales,
    int num_tokens,
    int num_groups,
    int hidden_dim,
    int out_dim,
    int num_d_tiles,
    int64_t a_stride_token,
    int64_t a_stride_group,
    int64_t a_stride_hidden,
    int64_t a_scale_stride_token,
    int64_t a_scale_stride_group,
    int64_t a_scale_stride_block,
    int64_t b_stride_group,
    int64_t b_stride_out,
    int64_t b_stride_hidden,
    int64_t b_scale_stride_group,
    int64_t b_scale_stride_out,
    int64_t b_scale_stride_hidden,
    int64_t out_stride_token,
    int64_t out_stride_group,
    int64_t out_stride_rank,
    bool b_scales_are_ue8m0) {
  int const tile_idx = blockIdx.x;
  int const d_tile_idx = tile_idx % num_d_tiles;
  int const h_idx = (tile_idx / num_d_tiles) % num_groups;
  int const b_idx = tile_idx / (num_d_tiles * num_groups);
  int const d_idx = d_tile_idx * kOutputTileD + threadIdx.x;
  if (b_idx >= num_tokens || d_idx >= out_dim) {
    return;
  }

  const __nv_fp8_e4m3* a_base = a + static_cast<int64_t>(b_idx) * a_stride_token
                                + static_cast<int64_t>(h_idx) * a_stride_group;
  const __nv_fp8_e4m3* b_base = b + static_cast<int64_t>(h_idx) * b_stride_group
                                + static_cast<int64_t>(d_idx) * b_stride_out;
  const float* sfa_base = a_scales
                          + static_cast<int64_t>(b_idx) * a_scale_stride_token
                          + static_cast<int64_t>(h_idx) * a_scale_stride_group;
  int64_t const sfb_out_tile = d_idx / kBlockK;
  int64_t const sfb_byte_base = static_cast<int64_t>(h_idx) * b_scale_stride_group
                                + sfb_out_tile * b_scale_stride_out;
  void const* sfb_packed = b_scales_are_ue8m0
      ? static_cast<const void*>(
            static_cast<const uint8_t*>(b_scales) + sfb_byte_base)
      : static_cast<const void*>(
            static_cast<const float*>(b_scales) + sfb_byte_base);

  float const accum = sm120_fp8_einsum_dot_fp8x4(
      a_base,
      b_base,
      sfa_base,
      a_scale_stride_block,
      sfb_packed,
      b_scale_stride_hidden,
      b_scales_are_ue8m0,
      hidden_dim);

  int64_t const out_idx = static_cast<int64_t>(b_idx) * out_stride_token
                          + static_cast<int64_t>(h_idx) * out_stride_group
                          + static_cast<int64_t>(d_idx) * out_stride_rank;
  output[out_idx] = __float2bfloat16(accum);
}

void launch_deepseek_v4_grouped_fp8_einsum(
    TensorView output,
    TensorView a,
    TensorView a_scales,
    TensorView b,
    TensorView b_scales,
    cudaStream_t stream,
    bool b_scales_are_ue8m0) {
  int const num_tokens = static_cast<int>(a.size(0));
  int const num_groups = static_cast<int>(a.size(1));
  int const hidden_dim = static_cast<int>(a.size(2));
  int const out_dim = static_cast<int>(b.size(1));
  if (num_tokens == 0 || num_groups == 0 || out_dim == 0) {
    return;
  }
  int const num_d_tiles = (out_dim + kOutputTileD - 1) / kOutputTileD;
  int const grid_dim = num_tokens * num_groups * num_d_tiles;

  deepseek_v4_grouped_fp8_einsum_kernel<<<grid_dim, kOutputTileD, 0, stream>>>(
      static_cast<nv_bfloat16*>(output.data_ptr()),
      static_cast<const __nv_fp8_e4m3*>(a.data_ptr()),
      static_cast<const float*>(a_scales.data_ptr()),
      static_cast<const __nv_fp8_e4m3*>(b.data_ptr()),
      b_scales.data_ptr(),
      num_tokens,
      num_groups,
      hidden_dim,
      out_dim,
      num_d_tiles,
      a.stride(0),
      a.stride(1),
      a.stride(2),
      a_scales.stride(0),
      a_scales.stride(1),
      a_scales.stride(2),
      b.stride(0),
      b.stride(1),
      b.stride(2),
      b_scales.stride(0),
      b_scales.stride(1),
      b_scales.stride(2),
      output.stride(0),
      output.stride(1),
      output.stride(2),
      b_scales_are_ue8m0);
  cudaError_t const status = cudaGetLastError();
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "sm12x_deepseek_v4_grouped_fp8_gemv launch failed: "
      << cudaGetErrorString(status);
}

}  // namespace

void sm12x_deepseek_v4_grouped_fp8_gemv(
    TensorView output,
    TensorView a,
    TensorView a_scales,
    TensorView b,
    TensorView b_scales) {
  // `a` and `a_scales` are typically non-contiguous views (post-transpose) of
  // buffers laid out as [G, T, ...] by the fused inverse-RoPE + FP8 quant
  // kernel; only check that they are CUDA tensors and respect strides at the
  // kernel level. `b` (wo_a weight) and `b_scales` must be contiguous so the
  // per-block scale base offset is well-defined.
  CHECK_CUDA(output);
  CHECK_CUDA(a);
  CHECK_CUDA(a_scales);
  CHECK_INPUT(b);
  CHECK_INPUT(b_scales);
  CHECK_DEVICE(output, a);
  CHECK_DEVICE(output, a_scales);
  CHECK_DEVICE(output, b);
  CHECK_DEVICE(output, b_scales);
  CHECK_DIM(3, output);
  CHECK_DIM(3, a);
  CHECK_DIM(3, a_scales);
  CHECK_DIM(3, b);
  CHECK_DIM(3, b_scales);
  TVM_FFI_ICHECK_EQ(output.dtype(), dl_bfloat16);
  TVM_FFI_ICHECK_EQ(a.dtype(), dl_float8_e4m3fn);
  TVM_FFI_ICHECK_EQ(a_scales.dtype(), dl_float32);
  TVM_FFI_ICHECK_EQ(b.dtype(), dl_float8_e4m3fn);

  bool const b_scales_are_ue8m0 = b_scales.dtype() == dl_uint8;
  TVM_FFI_ICHECK(b_scales_are_ue8m0 || b_scales.dtype() == dl_float32)
      << "b_scales dtype must be uint8 UE8M0 bytes or float32";

  int64_t const num_tokens = a.size(0);
  int64_t const num_groups = a.size(1);
  int64_t const hidden_dim = a.size(2);
  int64_t const out_dim = b.size(1);

  TVM_FFI_ICHECK_GE(num_tokens, 0);
  TVM_FFI_ICHECK_GT(num_groups, 0);
  TVM_FFI_ICHECK_GT(hidden_dim, 0);
  TVM_FFI_ICHECK_GT(out_dim, 0);

  TVM_FFI_ICHECK_EQ(b.size(0), num_groups);
  TVM_FFI_ICHECK_EQ(b.size(2), hidden_dim);
  TVM_FFI_ICHECK_EQ(a.stride(2), 1)
      << "a hidden dim must be contiguous (stride==1)";
  TVM_FFI_ICHECK_EQ(b.stride(2), 1)
      << "b hidden dim must be contiguous (stride==1)";

  TVM_FFI_ICHECK_EQ(a_scales.size(0), num_tokens);
  TVM_FFI_ICHECK_EQ(a_scales.size(1), num_groups);
  TVM_FFI_ICHECK_EQ(a_scales.size(2), (hidden_dim + kBlockK - 1) / kBlockK);

  TVM_FFI_ICHECK_EQ(b_scales.size(0), num_groups);
  TVM_FFI_ICHECK_EQ(b_scales.size(1), (out_dim + kBlockK - 1) / kBlockK);
  TVM_FFI_ICHECK_EQ(b_scales.size(2), (hidden_dim + kBlockK - 1) / kBlockK);

  TVM_FFI_ICHECK_EQ(output.size(0), num_tokens);
  TVM_FFI_ICHECK_EQ(output.size(1), num_groups);
  TVM_FFI_ICHECK_EQ(output.size(2), out_dim);

  if (num_tokens == 0) {
    return;
  }

  cudaSetDevice(output.device().device_id);
  cudaStream_t const stream = get_stream(output.device());
  launch_deepseek_v4_grouped_fp8_einsum(
      output, a, a_scales, b, b_scales, stream, b_scales_are_ue8m0);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(sm12x_deepseek_v4_grouped_fp8_gemv,
                              sm12x_deepseek_v4_grouped_fp8_gemv);
