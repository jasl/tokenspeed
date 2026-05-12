// Copyright (c) 2026 LightSeek Foundation
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
// This kernel targets the DeepSeek V4 Flash decode shape (small num_tokens,
// num_groups ~= 8, hidden ~= 2048, out_rank ~= 1024) where the operation is
// memory-bound. The schedule mirrors the existing
// `fp8_weight_gemv_ue8m0_kernel` in `sm12x_fp8_quantize.cu`: one CUDA block
// per (group, token, out_col) computes a single dot product over `hidden`
// with block-128 scale composition, then reduces across 128 threads via
// `block_sum_128`.

#include <cmath>
#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include "tvm_ffi_utils.h"

using tvm::ffi::TensorView;

namespace {

constexpr int kBlockK = 128;
constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = 4;

__device__ __forceinline__ float warp_sum(float value) {
  unsigned int const mask = 0xFFFFFFFFu;
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    value += __shfl_down_sync(mask, value, offset);
  }
  return value;
}

__device__ __forceinline__ float block_sum_128(float value) {
  __shared__ float warp_partials[kWarpsPerBlock];
  int const lane = threadIdx.x % kWarpSize;
  int const warp = threadIdx.x / kWarpSize;
  value = warp_sum(value);
  if (lane == 0) {
    warp_partials[warp] = value;
  }
  __syncthreads();

  value = threadIdx.x < kWarpsPerBlock ? warp_partials[lane] : 0.0f;
  if (warp == 0) {
    value = warp_sum(value);
  }
  return value;
}

__device__ __forceinline__ float decode_ue8m0_scale(uint8_t encoded) {
  // UE8M0 unbiased exponent -> float: shift to fp32 exponent field.
  uint32_t bits = static_cast<uint32_t>(encoded) << 23;
  return __uint_as_float(bits);
}

__device__ __forceinline__ float load_b_scale(
    const void* __restrict__ scales,
    int64_t offset,
    bool scales_are_ue8m0) {
  if (scales_are_ue8m0) {
    return decode_ue8m0_scale(static_cast<const uint8_t*>(scales)[offset]);
  }
  return static_cast<const float*>(scales)[offset];
}

__global__ void deepseek_v4_grouped_fp8_gemv_kernel(
    nv_bfloat16* __restrict__ output,
    const __nv_fp8_e4m3* __restrict__ a,
    const float* __restrict__ a_scales,
    const __nv_fp8_e4m3* __restrict__ b,
    const void* __restrict__ b_scales,
    int num_tokens,
    int num_groups,
    int hidden_dim,
    int out_dim,
    int hidden_blocks,
    int b_scale_out_blocks,
    int64_t a_stride_token,
    int64_t a_stride_group,
    int64_t a_stride_hidden,
    int64_t a_scale_stride_token,
    int64_t a_scale_stride_group,
    int64_t a_scale_stride_block,
    int64_t out_stride_token,
    int64_t out_stride_group,
    int64_t out_stride_rank,
    bool b_scales_are_ue8m0) {
  int const n = blockIdx.x;
  int const t = blockIdx.y;
  int const g = blockIdx.z;
  int const tid = threadIdx.x;

  if (n >= out_dim || t >= num_tokens || g >= num_groups) {
    return;
  }

  // Per-(t,g,n) base pointers.
  const __nv_fp8_e4m3* a_tg =
      a + t * a_stride_token + g * a_stride_group;
  const float* a_scale_tg =
      a_scales + t * a_scale_stride_token + g * a_scale_stride_group;
  // b is contiguous in [G, N, K]; row stride = hidden_dim.
  const __nv_fp8_e4m3* b_gn =
      b + (static_cast<int64_t>(g) * out_dim + n) * hidden_dim;
  int64_t const b_scale_n_block = n / kBlockK;
  int64_t const b_scale_base =
      (static_cast<int64_t>(g) * b_scale_out_blocks + b_scale_n_block)
          * hidden_blocks;

  float accum = 0.0f;
  for (int kb = 0; kb < hidden_blocks; ++kb) {
    float const a_s = a_scale_tg[kb * a_scale_stride_block];
    float const b_s =
        load_b_scale(b_scales, b_scale_base + kb, b_scales_are_ue8m0);
    float const scale = a_s * b_s;

    int const k = kb * kBlockK + tid;
    float const a_v = static_cast<float>(a_tg[k * a_stride_hidden]);
    float const b_v = static_cast<float>(b_gn[k]);
    accum += a_v * b_v * scale;
  }

  accum = block_sum_128(accum);
  if (tid == 0) {
    int64_t const out_idx = static_cast<int64_t>(t) * out_stride_token
                            + static_cast<int64_t>(g) * out_stride_group
                            + static_cast<int64_t>(n) * out_stride_rank;
    output[out_idx] = __float2bfloat16(accum);
  }
}

void launch_deepseek_v4_grouped_fp8_gemv(
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
  int const hidden_blocks = hidden_dim / kBlockK;
  int const b_scale_out_blocks =
      static_cast<int>(b_scales.size(1));

  dim3 const grid(out_dim, num_tokens, num_groups);
  deepseek_v4_grouped_fp8_gemv_kernel<<<grid, kBlockK, 0, stream>>>(
      static_cast<nv_bfloat16*>(output.data_ptr()),
      static_cast<const __nv_fp8_e4m3*>(a.data_ptr()),
      static_cast<const float*>(a_scales.data_ptr()),
      static_cast<const __nv_fp8_e4m3*>(b.data_ptr()),
      b_scales.data_ptr(),
      num_tokens,
      num_groups,
      hidden_dim,
      out_dim,
      hidden_blocks,
      b_scale_out_blocks,
      a.stride(0),
      a.stride(1),
      a.stride(2),
      a_scales.stride(0),
      a_scales.stride(1),
      a_scales.stride(2),
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
  // kernel can index them with a single base + offset.
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
  int64_t const hidden_blocks = hidden_dim / kBlockK;
  int64_t const b_scale_out_blocks = b_scales.size(1);

  TVM_FFI_ICHECK_GE(num_tokens, 0);
  TVM_FFI_ICHECK_GT(num_groups, 0);
  TVM_FFI_ICHECK_GT(hidden_dim, 0);
  TVM_FFI_ICHECK_GT(out_dim, 0);
  TVM_FFI_ICHECK_EQ(hidden_dim % kBlockK, 0)
      << "hidden_dim must be divisible by 128";
  TVM_FFI_ICHECK_EQ(out_dim % kBlockK, 0)
      << "out_dim must be divisible by 128";

  TVM_FFI_ICHECK_EQ(b.size(0), num_groups);
  TVM_FFI_ICHECK_EQ(b.size(2), hidden_dim);
  TVM_FFI_ICHECK_EQ(a.stride(2), 1)
      << "a hidden dim must be contiguous (stride==1)";

  TVM_FFI_ICHECK_EQ(a_scales.size(0), num_tokens);
  TVM_FFI_ICHECK_EQ(a_scales.size(1), num_groups);
  TVM_FFI_ICHECK_EQ(a_scales.size(2), hidden_blocks);

  TVM_FFI_ICHECK_EQ(b_scales.size(0), num_groups);
  TVM_FFI_ICHECK_EQ(b_scale_out_blocks, (out_dim + kBlockK - 1) / kBlockK);
  TVM_FFI_ICHECK_EQ(b_scales.size(2), hidden_blocks);

  TVM_FFI_ICHECK_EQ(output.size(0), num_tokens);
  TVM_FFI_ICHECK_EQ(output.size(1), num_groups);
  TVM_FFI_ICHECK_EQ(output.size(2), out_dim);

  if (num_tokens == 0) {
    return;
  }

  cudaSetDevice(output.device().device_id);
  cudaStream_t const stream = get_stream(output.device());
  launch_deepseek_v4_grouped_fp8_gemv(
      output, a, a_scales, b, b_scales, stream, b_scales_are_ue8m0);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(sm12x_deepseek_v4_grouped_fp8_gemv,
                              sm12x_deepseek_v4_grouped_fp8_gemv);
