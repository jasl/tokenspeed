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

#include <cmath>
#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "tvm_ffi_utils.h"

using tvm::ffi::TensorView;

namespace {

constexpr int kBlockK = 128;
constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = 4;
constexpr float kFp8E4m3Max = 448.0f;

template <typename scalar_t>
__device__ __forceinline__ float scalar_to_float(scalar_t value);

template <>
__device__ __forceinline__ float scalar_to_float<float>(float value) {
  return value;
}

template <>
__device__ __forceinline__ float scalar_to_float<half>(half value) {
  return __half2float(value);
}

template <>
__device__ __forceinline__ float scalar_to_float<nv_bfloat16>(
    nv_bfloat16 value) {
  return __bfloat162float(value);
}

__device__ __forceinline__ float warp_max(float value) {
  unsigned int const mask = 0xFFFFFFFFu;
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    value = fmaxf(value, __shfl_down_sync(mask, value, offset));
  }
  return value;
}

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
  uint32_t bits = static_cast<uint32_t>(encoded) << 23;
  return __uint_as_float(bits);
}

__device__ __forceinline__ float load_scale(
    const void* __restrict__ scales,
    int offset,
    bool scales_are_ue8m0) {
  if (scales_are_ue8m0) {
    return decode_ue8m0_scale(static_cast<const uint8_t*>(scales)[offset]);
  }
  return static_cast<const float*>(scales)[offset];
}

template <typename scalar_t>
__global__ void mxfp8_block128_quantize_kernel(
    uint8_t* __restrict__ output,
    float* __restrict__ output_scales,
    const scalar_t* __restrict__ values,
    int rows,
    int hidden_dim,
    int scale_k) {
  int const lane = threadIdx.x;
  int const row = blockIdx.y;
  int const block = blockIdx.x;
  int const col = block * kBlockK + lane;

  int64_t const idx = static_cast<int64_t>(row) * hidden_dim + col;
  float const value = scalar_to_float<scalar_t>(values[idx]);
  float max_abs = fabsf(value);
  max_abs = warp_max(max_abs);

  __shared__ float warp_values[4];
  int const warp_id = lane / kWarpSize;
  int const warp_lane = lane % kWarpSize;
  if (warp_lane == 0) {
    warp_values[warp_id] = max_abs;
  }
  __syncthreads();

  float block_max = 0.0f;
  if (warp_id == 0) {
    block_max = lane < 4 ? warp_values[lane] : 0.0f;
    block_max = warp_max(block_max);
    if (lane == 0) {
      warp_values[0] = block_max;
    }
  }
  __syncthreads();

  float const scale = fmaxf(warp_values[0], 1.0e-10f) / kFp8E4m3Max;
  if (lane == 0) {
    output_scales[row * scale_k + block] = scale;
  }

  float const scaled = fminf(fmaxf(value / scale, -kFp8E4m3Max), kFp8E4m3Max);
  reinterpret_cast<__nv_fp8_e4m3*>(output)[idx] = __nv_fp8_e4m3(scaled);
}

template <typename scalar_t>
__global__ void mxfp8_block128_quant_dequant_ue8m0_kernel(
    float* __restrict__ output,
    const scalar_t* __restrict__ values,
    int rows,
    int hidden_dim,
    int scale_k) {
  int const lane = threadIdx.x;
  int const row = blockIdx.y;
  int const block = blockIdx.x;
  int const col = block * kBlockK + lane;

  int64_t const idx = static_cast<int64_t>(row) * hidden_dim + col;
  float const value = scalar_to_float<scalar_t>(values[idx]);
  float max_abs = fabsf(value);
  max_abs = warp_max(max_abs);

  __shared__ float warp_values[4];
  int const warp_id = lane / kWarpSize;
  int const warp_lane = lane % kWarpSize;
  if (warp_lane == 0) {
    warp_values[warp_id] = max_abs;
  }
  __syncthreads();

  float block_max = 0.0f;
  if (warp_id == 0) {
    block_max = lane < 4 ? warp_values[lane] : 0.0f;
    block_max = warp_max(block_max);
    if (lane == 0) {
      warp_values[0] = block_max;
    }
  }
  __syncthreads();

  float const clamped_amax = fmaxf(warp_values[0], 1.0e-4f);
  float const exponent = ceilf(log2f(clamped_amax / kFp8E4m3Max));
  float const scale = exp2f(exponent);
  float const scaled = fminf(fmaxf(value / scale, -kFp8E4m3Max), kFp8E4m3Max);
  __nv_fp8_e4m3 const quantized(scaled);
  output[idx] = static_cast<float>(quantized) * scale;
}

__global__ void fp8_weight_gemv_ue8m0_kernel(
    nv_bfloat16* __restrict__ output,
    const float* __restrict__ values,
    const __nv_fp8_e4m3* __restrict__ weight,
    const void* __restrict__ weight_scales,
    int rows,
    int hidden_dim,
    int out_dim,
    int scale_k,
    bool scales_are_ue8m0) {
  int const out_col = blockIdx.x;
  int const row = blockIdx.y;
  int const tid = threadIdx.x;
  if (row >= rows || out_col >= out_dim) {
    return;
  }

  float accum = 0.0f;
  int const scale_n = out_col / kBlockK;
  int64_t const row_base = static_cast<int64_t>(row) * hidden_dim;
  int64_t const weight_base = static_cast<int64_t>(out_col) * hidden_dim;
  for (int k = tid; k < hidden_dim; k += blockDim.x) {
    int const scale_offset = scale_n * scale_k + k / kBlockK;
    float const w_scale =
        load_scale(weight_scales, scale_offset, scales_are_ue8m0);
    float const w = static_cast<float>(weight[weight_base + k]) * w_scale;
    accum += values[row_base + k] * w;
  }

  accum = block_sum_128(accum);
  if (tid == 0) {
    output[static_cast<int64_t>(row) * out_dim + out_col] =
        __float2bfloat16(accum);
  }
}

template <typename scalar_t>
void launch_mxfp8_block128_quantize(
    TensorView output,
    TensorView output_scales,
    TensorView values,
    cudaStream_t stream) {
  int const rows = static_cast<int>(values.size(0));
  int const hidden_dim = static_cast<int>(values.size(1));
  if (rows == 0) {
    return;
  }
  dim3 const grid(hidden_dim / kBlockK, rows);
  mxfp8_block128_quantize_kernel<scalar_t><<<grid, kBlockK, 0, stream>>>(
      static_cast<uint8_t*>(output.data_ptr()),
      static_cast<float*>(output_scales.data_ptr()),
      static_cast<const scalar_t*>(values.data_ptr()),
      rows,
      hidden_dim,
      static_cast<int>(output_scales.size(1)));
  cudaError_t const status = cudaGetLastError();
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "sm12x_mxfp8_block128_quantize launch failed: "
      << cudaGetErrorString(status);
}

template <typename scalar_t>
void launch_mxfp8_block128_quant_dequant_ue8m0(
    TensorView output,
    TensorView values,
    cudaStream_t stream) {
  int const rows = static_cast<int>(values.size(0));
  int const hidden_dim = static_cast<int>(values.size(1));
  if (rows == 0) {
    return;
  }
  dim3 const grid(hidden_dim / kBlockK, rows);
  mxfp8_block128_quant_dequant_ue8m0_kernel<scalar_t>
      <<<grid, kBlockK, 0, stream>>>(
          static_cast<float*>(output.data_ptr()),
          static_cast<const scalar_t*>(values.data_ptr()),
          rows,
          hidden_dim,
          static_cast<int>(output.size(1) / kBlockK));
  cudaError_t const status = cudaGetLastError();
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "sm12x_mxfp8_block128_quant_dequant_ue8m0 launch failed: "
      << cudaGetErrorString(status);
}

void launch_fp8_weight_gemv_ue8m0(
    TensorView output,
    TensorView values,
    TensorView weight,
    TensorView weight_scales,
    cudaStream_t stream,
    bool scales_are_ue8m0) {
  int const rows = static_cast<int>(values.size(0));
  int const hidden_dim = static_cast<int>(values.size(1));
  int const out_dim = static_cast<int>(weight.size(0));
  if (rows == 0 || out_dim == 0) {
    return;
  }
  dim3 const grid(out_dim, rows);
  fp8_weight_gemv_ue8m0_kernel<<<grid, 128, 0, stream>>>(
      static_cast<nv_bfloat16*>(output.data_ptr()),
      static_cast<const float*>(values.data_ptr()),
      static_cast<const __nv_fp8_e4m3*>(weight.data_ptr()),
      weight_scales.data_ptr(),
      rows,
      hidden_dim,
      out_dim,
      static_cast<int>(weight_scales.size(1)),
      scales_are_ue8m0);
  cudaError_t const status = cudaGetLastError();
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "sm12x_fp8_weight_gemv_ue8m0 launch failed: "
      << cudaGetErrorString(status);
}

}  // namespace

void sm12x_mxfp8_block128_quantize(
    TensorView output,
    TensorView output_scales,
    TensorView values) {
  CHECK_INPUT(output);
  CHECK_INPUT(output_scales);
  CHECK_INPUT(values);
  CHECK_DEVICE(output, output_scales);
  CHECK_DEVICE(output, values);
  CHECK_DIM(2, output);
  CHECK_DIM(2, output_scales);
  CHECK_DIM(2, values);
  TVM_FFI_ICHECK(output.IsContiguous()) << "output must be contiguous";
  TVM_FFI_ICHECK(output_scales.IsContiguous()) << "output_scales must be contiguous";
  TVM_FFI_ICHECK(values.IsContiguous()) << "values must be contiguous";
  TVM_FFI_ICHECK_EQ(output.dtype(), dl_float8_e4m3fn);
  TVM_FFI_ICHECK_EQ(output_scales.dtype(), dl_float32);

  int64_t const rows = values.size(0);
  int64_t const hidden_dim = values.size(1);
  TVM_FFI_ICHECK_GE(rows, 0);
  TVM_FFI_ICHECK_GT(hidden_dim, 0);
  TVM_FFI_ICHECK_EQ(hidden_dim % kBlockK, 0);
  TVM_FFI_ICHECK_EQ(output.size(0), rows);
  TVM_FFI_ICHECK_EQ(output.size(1), hidden_dim);
  TVM_FFI_ICHECK_EQ(output_scales.size(0), rows);
  TVM_FFI_ICHECK_EQ(output_scales.size(1), hidden_dim / kBlockK);

  cudaSetDevice(output.device().device_id);
  cudaStream_t const stream = get_stream(output.device());
  if (values.dtype() == dl_float32) {
    launch_mxfp8_block128_quantize<float>(output, output_scales, values, stream);
  } else if (values.dtype() == dl_float16) {
    launch_mxfp8_block128_quantize<half>(output, output_scales, values, stream);
  } else if (values.dtype() == dl_bfloat16) {
    launch_mxfp8_block128_quantize<nv_bfloat16>(
        output, output_scales, values, stream);
  } else {
    TVM_FFI_ICHECK(false)
        << "values dtype must be float32, float16, or bfloat16";
  }
}

void sm12x_mxfp8_block128_quant_dequant_ue8m0(
    TensorView output,
    TensorView values) {
  CHECK_INPUT(output);
  CHECK_INPUT(values);
  CHECK_DEVICE(output, values);
  CHECK_DIM(2, output);
  CHECK_DIM(2, values);
  TVM_FFI_ICHECK(output.IsContiguous()) << "output must be contiguous";
  TVM_FFI_ICHECK(values.IsContiguous()) << "values must be contiguous";
  TVM_FFI_ICHECK_EQ(output.dtype(), dl_float32);

  int64_t const rows = values.size(0);
  int64_t const hidden_dim = values.size(1);
  TVM_FFI_ICHECK_GE(rows, 0);
  TVM_FFI_ICHECK_GT(hidden_dim, 0);
  TVM_FFI_ICHECK_EQ(hidden_dim % kBlockK, 0);
  TVM_FFI_ICHECK_EQ(output.size(0), rows);
  TVM_FFI_ICHECK_EQ(output.size(1), hidden_dim);

  cudaSetDevice(output.device().device_id);
  cudaStream_t const stream = get_stream(output.device());
  if (values.dtype() == dl_float32) {
    launch_mxfp8_block128_quant_dequant_ue8m0<float>(output, values, stream);
  } else if (values.dtype() == dl_float16) {
    launch_mxfp8_block128_quant_dequant_ue8m0<half>(output, values, stream);
  } else if (values.dtype() == dl_bfloat16) {
    launch_mxfp8_block128_quant_dequant_ue8m0<nv_bfloat16>(
        output, values, stream);
  } else {
    TVM_FFI_ICHECK(false)
        << "values dtype must be float32, float16, or bfloat16";
  }
}

void sm12x_fp8_weight_gemv_ue8m0(
    TensorView output,
    TensorView values,
    TensorView weight,
    TensorView weight_scales) {
  CHECK_INPUT(output);
  CHECK_INPUT(values);
  CHECK_INPUT(weight);
  CHECK_INPUT(weight_scales);
  CHECK_DEVICE(output, values);
  CHECK_DEVICE(output, weight);
  CHECK_DEVICE(output, weight_scales);
  CHECK_DIM(2, output);
  CHECK_DIM(2, values);
  CHECK_DIM(2, weight);
  CHECK_DIM(2, weight_scales);
  TVM_FFI_ICHECK(output.IsContiguous()) << "output must be contiguous";
  TVM_FFI_ICHECK(values.IsContiguous()) << "values must be contiguous";
  TVM_FFI_ICHECK(weight.IsContiguous()) << "weight must be contiguous";
  TVM_FFI_ICHECK(weight_scales.IsContiguous())
      << "weight_scales must be contiguous";
  TVM_FFI_ICHECK_EQ(output.dtype(), dl_bfloat16);
  TVM_FFI_ICHECK_EQ(values.dtype(), dl_float32);
  TVM_FFI_ICHECK_EQ(weight.dtype(), dl_float8_e4m3fn);

  bool const scales_are_ue8m0 = weight_scales.dtype() == dl_uint8;
  TVM_FFI_ICHECK(scales_are_ue8m0 || weight_scales.dtype() == dl_float32)
      << "weight_scales dtype must be uint8 UE8M0 bytes or float32";

  int64_t const rows = values.size(0);
  int64_t const hidden_dim = values.size(1);
  int64_t const out_dim = weight.size(0);
  TVM_FFI_ICHECK_GE(rows, 0);
  TVM_FFI_ICHECK_GT(hidden_dim, 0);
  TVM_FFI_ICHECK_GT(out_dim, 0);
  TVM_FFI_ICHECK_EQ(hidden_dim % kBlockK, 0);
  TVM_FFI_ICHECK_EQ(weight.size(1), hidden_dim);
  TVM_FFI_ICHECK_EQ(output.size(0), rows);
  TVM_FFI_ICHECK_EQ(output.size(1), out_dim);
  TVM_FFI_ICHECK_EQ(weight_scales.size(0), (out_dim + kBlockK - 1) / kBlockK);
  TVM_FFI_ICHECK_EQ(weight_scales.size(1), hidden_dim / kBlockK);

  cudaSetDevice(output.device().device_id);
  cudaStream_t const stream = get_stream(output.device());
  launch_fp8_weight_gemv_ue8m0(
      output, values, weight, weight_scales, stream, scales_are_ue8m0);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(sm12x_mxfp8_block128_quantize,
                              sm12x_mxfp8_block128_quantize);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(sm12x_mxfp8_block128_quant_dequant_ue8m0,
                              sm12x_mxfp8_block128_quant_dequant_ue8m0);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(sm12x_fp8_weight_gemv_ue8m0,
                              sm12x_fp8_weight_gemv_ue8m0);
