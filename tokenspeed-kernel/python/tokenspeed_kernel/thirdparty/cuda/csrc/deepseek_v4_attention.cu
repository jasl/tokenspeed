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
//
// DeepSeek V4 fused SWA cache insert.
//
// Cache layout per paged block:
//   [0, block_size * 576): token data, each token [448 fp8 bytes | 64 bf16/fp16]
//   [block_size * 576, block_size * 584): scale bytes, 8 per token

#include <cmath>
#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include "tvm_ffi_utils.h"

using tvm::ffi::TensorView;

namespace {

constexpr int kHeadDim = 512;
constexpr int kRopeDim = 64;
constexpr int kHalfRopeDim = kRopeDim / 2;
constexpr int kNopeDim = kHeadDim - kRopeDim;
constexpr int kQuantBlock = 64;
constexpr int kNumQuantBlocks = kNopeDim / kQuantBlock;
constexpr int kScaleBytesPerToken = kNumQuantBlocks + 1;
constexpr int kTokenDataBytes = kNopeDim + kRopeDim * 2;
constexpr int kThreads = 256;
constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = kThreads / kWarpSize;
constexpr int kValuesPerThread = (kHeadDim + kThreads - 1) / kThreads;
constexpr int kValuesPerLane = kHeadDim / kWarpSize;
constexpr int kIndexerDim = 128;
constexpr int kIndexerRopeDim = 64;
constexpr int kIndexerHalfRopeDim = kIndexerRopeDim / 2;
constexpr int kIndexerNopeDim = kIndexerDim - kIndexerRopeDim;
constexpr int kIndexerStateWidth = kIndexerDim * 2;
constexpr int kCsaCompressRatio = 4;
constexpr int kCsaWindow = kCsaCompressRatio * 2;
constexpr int kMaxCompressWindow = 128;
constexpr int kIndexerCacheValueBytes = kIndexerDim;
constexpr int kIndexerCacheScaleBytes = 4;
constexpr int kSm12xMhcMaxMix = 32;
constexpr int kSm12xMhcMaxHc = 8;
constexpr float kFp8Max = 448.0f;
constexpr float kNegInf = -3.4028234663852886e+38F;

template <typename scalar_t>
__device__ __forceinline__ float scalar_to_float(scalar_t value);

template <>
__device__ __forceinline__ float scalar_to_float<half>(half value) {
  return __half2float(value);
}

template <>
__device__ __forceinline__ float scalar_to_float<nv_bfloat16>(nv_bfloat16 value) {
  return __bfloat162float(value);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t float_to_scalar(float value);

template <>
__device__ __forceinline__ half float_to_scalar<half>(float value) {
  return __float2half_rn(value);
}

template <>
__device__ __forceinline__ nv_bfloat16 float_to_scalar<nv_bfloat16>(float value) {
  return __float2bfloat16(value);
}

__device__ __forceinline__ float bf16_roundtrip(float value) {
  return __bfloat162float(__float2bfloat16(value));
}

__device__ __forceinline__ float sigmoidf_approx(float value) {
  return 1.0f / (1.0f + expf(-value));
}

__device__ __forceinline__ uint8_t encode_ue8m0_scale(float exponent) {
  float encoded = fminf(fmaxf(exponent + 127.0f, 0.0f), 255.0f);
  return static_cast<uint8_t>(encoded);
}

__device__ __forceinline__ float warp_sum(float value) {
  unsigned int const mask = 0xFFFFFFFFu;
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(mask, value, offset);
  }
  return value;
}

__device__ __forceinline__ float warp_sum_all(float value) {
  unsigned int const mask = 0xFFFFFFFFu;
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    value += __shfl_xor_sync(mask, value, offset, kWarpSize);
  }
  return value;
}

__device__ __forceinline__ float warp4_max_all(float value) {
  unsigned int const mask = 0xFFFFFFFFu;
  value = fmaxf(value, __shfl_xor_sync(mask, value, 1, kWarpSize));
  value = fmaxf(value, __shfl_xor_sync(mask, value, 2, kWarpSize));
  return value;
}

__device__ __forceinline__ float block_sum_256(
    float value,
    float* __restrict__ warp_partials) {
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp = threadIdx.x / kWarpSize;
  value = warp_sum(value);
  if (lane == 0) {
    warp_partials[warp] = value;
  }
  __syncthreads();

  value = threadIdx.x < kWarpsPerBlock ? warp_partials[lane] : 0.0f;
  if (warp == 0) {
    value = warp_sum(value);
  }
  __syncthreads();
  return value;
}

__device__ __forceinline__ void hadamard128_inplace(
    float* __restrict__ values) {
  const int tid = threadIdx.x;
  for (int step = 1; step < kIndexerDim; step <<= 1) {
    if (tid < kIndexerDim / 2) {
      const int group = tid / step;
      const int offset = tid - group * step;
      const int base = group * (step << 1) + offset;
      const float even = values[base];
      const float odd = values[base + step];
      values[base] = even + odd;
      values[base + step] = even - odd;
    }
    __syncthreads();
  }
}

template <typename scalar_t>
__device__ __forceinline__ float load_fp8_ds_mla_cache_value(
    const uint8_t* __restrict__ cache,
    int64_t cache_block_stride,
    int cache_block_size,
    int32_t slot,
    int dim) {
  if (slot < 0) {
    return 0.0f;
  }
  const int64_t block_idx = slot / cache_block_size;
  const int64_t pos_in_block = slot % cache_block_size;
  const uint8_t* block_base = cache + block_idx * cache_block_stride;
  const uint8_t* token_data = block_base + pos_in_block * kTokenDataBytes;
  const uint8_t* token_scales =
      block_base + static_cast<int64_t>(cache_block_size) * kTokenDataBytes +
      pos_in_block * kScaleBytesPerToken;

  if (dim < kNopeDim) {
    __nv_fp8_e4m3 value;
    value.__x = static_cast<__nv_fp8_storage_t>(token_data[dim]);
    const float scale =
        exp2f(static_cast<int>(token_scales[dim / kQuantBlock]) - 127);
    return static_cast<float>(value) * scale;
  }

  const scalar_t* rope_tail =
      reinterpret_cast<const scalar_t*>(token_data + kNopeDim);
  return scalar_to_float(rope_tail[dim - kNopeDim]);
}

template <typename scalar_t>
__device__ __forceinline__ float load_fp8_ds_mla_cache_value_cached(
    const uint8_t* __restrict__ token_data,
    const float* __restrict__ scales,
    int dim) {
  if (dim < kNopeDim) {
    __nv_fp8_e4m3 value;
    value.__x = static_cast<__nv_fp8_storage_t>(token_data[dim]);
    return static_cast<float>(value) * scales[dim / kQuantBlock];
  }

  const scalar_t* rope_tail =
      reinterpret_cast<const scalar_t*>(token_data + kNopeDim);
  return scalar_to_float(rope_tail[dim - kNopeDim]);
}

__global__ void csa_indexer_cache_insert_fp8_kernel(
    const float* __restrict__ state_cache,
    const int32_t* __restrict__ token_to_req_indices,
    const int64_t* __restrict__ positions,
    const int64_t* __restrict__ compressor_slot_mapping,
    const int32_t* __restrict__ block_table,
    const float* __restrict__ rms_norm_weight,
    const float* __restrict__ cos_sin_cache,
    uint8_t* __restrict__ kv_cache,
    const int64_t* __restrict__ kv_slot_mapping,
    float rms_norm_eps,
    int num_tokens,
    int state_block_size,
    int64_t state_cache_block_stride,
    int64_t state_cache_row_stride,
    int block_table_width,
    int kv_cache_block_size,
    int64_t kv_cache_block_stride,
    int64_t cos_sin_stride) {
  const int token_idx = blockIdx.x;
  const int tid = threadIdx.x;
  if (token_idx >= num_tokens) {
    return;
  }

  const int64_t position = positions[token_idx];
  const int64_t state_slot = compressor_slot_mapping[token_idx];
  const int64_t kv_slot = kv_slot_mapping[token_idx];
  if (state_slot < 0 || kv_slot < 0 ||
      ((position + 1) % kCsaCompressRatio) != 0 || block_table_width <= 0) {
    return;
  }

  const int32_t req_idx = token_to_req_indices[token_idx];
  if (req_idx < 0) {
    return;
  }

  __shared__ float compressed[kIndexerDim];
  __shared__ float values[kIndexerDim];
  __shared__ float reduction[kThreads];
  __shared__ float shared_variance_sum;
  __shared__ int32_t pages[kCsaWindow];
  __shared__ int32_t pos_in_block[kCsaWindow];
  __shared__ int valid_count;

  if (tid < kCsaWindow) {
    const int64_t local_pos = position - kCsaWindow + 1 + tid;
    int32_t page = -1;
    int32_t row_pos = 0;
    if (local_pos >= 0) {
      const int64_t table_idx = local_pos / state_block_size;
      if (table_idx >= 0 && table_idx < block_table_width) {
        page = block_table[static_cast<int64_t>(req_idx) * block_table_width +
                           table_idx];
        row_pos = static_cast<int32_t>(local_pos % state_block_size);
      }
    }
    pages[tid] = page;
    pos_in_block[tid] = row_pos;
  }
  if (tid == 0) {
    valid_count = 0;
  }
  __syncthreads();

  if (tid < kCsaWindow && pages[tid] >= 0) {
    atomicAdd(&valid_count, 1);
  }
  __syncthreads();
  if (valid_count == 0) {
    return;
  }

  if (tid < kIndexerDim) {
    float logits[kCsaWindow];
    float max_score = kNegInf;
#pragma unroll
    for (int offset = 0; offset < kCsaWindow; ++offset) {
      float score = kNegInf;
      if (pages[offset] >= 0) {
        const int head_offset =
            offset >= kCsaCompressRatio ? kIndexerDim : 0;
        const float* row =
            state_cache +
            static_cast<int64_t>(pages[offset]) * state_cache_block_stride +
            static_cast<int64_t>(pos_in_block[offset]) * state_cache_row_stride;
        score = row[kIndexerStateWidth + head_offset + tid];
      }
      logits[offset] = score;
      max_score = fmaxf(max_score, score);
    }

    float denom = 0.0f;
    float accum = 0.0f;
#pragma unroll
    for (int offset = 0; offset < kCsaWindow; ++offset) {
      if (pages[offset] < 0) {
        continue;
      }
      const float weight = expf(logits[offset] - max_score);
      const int head_offset =
          offset >= kCsaCompressRatio ? kIndexerDim : 0;
      const float* row =
          state_cache +
          static_cast<int64_t>(pages[offset]) * state_cache_block_stride +
          static_cast<int64_t>(pos_in_block[offset]) * state_cache_row_stride;
      accum += row[head_offset + tid] * weight;
      denom += weight;
    }
    compressed[tid] = denom > 0.0f ? accum / denom : 0.0f;
    reduction[tid] = compressed[tid] * compressed[tid];
  } else {
    reduction[tid] = 0.0f;
  }
  __syncthreads();

  const float variance_sum = block_sum_256(reduction[tid], reduction);
  if (tid == 0) {
    shared_variance_sum = variance_sum;
  }
  __syncthreads();
  const float rms_scale =
      rsqrtf(shared_variance_sum / static_cast<float>(kIndexerDim) + rms_norm_eps);
  if (tid < kIndexerDim) {
    values[tid] = compressed[tid] * rms_scale * rms_norm_weight[tid];
  }
  __syncthreads();

  const int64_t compressed_position =
      (position / kCsaCompressRatio) * kCsaCompressRatio;
  const float* cos_ptr = cos_sin_cache + compressed_position * cos_sin_stride;
  const float* sin_ptr = cos_ptr + kIndexerHalfRopeDim;
  if (tid < kIndexerNopeDim) {
    values[tid] = bf16_roundtrip(values[tid]);
  } else if (tid < kIndexerDim) {
    const int pair = (tid - kIndexerNopeDim) >> 1;
    const int dim_even = kIndexerNopeDim + pair * 2;
    const float x_even = values[dim_even];
    const float x_odd = values[dim_even + 1];
    const float cos_v = cos_ptr[pair];
    const float sin_v = sin_ptr[pair];
    const float rotated =
        ((tid - kIndexerNopeDim) & 1) == 0
            ? x_even * cos_v - x_odd * sin_v
            : x_even * sin_v + x_odd * cos_v;
    values[tid] = bf16_roundtrip(rotated);
  }
  __syncthreads();

  hadamard128_inplace(values);
  if (tid < kIndexerDim) {
    const float scaled = values[tid] * rsqrtf(static_cast<float>(kIndexerDim));
    values[tid] = bf16_roundtrip(scaled);
    reduction[tid] = fabsf(values[tid]);
  } else {
    reduction[tid] = 0.0f;
  }
  __syncthreads();

  for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduction[tid] = fmaxf(reduction[tid], reduction[tid + stride]);
    }
    __syncthreads();
  }

  const float absmax = fmaxf(reduction[0], 1.0e-10f);
  const float scale = exp2f(ceilf(log2f(absmax / kFp8Max)));
  const int64_t block_idx = kv_slot / kv_cache_block_size;
  const int64_t cache_pos = kv_slot % kv_cache_block_size;
  uint8_t* block_base = kv_cache + block_idx * kv_cache_block_stride;
  uint8_t* value_base = block_base + cache_pos * kIndexerCacheValueBytes;
  uint8_t* scale_base =
      block_base + static_cast<int64_t>(kv_cache_block_size) *
                       kIndexerCacheValueBytes +
      cache_pos * kIndexerCacheScaleBytes;

  if (tid < kIndexerDim) {
    float quant_value = values[tid] / scale;
    quant_value = fminf(fmaxf(quant_value, -kFp8Max), kFp8Max);
    const __nv_fp8_storage_t storage =
        __nv_cvt_float_to_fp8(quant_value, __NV_SATFINITE, __NV_E4M3);
    value_base[tid] = static_cast<uint8_t>(storage);
  }
  if (tid == 0) {
    const uint32_t scale_bits = __float_as_uint(scale);
    scale_base[0] = static_cast<uint8_t>(scale_bits & 0xFFu);
    scale_base[1] = static_cast<uint8_t>((scale_bits >> 8) & 0xFFu);
    scale_base[2] = static_cast<uint8_t>((scale_bits >> 16) & 0xFFu);
    scale_base[3] = static_cast<uint8_t>((scale_bits >> 24) & 0xFFu);
  }
}

__global__ void compressed_kv_cache_insert_kernel(
    const float* __restrict__ state_cache,
    const int32_t* __restrict__ token_to_req_indices,
    const int64_t* __restrict__ positions,
    const int64_t* __restrict__ compressor_slot_mapping,
    const int32_t* __restrict__ block_table,
    const float* __restrict__ rms_norm_weight,
    const float* __restrict__ cos_sin_cache,
    uint8_t* __restrict__ kv_cache,
    const int64_t* __restrict__ kv_slot_mapping,
    float rms_norm_eps,
    int num_tokens,
    int state_block_size,
    int64_t state_cache_block_stride,
    int64_t state_cache_row_stride,
    int block_table_width,
    int kv_cache_block_size,
    int64_t kv_cache_block_stride,
    int64_t cos_sin_stride,
    int compress_ratio,
    int window,
    int state_width,
    int overlap) {
  const int token_idx = blockIdx.x;
  const int tid = threadIdx.x;
  if (token_idx >= num_tokens) {
    return;
  }

  const int64_t position = positions[token_idx];
  const int64_t state_slot = compressor_slot_mapping[token_idx];
  const int64_t kv_slot = kv_slot_mapping[token_idx];
  if (state_slot < 0 || kv_slot < 0 || ((position + 1) % compress_ratio) != 0 ||
      block_table_width <= 0) {
    return;
  }

  const int32_t req_idx = token_to_req_indices[token_idx];
  if (req_idx < 0) {
    return;
  }

  __shared__ float values[kHeadDim];
  __shared__ float rope_values[kRopeDim];
  __shared__ float reduction[kThreads];
  __shared__ float scales[kNumQuantBlocks];
  __shared__ int32_t pages[kMaxCompressWindow];
  __shared__ int32_t pos_in_block[kMaxCompressWindow];
  __shared__ float variance_sum;
  __shared__ int valid_count;

  if (tid < window) {
    const int64_t local_pos = position - window + 1 + tid;
    int32_t page = -1;
    int32_t row_pos = 0;
    if (local_pos >= 0) {
      const int64_t table_idx = local_pos / state_block_size;
      if (table_idx >= 0 && table_idx < block_table_width) {
        page = block_table[static_cast<int64_t>(req_idx) * block_table_width +
                           table_idx];
        row_pos = static_cast<int32_t>(local_pos % state_block_size);
      }
    }
    pages[tid] = page;
    pos_in_block[tid] = row_pos;
  }
  if (tid == 0) {
    valid_count = 0;
  }
  __syncthreads();

  if (tid < window && pages[tid] >= 0) {
    atomicAdd(&valid_count, 1);
  }
  __syncthreads();
  if (valid_count == 0) {
    return;
  }

  float local_variance = 0.0f;
  for (int dim = tid; dim < kHeadDim; dim += blockDim.x) {
    float max_score = kNegInf;
    for (int offset = 0; offset < window; ++offset) {
      float score = kNegInf;
      if (pages[offset] >= 0) {
        const int head_offset =
            (overlap != 0 && offset >= compress_ratio) ? kHeadDim : 0;
        const float* row =
            state_cache +
            static_cast<int64_t>(pages[offset]) * state_cache_block_stride +
            static_cast<int64_t>(pos_in_block[offset]) * state_cache_row_stride;
        score = row[state_width + head_offset + dim];
      }
      max_score = fmaxf(max_score, score);
    }

    float denom = 0.0f;
    float accum = 0.0f;
    for (int offset = 0; offset < window; ++offset) {
      if (pages[offset] < 0) {
        continue;
      }
      const int head_offset =
          (overlap != 0 && offset >= compress_ratio) ? kHeadDim : 0;
      const float* row =
          state_cache +
          static_cast<int64_t>(pages[offset]) * state_cache_block_stride +
          static_cast<int64_t>(pos_in_block[offset]) * state_cache_row_stride;
      const float score = row[state_width + head_offset + dim];
      const float weight = expf(score - max_score);
      accum += row[head_offset + dim] * weight;
      denom += weight;
    }
    const float compressed = denom > 0.0f ? accum / denom : 0.0f;
    values[dim] = compressed;
    local_variance += compressed * compressed;
  }
  reduction[tid] = local_variance;
  __syncthreads();

  const float sum = block_sum_256(reduction[tid], reduction);
  if (tid == 0) {
    variance_sum = sum;
  }
  __syncthreads();

  const float rms_scale =
      rsqrtf(variance_sum / static_cast<float>(kHeadDim) + rms_norm_eps);
  for (int dim = tid; dim < kHeadDim; dim += blockDim.x) {
    values[dim] = values[dim] * rms_scale * rms_norm_weight[dim];
  }
  __syncthreads();

  for (int dim = tid; dim < kRopeDim; dim += blockDim.x) {
    rope_values[dim] = values[kNopeDim + dim];
  }
  __syncthreads();

  const int64_t compressed_position = (position / compress_ratio) * compress_ratio;
  const float* cos_ptr = cos_sin_cache + compressed_position * cos_sin_stride;
  const float* sin_ptr = cos_ptr + kHalfRopeDim;
  for (int pair = tid; pair < kHalfRopeDim; pair += blockDim.x) {
    const int dim = pair * 2;
    const float x_even = rope_values[dim];
    const float x_odd = rope_values[dim + 1];
    const float cos_v = cos_ptr[pair];
    const float sin_v = sin_ptr[pair];
    rope_values[dim] = x_even * cos_v - x_odd * sin_v;
    rope_values[dim + 1] = x_even * sin_v + x_odd * cos_v;
  }
  __syncthreads();

  const int64_t block_idx = kv_slot / kv_cache_block_size;
  const int64_t cache_pos = kv_slot % kv_cache_block_size;
  uint8_t* block_base = kv_cache + block_idx * kv_cache_block_stride;
  uint8_t* token_data = block_base + cache_pos * kTokenDataBytes;
  uint8_t* token_scales =
      block_base + static_cast<int64_t>(kv_cache_block_size) * kTokenDataBytes +
      cache_pos * kScaleBytesPerToken;

  if (tid < kNumQuantBlocks) {
    float absmax = 0.0f;
    const int start = tid * kQuantBlock;
    for (int i = 0; i < kQuantBlock; ++i) {
      const float rounded = bf16_roundtrip(values[start + i]);
      values[start + i] = rounded;
      absmax = fmaxf(absmax, fabsf(rounded));
    }
    absmax = fmaxf(absmax, 1.0e-4f);
    const float exponent = ceilf(log2f(absmax / kFp8Max));
    scales[tid] = exponent;
    token_scales[tid] = encode_ue8m0_scale(exponent);
  }
  if (tid == 0) {
    token_scales[kNumQuantBlocks] = 0;
  }
  __syncthreads();

  for (int dim = tid; dim < kNopeDim; dim += blockDim.x) {
    const float inv_scale = exp2f(-scales[dim / kQuantBlock]);
    float scaled = values[dim] * inv_scale;
    scaled = fminf(fmaxf(scaled, -kFp8Max), kFp8Max);
    const __nv_fp8_storage_t storage =
        __nv_cvt_float_to_fp8(scaled, __NV_SATFINITE, __NV_E4M3);
    token_data[dim] = static_cast<uint8_t>(storage);
  }

  nv_bfloat16* rope_tail = reinterpret_cast<nv_bfloat16*>(token_data + kNopeDim);
  for (int dim = tid; dim < kRopeDim; dim += blockDim.x) {
    rope_tail[dim] = __float2bfloat16(rope_values[dim]);
  }
}

template <typename scalar_t>
__global__ void fused_qnorm_rope_kv_insert_kernel(
    scalar_t* __restrict__ q,
    const scalar_t* __restrict__ kv,
    uint8_t* __restrict__ k_cache,
    const int64_t* __restrict__ slot_mapping,
    const int64_t* __restrict__ positions,
    const float* __restrict__ cos_sin_cache,
    float rms_norm_eps,
    int num_tokens_full,
    int num_tokens_insert,
    int num_heads,
    int cache_block_size,
    int64_t cache_block_stride) {
  const int warp_id = threadIdx.x / kWarpSize;
  const int lane_id = threadIdx.x & (kWarpSize - 1);
  const int global_warp = blockIdx.x * kWarpsPerBlock + warp_id;
  const int slots_per_token = num_heads + 1;
  const int token_idx = global_warp / slots_per_token;
  const int task_idx = global_warp - token_idx * slots_per_token;

  if (token_idx >= num_tokens_full) {
    return;
  }

  const bool is_kv = task_idx == num_heads;
  if (is_kv && token_idx >= num_tokens_insert) {
    return;
  }
  if (task_idx > num_heads) {
    return;
  }

  const int dim_base = lane_id * kValuesPerLane;
  float values[kValuesPerLane];

  const scalar_t* src;
  if (is_kv) {
    src = kv + static_cast<int64_t>(token_idx) * kHeadDim + dim_base;
  } else {
    src = q + (static_cast<int64_t>(token_idx) * num_heads + task_idx) * kHeadDim +
          dim_base;
  }

#pragma unroll
  for (int i = 0; i < kValuesPerLane; ++i) {
    values[i] = scalar_to_float(src[i]);
  }

  if (!is_kv) {
    float local_sum = 0.0f;
#pragma unroll
    for (int i = 0; i < kValuesPerLane; ++i) {
      local_sum += values[i] * values[i];
    }
    const float rms_scale =
        rsqrtf(warp_sum_all(local_sum) / static_cast<float>(kHeadDim) +
               rms_norm_eps);
#pragma unroll
    for (int i = 0; i < kValuesPerLane; ++i) {
      values[i] *= rms_scale;
    }
  }

  const bool is_rope_lane = dim_base >= kNopeDim;
  if (is_rope_lane) {
    const int64_t position = positions[token_idx];
    const float* cos_ptr = cos_sin_cache + position * kRopeDim;
    const float* sin_ptr = cos_ptr + kHalfRopeDim;
    const int rope_base = dim_base - kNopeDim;

#pragma unroll
    for (int pair = 0; pair < kValuesPerLane / 2; ++pair) {
      const int half_idx = (rope_base + pair * 2) / 2;
      const float x_even = values[pair * 2];
      const float x_odd = values[pair * 2 + 1];
      const float cos_v = cos_ptr[half_idx];
      const float sin_v = sin_ptr[half_idx];
      values[pair * 2] = x_even * cos_v - x_odd * sin_v;
      values[pair * 2 + 1] = x_even * sin_v + x_odd * cos_v;
    }
  }

  if (!is_kv) {
    scalar_t* dst =
        q + (static_cast<int64_t>(token_idx) * num_heads + task_idx) * kHeadDim +
        dim_base;
#pragma unroll
    for (int i = 0; i < kValuesPerLane; ++i) {
      dst[i] = float_to_scalar<scalar_t>(values[i]);
    }
    return;
  }

  const int64_t slot = slot_mapping[token_idx];
  if (slot < 0) {
    return;
  }

  // Match vLLM's numeric contract: materialize K at activation dtype before
  // the UE8M0 absmax and final cache write.
#pragma unroll
  for (int i = 0; i < kValuesPerLane; ++i) {
    values[i] = scalar_to_float(float_to_scalar<scalar_t>(values[i]));
  }

  const int64_t block_idx = slot / cache_block_size;
  const int64_t pos_in_block = slot % cache_block_size;
  uint8_t* block_base = k_cache + block_idx * cache_block_stride;
  uint8_t* token_data = block_base + pos_in_block * kTokenDataBytes;
  uint8_t* token_scales =
      block_base + static_cast<int64_t>(cache_block_size) * kTokenDataBytes +
      pos_in_block * kScaleBytesPerToken;

  float local_absmax = 0.0f;
#pragma unroll
  for (int i = 0; i < kValuesPerLane; ++i) {
    local_absmax = fmaxf(local_absmax, fabsf(values[i]));
  }
  const float absmax = fmaxf(warp4_max_all(local_absmax), 1.0e-4f);
  const float exponent = ceilf(log2f(absmax / kFp8Max));
  const float inv_scale = exp2f(-exponent);

  if (!is_rope_lane) {
#pragma unroll
    for (int i = 0; i < kValuesPerLane; ++i) {
      float scaled = values[i] * inv_scale;
      scaled = fminf(fmaxf(scaled, -kFp8Max), kFp8Max);
      const __nv_fp8_storage_t storage =
          __nv_cvt_float_to_fp8(scaled, __NV_SATFINITE, __NV_E4M3);
      token_data[dim_base + i] = static_cast<uint8_t>(storage);
    }
    if ((lane_id & 3) == 0) {
      token_scales[lane_id >> 2] = encode_ue8m0_scale(exponent);
    }
    if (lane_id == 0) {
      token_scales[kNumQuantBlocks] = 0;
    }
  } else {
    scalar_t* rope_tail = reinterpret_cast<scalar_t*>(token_data + kNopeDim);
    const int rope_base = dim_base - kNopeDim;
#pragma unroll
    for (int i = 0; i < kValuesPerLane; ++i) {
      rope_tail[rope_base + i] = float_to_scalar<scalar_t>(values[i]);
    }
  }
}

template <typename scalar_t>
void launch_fused_qnorm_rope_kv_insert(
    scalar_t* q,
    const scalar_t* kv,
    uint8_t* k_cache,
    const int64_t* slot_mapping,
    const int64_t* positions,
    const float* cos_sin_cache,
    float rms_norm_eps,
    int num_tokens_full,
    int num_tokens_insert,
    int num_heads,
    int cache_block_size,
    int64_t cache_block_stride,
    cudaStream_t stream) {
  const int64_t total_slots =
      static_cast<int64_t>(num_tokens_full) * (num_heads + 1);
  const dim3 grid(static_cast<unsigned int>(
      (total_slots + kWarpsPerBlock - 1) / kWarpsPerBlock));
  fused_qnorm_rope_kv_insert_kernel<scalar_t><<<grid, kThreads, 0, stream>>>(
      q, kv, k_cache, slot_mapping, positions, cos_sin_cache, rms_norm_eps,
      num_tokens_full, num_tokens_insert, num_heads, cache_block_size,
      cache_block_stride);
}

__global__ void save_compressor_state_kernel(
    const float* __restrict__ kv,
    const float* __restrict__ score,
    const float* __restrict__ ape,
    float* __restrict__ state_cache,
    const int64_t* __restrict__ slot_mapping,
    const int64_t* __restrict__ positions,
    int num_tokens,
    int state_width,
    int block_size,
    int64_t state_cache_block_stride,
    int64_t state_cache_row_stride,
    int compress_ratio) {
  const int token_idx = blockIdx.x;
  const int tid = threadIdx.x;
  if (token_idx >= num_tokens) {
    return;
  }

  const int64_t slot = slot_mapping[token_idx];
  if (slot < 0) {
    return;
  }

  const int64_t block_idx = slot / block_size;
  const int64_t pos_in_block = slot % block_size;
  const int64_t position = positions[token_idx];
  const int slot_idx = static_cast<int>(position % compress_ratio);
  float* row = state_cache + block_idx * state_cache_block_stride +
               pos_in_block * state_cache_row_stride;
  const float* kv_row = kv + static_cast<int64_t>(token_idx) * state_width;
  const float* score_row = score + static_cast<int64_t>(token_idx) * state_width;

  for (int dim = tid; dim < state_width; dim += blockDim.x) {
    row[dim] = kv_row[dim];

    float ape_value;
    if (compress_ratio == kCsaCompressRatio && (state_width % 2) == 0) {
      const int half_width = state_width / 2;
      const int ape_row = dim < half_width ? slot_idx
                                           : slot_idx + kCsaCompressRatio;
      const int ape_col = dim < half_width ? dim : dim - half_width;
      ape_value = ape[static_cast<int64_t>(ape_row) * half_width + ape_col];
    } else {
      ape_value = ape[static_cast<int64_t>(slot_idx) * state_width + dim];
    }
    row[state_width + dim] = score_row[dim] + ape_value;
  }
}

template <typename scalar_t>
__global__ void sparse_mla_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ kv,
    const int32_t* __restrict__ indices,
    const int64_t* __restrict__ topk_length,
    const float* __restrict__ attn_sink,
    scalar_t* __restrict__ output,
    float sm_scale,
    int num_tokens,
    int num_heads,
    int topk_width,
    int attn_sink_count) {
  const int token_idx = blockIdx.x;
  const int head_idx = blockIdx.y;
  const int tid = threadIdx.x;
  if (token_idx >= num_tokens || head_idx >= num_heads) {
    return;
  }

  __shared__ float reduction[kThreads];
  __shared__ float shared_score;
  __shared__ float shared_max_score;
  __shared__ float shared_weight;
  __shared__ float shared_denom;

  const scalar_t* q_ptr =
      q + (static_cast<int64_t>(token_idx) * num_heads + head_idx) * kHeadDim;
  scalar_t* out_ptr =
      output + (static_cast<int64_t>(token_idx) * num_heads + head_idx) * kHeadDim;
  const int64_t raw_length = topk_length[token_idx];
  const int length =
      raw_length <= 0 ? 0 : (raw_length > topk_width ? topk_width : static_cast<int>(raw_length));
  const bool has_sink = attn_sink_count > head_idx;
  float q_values[kValuesPerThread];
  #pragma unroll
  for (int lane = 0; lane < kValuesPerThread; ++lane) {
    const int dim = tid + lane * blockDim.x;
    q_values[lane] = dim < kHeadDim ? scalar_to_float(q_ptr[dim]) : 0.0f;
  }

  if (tid == 0) {
    shared_max_score = has_sink ? attn_sink[head_idx] : kNegInf;
  }
  __syncthreads();

  for (int candidate = 0; candidate < length; ++candidate) {
    const int32_t kv_idx = indices[token_idx * topk_width + candidate];
    if (kv_idx < 0) {
      continue;
    }
    const scalar_t* kv_ptr = kv + static_cast<int64_t>(kv_idx) * kHeadDim;
    float local = 0.0f;
    #pragma unroll
    for (int lane = 0; lane < kValuesPerThread; ++lane) {
      const int dim = tid + lane * blockDim.x;
      if (dim < kHeadDim) {
        local += q_values[lane] * scalar_to_float(kv_ptr[dim]);
      }
    }
    const float score_sum = block_sum_256(local, reduction);
    if (tid == 0) {
      shared_score = score_sum * sm_scale;
      shared_max_score = fmaxf(shared_max_score, shared_score);
    }
    __syncthreads();
  }

  const float max_score = shared_max_score;
  if (tid == 0) {
    shared_denom = has_sink ? expf(attn_sink[head_idx] - max_score) : 0.0f;
    if (!isfinite(shared_denom)) {
      shared_denom = 0.0f;
    }
  }
  __syncthreads();

  float accum_values[kValuesPerThread] = {0.0f};
  for (int candidate = 0; candidate < length; ++candidate) {
    const int32_t kv_idx = indices[token_idx * topk_width + candidate];
    if (kv_idx < 0) {
      continue;
    }
    const scalar_t* kv_ptr = kv + static_cast<int64_t>(kv_idx) * kHeadDim;
    float local = 0.0f;
    float kv_values[kValuesPerThread];
    #pragma unroll
    for (int lane = 0; lane < kValuesPerThread; ++lane) {
      const int dim = tid + lane * blockDim.x;
      if (dim < kHeadDim) {
        const float kv_value = scalar_to_float(kv_ptr[dim]);
        kv_values[lane] = kv_value;
        local += q_values[lane] * kv_value;
      } else {
        kv_values[lane] = 0.0f;
      }
    }
    const float score_sum = block_sum_256(local, reduction);
    if (tid == 0) {
      shared_weight = expf(score_sum * sm_scale - max_score);
      if (!isfinite(shared_weight)) {
        shared_weight = 0.0f;
      }
      shared_denom += shared_weight;
    }
    __syncthreads();
    const float weight = shared_weight;
    #pragma unroll
    for (int lane = 0; lane < kValuesPerThread; ++lane) {
      accum_values[lane] += weight * kv_values[lane];
    }
    __syncthreads();
  }

  const float denom = shared_denom;
  int lane = 0;
  for (int dim = tid; dim < kHeadDim; dim += blockDim.x, ++lane) {
    const float value = denom > 0.0f ? accum_values[lane] / denom : 0.0f;
    out_ptr[dim] = float_to_scalar<scalar_t>(value);
  }
}

template <typename scalar_t>
__global__ void sparse_mla_fp8_cache_kernel(
    const scalar_t* __restrict__ q,
    const uint8_t* __restrict__ swa_cache,
    const int32_t* __restrict__ swa_indices,
    const int64_t* __restrict__ swa_lens,
    const uint8_t* __restrict__ compressed_cache,
    const int32_t* __restrict__ extra_indices,
    const int64_t* __restrict__ extra_lens,
    const float* __restrict__ attn_sink,
    scalar_t* __restrict__ output,
    float sm_scale,
    int num_tokens,
    int num_heads,
    int swa_width,
    int extra_width,
    int swa_block_size,
    int compressed_block_size,
    int64_t swa_cache_block_stride,
    int64_t compressed_cache_block_stride,
    int attn_sink_count) {
  const int token_idx = blockIdx.x;
  const int head_idx = blockIdx.y;
  const int tid = threadIdx.x;
  if (token_idx >= num_tokens || head_idx >= num_heads) {
    return;
  }

  __shared__ float reduction[kWarpsPerBlock];
  __shared__ float shared_score;
  __shared__ float shared_max_score;
  __shared__ float shared_weight;
  __shared__ float shared_denom;
  __shared__ float shared_cache_scales[kNumQuantBlocks];

  const scalar_t* q_ptr =
      q + (static_cast<int64_t>(token_idx) * num_heads + head_idx) * kHeadDim;
  scalar_t* out_ptr =
      output + (static_cast<int64_t>(token_idx) * num_heads + head_idx) * kHeadDim;
  const int64_t raw_swa_len = swa_lens[token_idx];
  const int64_t raw_extra_len = extra_lens[token_idx];
  const int swa_len = raw_swa_len <= 0
                          ? 0
                          : (raw_swa_len > swa_width ? swa_width
                                                     : static_cast<int>(raw_swa_len));
  const int extra_len =
      raw_extra_len <= 0
          ? 0
          : (raw_extra_len > extra_width ? extra_width
                                         : static_cast<int>(raw_extra_len));
  const int total_len = extra_len + swa_len;
  const bool has_sink = attn_sink_count > head_idx;
  float q_values[kValuesPerThread];
  #pragma unroll
  for (int lane = 0; lane < kValuesPerThread; ++lane) {
    const int dim = tid + lane * blockDim.x;
    q_values[lane] = dim < kHeadDim ? scalar_to_float(q_ptr[dim]) : 0.0f;
  }

  if (tid == 0) {
    shared_max_score = has_sink ? attn_sink[head_idx] : kNegInf;
  }
  __syncthreads();

  for (int candidate = 0; candidate < total_len; ++candidate) {
    const bool use_extra = candidate < extra_len;
    const int candidate_offset = use_extra ? candidate : candidate - extra_len;
    const int32_t slot =
        use_extra
            ? extra_indices[token_idx * extra_width + candidate_offset]
            : swa_indices[token_idx * swa_width + candidate_offset];
    if (slot < 0) {
      continue;
    }
    const uint8_t* cache = use_extra ? compressed_cache : swa_cache;
    const int cache_block_size =
        use_extra ? compressed_block_size : swa_block_size;
    const int64_t cache_block_stride =
        use_extra ? compressed_cache_block_stride : swa_cache_block_stride;
    const int64_t block_idx = slot / cache_block_size;
    const int64_t pos_in_block = slot % cache_block_size;
    const uint8_t* block_base = cache + block_idx * cache_block_stride;
    const uint8_t* token_data = block_base + pos_in_block * kTokenDataBytes;
    const uint8_t* token_scales =
        block_base + static_cast<int64_t>(cache_block_size) * kTokenDataBytes +
        pos_in_block * kScaleBytesPerToken;
    if (tid < kNumQuantBlocks) {
      shared_cache_scales[tid] =
          exp2f(static_cast<int>(token_scales[tid]) - 127);
    }
    __syncthreads();
    float local = 0.0f;
    #pragma unroll
    for (int lane = 0; lane < kValuesPerThread; ++lane) {
      const int dim = tid + lane * blockDim.x;
      const float kv_value = load_fp8_ds_mla_cache_value_cached<scalar_t>(
          token_data, shared_cache_scales, dim);
      local += q_values[lane] * kv_value;
    }
    const float score_sum = block_sum_256(local, reduction);
    if (tid == 0) {
      shared_score = score_sum * sm_scale;
      shared_max_score = fmaxf(shared_max_score, shared_score);
    }
    __syncthreads();
  }

  const float max_score = shared_max_score;
  if (tid == 0) {
    shared_denom = has_sink ? expf(attn_sink[head_idx] - max_score) : 0.0f;
    if (!isfinite(shared_denom)) {
      shared_denom = 0.0f;
    }
  }
  __syncthreads();

  float accum_values[kValuesPerThread] = {0.0f};
  for (int candidate = 0; candidate < total_len; ++candidate) {
    const bool use_extra = candidate < extra_len;
    const int candidate_offset = use_extra ? candidate : candidate - extra_len;
    const int32_t slot =
        use_extra
            ? extra_indices[token_idx * extra_width + candidate_offset]
            : swa_indices[token_idx * swa_width + candidate_offset];
    if (slot < 0) {
      continue;
    }
    const uint8_t* cache = use_extra ? compressed_cache : swa_cache;
    const int cache_block_size =
        use_extra ? compressed_block_size : swa_block_size;
    const int64_t cache_block_stride =
        use_extra ? compressed_cache_block_stride : swa_cache_block_stride;
    const int64_t block_idx = slot / cache_block_size;
    const int64_t pos_in_block = slot % cache_block_size;
    const uint8_t* block_base = cache + block_idx * cache_block_stride;
    const uint8_t* token_data = block_base + pos_in_block * kTokenDataBytes;
    const uint8_t* token_scales =
        block_base + static_cast<int64_t>(cache_block_size) * kTokenDataBytes +
        pos_in_block * kScaleBytesPerToken;
    if (tid < kNumQuantBlocks) {
      shared_cache_scales[tid] =
          exp2f(static_cast<int>(token_scales[tid]) - 127);
    }
    __syncthreads();
    float local = 0.0f;
    float kv_values[kValuesPerThread];
    #pragma unroll
    for (int lane = 0; lane < kValuesPerThread; ++lane) {
      const int dim = tid + lane * blockDim.x;
      const float kv_value = load_fp8_ds_mla_cache_value_cached<scalar_t>(
          token_data, shared_cache_scales, dim);
      kv_values[lane] = kv_value;
      local += q_values[lane] * kv_value;
    }
    const float score_sum = block_sum_256(local, reduction);
    if (tid == 0) {
      shared_weight = expf(score_sum * sm_scale - max_score);
      if (!isfinite(shared_weight)) {
        shared_weight = 0.0f;
      }
      shared_denom += shared_weight;
    }
    __syncthreads();
    const float weight = shared_weight;
    #pragma unroll
    for (int lane = 0; lane < kValuesPerThread; ++lane) {
      accum_values[lane] += weight * kv_values[lane];
    }
    __syncthreads();
  }

  const float denom = shared_denom;
  int lane = 0;
  for (int dim = tid; dim < kHeadDim; dim += blockDim.x, ++lane) {
    const float value = denom > 0.0f ? accum_values[lane] / denom : 0.0f;
    out_ptr[dim] = float_to_scalar<scalar_t>(value);
  }
}

template <typename scalar_t>
__global__ void sparse_mla_fp8_cache_online_softmax_kernel(
    const scalar_t* __restrict__ q,
    const uint8_t* __restrict__ swa_cache,
    const int32_t* __restrict__ swa_indices,
    const int64_t* __restrict__ swa_lens,
    const uint8_t* __restrict__ compressed_cache,
    const int32_t* __restrict__ extra_indices,
    const int64_t* __restrict__ extra_lens,
    const float* __restrict__ attn_sink,
    scalar_t* __restrict__ output,
    float sm_scale,
    int num_tokens,
    int num_heads,
    int swa_width,
    int extra_width,
    int swa_block_size,
    int compressed_block_size,
    int64_t swa_cache_block_stride,
    int64_t compressed_cache_block_stride,
    int attn_sink_count) {
  const int token_idx = blockIdx.x;
  const int head_idx = blockIdx.y;
  const int tid = threadIdx.x;
  if (token_idx >= num_tokens || head_idx >= num_heads) {
    return;
  }

  __shared__ float reduction[kWarpsPerBlock];
  __shared__ float shared_max_score;
  __shared__ float shared_weight;
  __shared__ float shared_rescale;
  __shared__ float shared_denom;
  __shared__ float shared_cache_scales[kNumQuantBlocks];

  const scalar_t* q_ptr =
      q + (static_cast<int64_t>(token_idx) * num_heads + head_idx) * kHeadDim;
  scalar_t* out_ptr =
      output + (static_cast<int64_t>(token_idx) * num_heads + head_idx) * kHeadDim;
  const int64_t raw_swa_len = swa_lens[token_idx];
  const int64_t raw_extra_len = extra_lens[token_idx];
  const int swa_len = raw_swa_len <= 0
                          ? 0
                          : (raw_swa_len > swa_width ? swa_width
                                                     : static_cast<int>(raw_swa_len));
  const int extra_len =
      raw_extra_len <= 0
          ? 0
          : (raw_extra_len > extra_width ? extra_width
                                         : static_cast<int>(raw_extra_len));
  const int total_len = extra_len + swa_len;
  const bool has_finite_sink =
      attn_sink_count > head_idx && isfinite(attn_sink[head_idx]);

  float q_values[kValuesPerThread];
#pragma unroll
  for (int lane = 0; lane < kValuesPerThread; ++lane) {
    const int dim = tid + lane * blockDim.x;
    q_values[lane] = dim < kHeadDim ? scalar_to_float(q_ptr[dim]) : 0.0f;
  }

  if (tid == 0) {
    shared_max_score = has_finite_sink ? attn_sink[head_idx] : kNegInf;
    shared_denom = has_finite_sink ? 1.0f : 0.0f;
  }
  __syncthreads();

  float accum_values[kValuesPerThread] = {0.0f};
  for (int candidate = 0; candidate < total_len; ++candidate) {
    const bool use_extra = candidate < extra_len;
    const int candidate_offset = use_extra ? candidate : candidate - extra_len;
    const int32_t slot =
        use_extra
            ? extra_indices[token_idx * extra_width + candidate_offset]
            : swa_indices[token_idx * swa_width + candidate_offset];
    if (slot < 0) {
      continue;
    }
    const uint8_t* cache = use_extra ? compressed_cache : swa_cache;
    const int cache_block_size =
        use_extra ? compressed_block_size : swa_block_size;
    const int64_t cache_block_stride =
        use_extra ? compressed_cache_block_stride : swa_cache_block_stride;
    const int64_t block_idx = slot / cache_block_size;
    const int64_t pos_in_block = slot % cache_block_size;
    const uint8_t* block_base = cache + block_idx * cache_block_stride;
    const uint8_t* token_data = block_base + pos_in_block * kTokenDataBytes;
    const uint8_t* token_scales =
        block_base + static_cast<int64_t>(cache_block_size) * kTokenDataBytes +
        pos_in_block * kScaleBytesPerToken;
    if (tid < kNumQuantBlocks) {
      shared_cache_scales[tid] =
          exp2f(static_cast<int>(token_scales[tid]) - 127);
    }
    __syncthreads();

    float local = 0.0f;
    float kv_values[kValuesPerThread];
#pragma unroll
    for (int lane = 0; lane < kValuesPerThread; ++lane) {
      const int dim = tid + lane * blockDim.x;
      const float kv_value = load_fp8_ds_mla_cache_value_cached<scalar_t>(
          token_data, shared_cache_scales, dim);
      kv_values[lane] = kv_value;
      local += q_values[lane] * kv_value;
    }

    const float score_sum = block_sum_256(local, reduction);
    if (tid == 0) {
      const float score = score_sum * sm_scale;
      if (isfinite(score)) {
        const float old_max_score = shared_max_score;
        const float new_max_score = fmaxf(old_max_score, score);
        const float rescale =
            isfinite(old_max_score) ? expf(old_max_score - new_max_score) : 0.0f;
        float weight = expf(score - new_max_score);
        if (!isfinite(weight)) {
          weight = 0.0f;
        }
        shared_denom = shared_denom * rescale + weight;
        shared_max_score = new_max_score;
        shared_rescale = rescale;
        shared_weight = weight;
      } else {
        shared_rescale = 1.0f;
        shared_weight = 0.0f;
      }
    }
    __syncthreads();

    const float rescale = shared_rescale;
    const float weight = shared_weight;
#pragma unroll
    for (int lane = 0; lane < kValuesPerThread; ++lane) {
      accum_values[lane] = accum_values[lane] * rescale + weight * kv_values[lane];
    }
    __syncthreads();
  }

  const float denom = shared_denom;
  int lane = 0;
  for (int dim = tid; dim < kHeadDim; dim += blockDim.x, ++lane) {
    const float value = denom > 0.0f ? accum_values[lane] / denom : 0.0f;
    out_ptr[dim] = float_to_scalar<scalar_t>(value);
  }
}

// K-split variant of the online-softmax kernel.
//
// Each block processes one chunk of `total_len` candidates for one
// (token, head) pair and writes its partial state -- ``(max, denom,
// acc[head_dim])`` -- to a scratch buffer. The attention sink is NOT
// applied here so the partials can be merged with the sink in one
// place (see ``sparse_mla_fp8_cache_partials_reduce_kernel``). The
// chunk boundary is purely on the candidate index axis, so each chunk
// still walks both extra and SWA candidates inside its slice.
//
// Grid: ``(num_tokens, num_heads, k_split)``.
// T2-α-2 Path B: per-kind specialised K-split chunk kernel.
//
// ``kSwaOnly = true`` collapses the chunk loop to the SWA-only path:
// constant-folds ``use_extra = false`` everywhere, skips the
// ``extra_lens[token_idx]`` gmem read, and lets the compiler drop the
// dead compressed-cache pointer arithmetic. DSv4-Flash layers with
// ``compress_ratio <= 1`` (3 of 44 layers, see plan-doc layer mix)
// route here under ``auto``; everything else uses the mixed
// ``kSwaOnly = false`` instantiation that is identical to the previous
// single-template kernel.
//
// The speedup is bounded by 3/44 layers being SWA-only; the win shows
// up as a tiny per-decode-step bookkeeping reduction, not a kernel-
// time blowout. The dispatch path is the structural win: future
// hca / csa specialisations (W2 fuse, tile-shape tuning) get to slot
// into the same dispatch.
template <typename scalar_t, bool kSwaOnly>
__global__ void sparse_mla_fp8_cache_online_softmax_k_split_kernel_impl(
    const scalar_t* __restrict__ q,
    const uint8_t* __restrict__ swa_cache,
    const int32_t* __restrict__ swa_indices,
    const int64_t* __restrict__ swa_lens,
    const uint8_t* __restrict__ compressed_cache,
    const int32_t* __restrict__ extra_indices,
    const int64_t* __restrict__ extra_lens,
    float* __restrict__ partials,
    float sm_scale,
    int num_tokens,
    int num_heads,
    int swa_width,
    int extra_width,
    int swa_block_size,
    int compressed_block_size,
    int64_t swa_cache_block_stride,
    int64_t compressed_cache_block_stride,
    int k_split,
    int64_t partials_stride) {
  const int token_idx = blockIdx.x;
  const int head_idx = blockIdx.y;
  const int k_idx = blockIdx.z;
  const int tid = threadIdx.x;
  if (token_idx >= num_tokens || head_idx >= num_heads || k_idx >= k_split) {
    return;
  }

  __shared__ float reduction[kWarpsPerBlock];
  __shared__ float shared_max_score;
  __shared__ float shared_weight;
  __shared__ float shared_rescale;
  __shared__ float shared_denom;
  __shared__ float shared_cache_scales[kNumQuantBlocks];

  const scalar_t* q_ptr =
      q + (static_cast<int64_t>(token_idx) * num_heads + head_idx) * kHeadDim;

  const int64_t raw_swa_len = swa_lens[token_idx];
  const int swa_len = raw_swa_len <= 0
                          ? 0
                          : (raw_swa_len > swa_width ? swa_width
                                                     : static_cast<int>(raw_swa_len));
  int extra_len;
  if constexpr (kSwaOnly) {
    // SWA-only: ``extra_lens`` and ``compressed_cache`` are unused; the
    // caller is contracted to pass ``extra_width == 0``.
    extra_len = 0;
  } else {
    const int64_t raw_extra_len = extra_lens[token_idx];
    extra_len =
        raw_extra_len <= 0
            ? 0
            : (raw_extra_len > extra_width ? extra_width
                                           : static_cast<int>(raw_extra_len));
  }
  const int total_len = extra_len + swa_len;
  // Contiguous chunk on the candidate axis. Empty chunks (chunk_start ==
  // chunk_end) are valid -- they write (m=-inf, d=0, acc=0).
  const int chunk_start = static_cast<int>(
      (static_cast<int64_t>(k_idx) * total_len) / k_split);
  const int chunk_end = static_cast<int>(
      (static_cast<int64_t>(k_idx + 1) * total_len) / k_split);

  float q_values[kValuesPerThread];
#pragma unroll
  for (int lane = 0; lane < kValuesPerThread; ++lane) {
    const int dim = tid + lane * blockDim.x;
    q_values[lane] = dim < kHeadDim ? scalar_to_float(q_ptr[dim]) : 0.0f;
  }

  if (tid == 0) {
    shared_max_score = kNegInf;
    shared_denom = 0.0f;
  }
  __syncthreads();

  float accum_values[kValuesPerThread] = {0.0f};
  for (int candidate = chunk_start; candidate < chunk_end; ++candidate) {
    int32_t slot;
    const uint8_t* cache;
    int cache_block_size;
    int64_t cache_block_stride;
    if constexpr (kSwaOnly) {
      slot = swa_indices[token_idx * swa_width + candidate];
      cache = swa_cache;
      cache_block_size = swa_block_size;
      cache_block_stride = swa_cache_block_stride;
    } else {
      const bool use_extra = candidate < extra_len;
      const int candidate_offset =
          use_extra ? candidate : candidate - extra_len;
      slot = use_extra
                 ? extra_indices[token_idx * extra_width + candidate_offset]
                 : swa_indices[token_idx * swa_width + candidate_offset];
      cache = use_extra ? compressed_cache : swa_cache;
      cache_block_size = use_extra ? compressed_block_size : swa_block_size;
      cache_block_stride =
          use_extra ? compressed_cache_block_stride : swa_cache_block_stride;
    }
    if (slot < 0) {
      continue;
    }
    const int64_t block_idx = slot / cache_block_size;
    const int64_t pos_in_block = slot % cache_block_size;
    const uint8_t* block_base = cache + block_idx * cache_block_stride;
    const uint8_t* token_data = block_base + pos_in_block * kTokenDataBytes;
    const uint8_t* token_scales =
        block_base + static_cast<int64_t>(cache_block_size) * kTokenDataBytes +
        pos_in_block * kScaleBytesPerToken;
    if (tid < kNumQuantBlocks) {
      shared_cache_scales[tid] =
          exp2f(static_cast<int>(token_scales[tid]) - 127);
    }
    __syncthreads();

    float local = 0.0f;
    float kv_values[kValuesPerThread];
#pragma unroll
    for (int lane = 0; lane < kValuesPerThread; ++lane) {
      const int dim = tid + lane * blockDim.x;
      const float kv_value = load_fp8_ds_mla_cache_value_cached<scalar_t>(
          token_data, shared_cache_scales, dim);
      kv_values[lane] = kv_value;
      local += q_values[lane] * kv_value;
    }

    const float score_sum = block_sum_256(local, reduction);
    if (tid == 0) {
      const float score = score_sum * sm_scale;
      if (isfinite(score)) {
        const float old_max_score = shared_max_score;
        const float new_max_score = fmaxf(old_max_score, score);
        const float rescale =
            isfinite(old_max_score) ? expf(old_max_score - new_max_score) : 0.0f;
        float weight = expf(score - new_max_score);
        if (!isfinite(weight)) {
          weight = 0.0f;
        }
        shared_denom = shared_denom * rescale + weight;
        shared_max_score = new_max_score;
        shared_rescale = rescale;
        shared_weight = weight;
      } else {
        shared_rescale = 1.0f;
        shared_weight = 0.0f;
      }
    }
    __syncthreads();

    const float rescale = shared_rescale;
    const float weight = shared_weight;
#pragma unroll
    for (int lane = 0; lane < kValuesPerThread; ++lane) {
      accum_values[lane] = accum_values[lane] * rescale + weight * kv_values[lane];
    }
    __syncthreads();
  }

  // Write partial (max, denom, acc) to scratch.
  //
  // Layout: ``[num_tokens, num_heads, k_split, head_dim + 2]``. The first
  // two floats per (token, head, k_idx) slot hold ``max`` and ``denom``;
  // the next ``head_dim`` floats hold ``acc``. ``partials_stride`` is the
  // stride (in floats) between one (token, head, k_idx) slot and the next
  // along the k axis -- the launcher fills it in as ``head_dim + 2``.
  float* partial_base =
      partials +
      ((static_cast<int64_t>(token_idx) * num_heads + head_idx) * k_split +
       k_idx) *
          partials_stride;
  if (tid == 0) {
    partial_base[0] = shared_max_score;
    partial_base[1] = shared_denom;
  }
  int acc_lane = 0;
  for (int dim = tid; dim < kHeadDim; dim += blockDim.x, ++acc_lane) {
    partial_base[2 + dim] = accum_values[acc_lane];
  }
}

// Reduce K_SPLIT partial (max, denom, acc) slots per (token, head) into a
// single final output, folding in the attention sink as a "phantom" partial
// with score=sink, denom=1, acc=0 to match the single-block kernel above.
//
// Grid: ``(num_tokens, num_heads)``. Block: ``kThreads``.
template <typename scalar_t>
__global__ void sparse_mla_fp8_cache_partials_reduce_kernel(
    const float* __restrict__ partials,
    const float* __restrict__ attn_sink,
    scalar_t* __restrict__ output,
    int num_tokens,
    int num_heads,
    int k_split,
    int64_t partials_stride,
    int attn_sink_count) {
  const int token_idx = blockIdx.x;
  const int head_idx = blockIdx.y;
  const int tid = threadIdx.x;
  if (token_idx >= num_tokens || head_idx >= num_heads) {
    return;
  }

  scalar_t* out_ptr =
      output + (static_cast<int64_t>(token_idx) * num_heads + head_idx) * kHeadDim;
  const float* base_for_pair =
      partials +
      (static_cast<int64_t>(token_idx) * num_heads + head_idx) * k_split *
          partials_stride;

  // Step 1: find global max across K partials + finite sink (if any).
  // Only thread 0 owns this in shared memory; broadcast to the warp.
  __shared__ float shared_global_max;
  __shared__ float shared_total_denom;
  __shared__ float shared_rescales[16];  // enough for K_SPLIT up to 16
  if (tid == 0) {
    float global_max = kNegInf;
    for (int k = 0; k < k_split; ++k) {
      const float m_k = base_for_pair[k * partials_stride];
      const float d_k = base_for_pair[k * partials_stride + 1];
      // Only consider partials that actually saw a candidate.
      if (d_k > 0.0f && isfinite(m_k)) {
        global_max = fmaxf(global_max, m_k);
      }
    }
    const bool has_finite_sink =
        attn_sink_count > head_idx && isfinite(attn_sink[head_idx]);
    if (has_finite_sink) {
      global_max = fmaxf(global_max, attn_sink[head_idx]);
    }

    float total_denom = 0.0f;
    for (int k = 0; k < k_split; ++k) {
      const float m_k = base_for_pair[k * partials_stride];
      const float d_k = base_for_pair[k * partials_stride + 1];
      float rescale = 0.0f;
      if (d_k > 0.0f && isfinite(m_k) && isfinite(global_max)) {
        rescale = expf(m_k - global_max);
        if (!isfinite(rescale)) {
          rescale = 0.0f;
        }
      }
      shared_rescales[k] = rescale;
      total_denom += d_k * rescale;
    }
    if (has_finite_sink) {
      total_denom += expf(attn_sink[head_idx] - global_max);
    }
    shared_global_max = global_max;
    shared_total_denom = total_denom;
  }
  __syncthreads();

  const float total_denom = shared_total_denom;
  // Step 2: per-element accumulation.
  for (int dim = tid; dim < kHeadDim; dim += blockDim.x) {
    float numerator = 0.0f;
    for (int k = 0; k < k_split; ++k) {
      numerator +=
          shared_rescales[k] * base_for_pair[k * partials_stride + 2 + dim];
    }
    const float value = total_denom > 0.0f ? numerator / total_denom : 0.0f;
    out_ptr[dim] = float_to_scalar<scalar_t>(value);
  }
}

__global__ void decode_indices_kernel(
    const int64_t* __restrict__ positions,
    const int32_t* __restrict__ token_to_req_indices,
    const int32_t* __restrict__ block_table,
    const int32_t* __restrict__ topk_indices,
    int32_t* __restrict__ swa_indices,
    int32_t* __restrict__ swa_lens,
    int32_t* __restrict__ extra_indices,
    int32_t* __restrict__ extra_lens,
    int num_tokens,
    int block_table_width,
    int window_size,
    int swa_width,
    int swa_block_size,
    int compress_ratio,
    int compressed_width,
    int compressed_block_size,
    int topk_width,
    int full_candidate_max_len) {
  const int token_idx = blockIdx.x;
  const int tid = threadIdx.x;
  if (token_idx >= num_tokens) {
    return;
  }

  const int64_t position = positions[token_idx];
  const int32_t req_idx = token_to_req_indices[token_idx];
  const int64_t start = max(
      static_cast<int64_t>(0),
      position - static_cast<int64_t>(window_size) + 1);
  const int swa_len = static_cast<int>(
      min(static_cast<int64_t>(window_size), position - start + 1));

  for (int offset = tid; offset < swa_width; offset += blockDim.x) {
    int32_t slot = -1;
    if (req_idx >= 0 && offset < swa_len) {
      const int64_t local = start + offset;
      const int64_t page_idx = local / swa_block_size;
      if (page_idx >= 0 && page_idx < block_table_width) {
        const int32_t page =
            block_table[static_cast<int64_t>(req_idx) * block_table_width +
                        page_idx];
        slot = page * swa_block_size +
               static_cast<int32_t>(local % swa_block_size);
      }
    }
    swa_indices[static_cast<int64_t>(token_idx) * swa_width + offset] = slot;
  }
  if (tid == 0) {
    swa_lens[token_idx] = req_idx >= 0 ? swa_len : 0;
  }

  for (int offset = tid; offset < compressed_width; offset += blockDim.x) {
    extra_indices[static_cast<int64_t>(token_idx) * compressed_width + offset] =
        -1;
  }
  if (tid == 0) {
    int count = 0;
    if (req_idx >= 0 && compress_ratio == kCsaCompressRatio &&
        topk_width == 0 && full_candidate_max_len >= 0) {
      const int64_t candidate_count =
          min((position + 1) / compress_ratio,
              static_cast<int64_t>(compressed_width));
      for (int64_t local = 0; local < candidate_count; ++local) {
        const int64_t page_idx = local / compressed_block_size;
        if (page_idx < 0 || page_idx >= block_table_width) {
          continue;
        }
        const int32_t page =
            block_table[static_cast<int64_t>(req_idx) * block_table_width +
                        page_idx];
        extra_indices[static_cast<int64_t>(token_idx) * compressed_width +
                      count] =
            page * compressed_block_size + (local % compressed_block_size);
        ++count;
      }
    } else if (
        req_idx >= 0 && compress_ratio > 1 &&
        compress_ratio != kCsaCompressRatio && topk_width == 0 &&
        full_candidate_max_len >= 0) {
      const int64_t candidate_count =
          min((position + 1) / compress_ratio,
              static_cast<int64_t>(compressed_width));
      for (int64_t local = 0; local < candidate_count; ++local) {
        const int64_t page_idx = local / compressed_block_size;
        if (page_idx < 0 || page_idx >= block_table_width) {
          continue;
        }
        const int32_t page =
            block_table[static_cast<int64_t>(req_idx) * block_table_width +
                        page_idx];
        extra_indices[static_cast<int64_t>(token_idx) * compressed_width +
                      count] =
            page * compressed_block_size + (local % compressed_block_size);
        ++count;
      }
    } else if (
        req_idx >= 0 && compress_ratio == kCsaCompressRatio && topk_width > 0) {
      for (int rank = 0; rank < topk_width && count < compressed_width; ++rank) {
        const int32_t local =
            topk_indices[static_cast<int64_t>(token_idx) * topk_width + rank];
        if (local < 0) {
          continue;
        }
        const int64_t page_idx = local / compressed_block_size;
        if (page_idx < 0 || page_idx >= block_table_width) {
          continue;
        }
        const int32_t page =
            block_table[static_cast<int64_t>(req_idx) * block_table_width +
                        page_idx];
        extra_indices[static_cast<int64_t>(token_idx) * compressed_width +
                      count] =
            page * compressed_block_size + (local % compressed_block_size);
        ++count;
      }
    }
    extra_lens[token_idx] = count;
  }
}

__global__ void full_candidate_topk_kernel(
    const int64_t* __restrict__ positions,
    int32_t* __restrict__ topk,
    int num_tokens,
    int width,
    int compress_ratio) {
  const int token_idx = blockIdx.x;
  const int tid = threadIdx.x;
  if (token_idx >= num_tokens) {
    return;
  }
  const int64_t length = (positions[token_idx] + 1) / compress_ratio;
  for (int offset = tid; offset < width; offset += blockDim.x) {
    topk[static_cast<int64_t>(token_idx) * width + offset] =
        offset < length ? offset : -1;
  }
}

template <typename scalar_t>
void launch_sparse_mla(
    const scalar_t* q,
    const scalar_t* kv,
    const int32_t* indices,
    const int64_t* topk_length,
    const float* attn_sink,
    scalar_t* output,
    float sm_scale,
    int num_tokens,
    int num_heads,
    int topk_width,
    int attn_sink_count,
    cudaStream_t stream) {
  cudaMemsetAsync(
      output,
      0,
      static_cast<size_t>(num_tokens) * num_heads * kHeadDim * sizeof(scalar_t),
      stream);
  const dim3 grid(num_tokens, num_heads);
  sparse_mla_kernel<scalar_t><<<grid, kThreads, 0, stream>>>(
      q, kv, indices, topk_length, attn_sink, output, sm_scale, num_tokens,
      num_heads, topk_width, attn_sink_count);
}

template <typename scalar_t>
void launch_sparse_mla_fp8_cache(
    const scalar_t* q,
    const uint8_t* swa_cache,
    const int32_t* swa_indices,
    const int64_t* swa_lens,
    const uint8_t* compressed_cache,
    const int32_t* extra_indices,
    const int64_t* extra_lens,
    const float* attn_sink,
    scalar_t* output,
    float* partials,
    float sm_scale,
    int num_tokens,
    int num_heads,
    int swa_width,
    int extra_width,
    int swa_block_size,
    int compressed_block_size,
    int64_t swa_cache_block_stride,
    int64_t compressed_cache_block_stride,
    int attn_sink_count,
    bool online_softmax,
    int k_split,
    cudaStream_t stream) {
  if (k_split > 1) {
    // K-split path. ``partials`` MUST be non-null and large enough for
    // ``num_tokens * num_heads * k_split * (kHeadDim + 2)`` floats.
    const dim3 chunk_grid(num_tokens, num_heads, k_split);
    const int64_t partials_stride = kHeadDim + 2;
    // T2-α-2 Path B: dispatch the SWA-only specialised template when the
    // caller passed ``compressed_cache == nullptr`` (extra_width == 0).
    // The mixed kSwaOnly=false instantiation matches the previous
    // single-template kernel byte-for-byte.
    if (extra_width == 0) {
      sparse_mla_fp8_cache_online_softmax_k_split_kernel_impl<scalar_t, true>
          <<<chunk_grid, kThreads, 0, stream>>>(
              q, swa_cache, swa_indices, swa_lens, /*compressed_cache=*/nullptr,
              /*extra_indices=*/nullptr, /*extra_lens=*/nullptr, partials,
              sm_scale, num_tokens, num_heads, swa_width, /*extra_width=*/0,
              swa_block_size, compressed_block_size, swa_cache_block_stride,
              compressed_cache_block_stride, k_split, partials_stride);
    } else {
      sparse_mla_fp8_cache_online_softmax_k_split_kernel_impl<scalar_t, false>
          <<<chunk_grid, kThreads, 0, stream>>>(
              q, swa_cache, swa_indices, swa_lens, compressed_cache,
              extra_indices, extra_lens, partials, sm_scale, num_tokens,
              num_heads, swa_width, extra_width, swa_block_size,
              compressed_block_size, swa_cache_block_stride,
              compressed_cache_block_stride, k_split, partials_stride);
    }
    const dim3 reduce_grid(num_tokens, num_heads);
    sparse_mla_fp8_cache_partials_reduce_kernel<scalar_t>
        <<<reduce_grid, kThreads, 0, stream>>>(
            partials, attn_sink, output, num_tokens, num_heads, k_split,
            partials_stride, attn_sink_count);
    return;
  }

  const dim3 grid(num_tokens, num_heads);
  if (online_softmax) {
    sparse_mla_fp8_cache_online_softmax_kernel<scalar_t>
        <<<grid, kThreads, 0, stream>>>(
            q, swa_cache, swa_indices, swa_lens, compressed_cache, extra_indices,
            extra_lens, attn_sink, output, sm_scale, num_tokens, num_heads,
            swa_width, extra_width, swa_block_size, compressed_block_size,
            swa_cache_block_stride, compressed_cache_block_stride,
            attn_sink_count);
  } else {
    sparse_mla_fp8_cache_kernel<scalar_t><<<grid, kThreads, 0, stream>>>(
        q, swa_cache, swa_indices, swa_lens, compressed_cache, extra_indices,
        extra_lens, attn_sink, output, sm_scale, num_tokens, num_heads, swa_width,
        extra_width, swa_block_size, compressed_block_size,
        swa_cache_block_stride, compressed_cache_block_stride, attn_sink_count);
  }
}

void launch_decode_indices(
    const int64_t* positions,
    const int32_t* token_to_req_indices,
    const int32_t* block_table,
    const int32_t* topk_indices,
    int32_t* swa_indices,
    int32_t* swa_lens,
    int32_t* extra_indices,
    int32_t* extra_lens,
    int num_tokens,
    int block_table_width,
    int window_size,
    int swa_width,
    int swa_block_size,
    int compress_ratio,
    int compressed_width,
    int compressed_block_size,
    int topk_width,
    int full_candidate_max_len,
    cudaStream_t stream) {
  decode_indices_kernel<<<num_tokens, kThreads, 0, stream>>>(
      positions, token_to_req_indices, block_table, topk_indices, swa_indices,
      swa_lens, extra_indices, extra_lens, num_tokens, block_table_width,
      window_size, swa_width, swa_block_size, compress_ratio, compressed_width,
      compressed_block_size, topk_width, full_candidate_max_len);
}

void launch_full_candidate_topk(
    const int64_t* positions,
    int32_t* topk,
    int num_tokens,
    int width,
    int compress_ratio,
    cudaStream_t stream) {
  full_candidate_topk_kernel<<<num_tokens, kThreads, 0, stream>>>(
      positions, topk, num_tokens, width, compress_ratio);
}

void launch_save_compressor_state(
    const float* kv,
    const float* score,
    const float* ape,
    float* state_cache,
    const int64_t* slot_mapping,
    const int64_t* positions,
    int num_tokens,
    int state_width,
    int block_size,
    int64_t state_cache_block_stride,
    int64_t state_cache_row_stride,
    int compress_ratio,
    cudaStream_t stream) {
  save_compressor_state_kernel<<<num_tokens, kThreads, 0, stream>>>(
      kv, score, ape, state_cache, slot_mapping, positions, num_tokens,
      state_width, block_size, state_cache_block_stride, state_cache_row_stride,
      compress_ratio);
}

void launch_compressed_kv_cache_insert(
    const float* state_cache,
    const int32_t* token_to_req_indices,
    const int64_t* positions,
    const int64_t* compressor_slot_mapping,
    const int32_t* block_table,
    const float* rms_norm_weight,
    const float* cos_sin_cache,
    uint8_t* kv_cache,
    const int64_t* kv_slot_mapping,
    float rms_norm_eps,
    int num_tokens,
    int state_block_size,
    int64_t state_cache_block_stride,
    int64_t state_cache_row_stride,
    int block_table_width,
    int kv_cache_block_size,
    int64_t kv_cache_block_stride,
    int64_t cos_sin_stride,
    int compress_ratio,
    int state_width,
    cudaStream_t stream) {
  const int overlap = compress_ratio == kCsaCompressRatio ? 1 : 0;
  const int window = overlap != 0 ? kCsaWindow : compress_ratio;
  compressed_kv_cache_insert_kernel<<<num_tokens, kThreads, 0, stream>>>(
      state_cache, token_to_req_indices, positions, compressor_slot_mapping,
      block_table, rms_norm_weight, cos_sin_cache, kv_cache, kv_slot_mapping,
      rms_norm_eps, num_tokens, state_block_size, state_cache_block_stride,
      state_cache_row_stride, block_table_width, kv_cache_block_size,
      kv_cache_block_stride, cos_sin_stride, compress_ratio, window,
      state_width, overlap);
}

__global__ void sm12x_mhc_pre_kernel(
    const nv_bfloat16* __restrict__ residual,
    const float* __restrict__ fn,
    const float* __restrict__ hc_scale,
    const float* __restrict__ hc_base,
    nv_bfloat16* __restrict__ layer_input,
    float* __restrict__ post,
    float* __restrict__ comb,
    int num_tokens,
    int hc_mult,
    int hidden_size,
    int mix_hc,
    float rms_eps,
    float hc_eps,
    int sinkhorn_iters) {
  const int token_idx = blockIdx.x;
  const int tid = threadIdx.x;
  if (token_idx >= num_tokens) {
    return;
  }

  const int width = hc_mult * hidden_size;
  float partial[kSm12xMhcMaxMix + 1];
  for (int j = 0; j <= kSm12xMhcMaxMix; ++j) {
    partial[j] = 0.0f;
  }

  const int64_t residual_base = static_cast<int64_t>(token_idx) * width;
  for (int idx = tid; idx < width; idx += blockDim.x) {
    const float value = __bfloat162float(residual[residual_base + idx]);
    partial[0] += value * value;
    for (int j = 0; j < mix_hc; ++j) {
      partial[j + 1] += value * fn[static_cast<int64_t>(j) * width + idx];
    }
  }

  extern __shared__ float shared[];
  for (int j = 0; j <= mix_hc; ++j) {
    shared[j * blockDim.x + tid] = partial[j];
  }
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      for (int j = 0; j <= mix_hc; ++j) {
        shared[j * blockDim.x + tid] += shared[j * blockDim.x + tid + stride];
      }
    }
    __syncthreads();
  }

  float* pre_shared = shared + (mix_hc + 1) * blockDim.x;
  if (tid == 0) {
    const float rms = rsqrtf(shared[0] / static_cast<float>(width) + rms_eps);
    float mixes[kSm12xMhcMaxMix];
    for (int j = 0; j < kSm12xMhcMaxMix; ++j) {
      mixes[j] = 0.0f;
    }
    for (int j = 0; j < mix_hc; ++j) {
      mixes[j] = shared[(j + 1) * blockDim.x] * rms;
    }

    for (int i = 0; i < hc_mult; ++i) {
      const float pre =
          sigmoidf_approx(mixes[i] * hc_scale[0] + hc_base[i]) + hc_eps;
      pre_shared[i] = pre;
      post[static_cast<int64_t>(token_idx) * hc_mult + i] =
          sigmoidf_approx(
              mixes[hc_mult + i] * hc_scale[1] + hc_base[hc_mult + i]) *
          2.0f;
    }

    float cm[kSm12xMhcMaxHc][kSm12xMhcMaxHc];
    const int comb_offset = hc_mult * 2;
    for (int row = 0; row < hc_mult; ++row) {
      float row_max = kNegInf;
      for (int col = 0; col < hc_mult; ++col) {
        const int idx = comb_offset + row * hc_mult + col;
        const float value = mixes[idx] * hc_scale[2] + hc_base[idx];
        cm[row][col] = value;
        row_max = fmaxf(row_max, value);
      }
      float row_sum = 0.0f;
      for (int col = 0; col < hc_mult; ++col) {
        const float value = expf(cm[row][col] - row_max);
        cm[row][col] = value;
        row_sum += value;
      }
      for (int col = 0; col < hc_mult; ++col) {
        cm[row][col] = cm[row][col] / row_sum + hc_eps;
      }
    }

    for (int col = 0; col < hc_mult; ++col) {
      float col_sum = 0.0f;
      for (int row = 0; row < hc_mult; ++row) {
        col_sum += cm[row][col];
      }
      for (int row = 0; row < hc_mult; ++row) {
        cm[row][col] = cm[row][col] / (col_sum + hc_eps);
      }
    }

    for (int iter = 1; iter < sinkhorn_iters; ++iter) {
      for (int row = 0; row < hc_mult; ++row) {
        float row_sum = 0.0f;
        for (int col = 0; col < hc_mult; ++col) {
          row_sum += cm[row][col];
        }
        for (int col = 0; col < hc_mult; ++col) {
          cm[row][col] = cm[row][col] / (row_sum + hc_eps);
        }
      }
      for (int col = 0; col < hc_mult; ++col) {
        float col_sum = 0.0f;
        for (int row = 0; row < hc_mult; ++row) {
          col_sum += cm[row][col];
        }
        for (int row = 0; row < hc_mult; ++row) {
          cm[row][col] = cm[row][col] / (col_sum + hc_eps);
        }
      }
    }

    for (int row = 0; row < hc_mult; ++row) {
      for (int col = 0; col < hc_mult; ++col) {
        comb[(static_cast<int64_t>(token_idx) * hc_mult + row) * hc_mult +
             col] = cm[row][col];
      }
    }
  }
  __syncthreads();

  for (int h = tid; h < hidden_size; h += blockDim.x) {
    float value = 0.0f;
    for (int i = 0; i < hc_mult; ++i) {
      value += pre_shared[i] *
               __bfloat162float(
                   residual[residual_base + i * hidden_size + h]);
    }
    layer_input[static_cast<int64_t>(token_idx) * hidden_size + h] =
        __float2bfloat16(value);
  }
}

__global__ void sm12x_mhc_pre_split_partial_kernel(
    const nv_bfloat16* __restrict__ residual,
    const float* __restrict__ fn,
    float* __restrict__ partials,
    int num_tokens,
    int num_splits,
    int hc_mult,
    int hidden_size,
    int mix_hc) {
  const int token_idx = blockIdx.x;
  const int split_idx = blockIdx.y;
  const int tid = threadIdx.x;
  if (token_idx >= num_tokens || split_idx >= num_splits) {
    return;
  }

  const int width = hc_mult * hidden_size;
  const int split_begin =
      static_cast<int>((static_cast<int64_t>(width) * split_idx) / num_splits);
  const int split_end = static_cast<int>(
      (static_cast<int64_t>(width) * (split_idx + 1)) / num_splits);
  float partial[kSm12xMhcMaxMix + 1];
  for (int j = 0; j <= kSm12xMhcMaxMix; ++j) {
    partial[j] = 0.0f;
  }

  const int64_t residual_base =
      static_cast<int64_t>(token_idx) * static_cast<int64_t>(width);
  for (int idx = split_begin + tid; idx < split_end; idx += blockDim.x) {
    const float value = __bfloat162float(residual[residual_base + idx]);
    partial[0] += value * value;
    for (int j = 0; j < mix_hc; ++j) {
      partial[j + 1] += value * fn[static_cast<int64_t>(j) * width + idx];
    }
  }

  extern __shared__ float shared[];
  for (int j = 0; j <= mix_hc; ++j) {
    shared[j * blockDim.x + tid] = partial[j];
  }
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      for (int j = 0; j <= mix_hc; ++j) {
        shared[j * blockDim.x + tid] += shared[j * blockDim.x + tid + stride];
      }
    }
    __syncthreads();
  }

  if (tid <= mix_hc) {
    partials[(static_cast<int64_t>(token_idx) * num_splits + split_idx) *
                 (mix_hc + 1) +
             tid] = shared[tid * blockDim.x];
  }
}

__global__ void sm12x_mhc_pre_split_finalize_kernel(
    const nv_bfloat16* __restrict__ residual,
    float* __restrict__ partials,
    const float* __restrict__ hc_scale,
    const float* __restrict__ hc_base,
    nv_bfloat16* __restrict__ layer_input,
    float* __restrict__ post,
    float* __restrict__ comb,
    int num_tokens,
    int num_splits,
    int hc_mult,
    int hidden_size,
    int mix_hc,
    float rms_eps,
    float hc_eps,
    int sinkhorn_iters,
    int write_layer_input) {
  const int token_idx = blockIdx.x;
  const int tid = threadIdx.x;
  if (token_idx >= num_tokens) {
    return;
  }

  extern __shared__ float pre_shared[];
  const int width = hc_mult * hidden_size;
  if (tid == 0) {
    float sums[kSm12xMhcMaxMix + 1];
    for (int j = 0; j <= kSm12xMhcMaxMix; ++j) {
      sums[j] = 0.0f;
    }
    const int64_t partial_base =
        static_cast<int64_t>(token_idx) * num_splits * (mix_hc + 1);
    for (int split = 0; split < num_splits; ++split) {
      for (int j = 0; j <= mix_hc; ++j) {
        sums[j] += partials[partial_base + split * (mix_hc + 1) + j];
      }
    }

    const float rms = rsqrtf(sums[0] / static_cast<float>(width) + rms_eps);
    float mixes[kSm12xMhcMaxMix];
    for (int j = 0; j < kSm12xMhcMaxMix; ++j) {
      mixes[j] = 0.0f;
    }
    for (int j = 0; j < mix_hc; ++j) {
      mixes[j] = sums[j + 1] * rms;
    }

    for (int i = 0; i < hc_mult; ++i) {
      const float pre =
          sigmoidf_approx(mixes[i] * hc_scale[0] + hc_base[i]) + hc_eps;
      pre_shared[i] = pre;
      partials[partial_base + i] = pre;
      post[static_cast<int64_t>(token_idx) * hc_mult + i] =
          sigmoidf_approx(
              mixes[hc_mult + i] * hc_scale[1] + hc_base[hc_mult + i]) *
          2.0f;
    }

    float cm[kSm12xMhcMaxHc][kSm12xMhcMaxHc];
    const int comb_offset = hc_mult * 2;
    for (int row = 0; row < hc_mult; ++row) {
      float row_max = kNegInf;
      for (int col = 0; col < hc_mult; ++col) {
        const int idx = comb_offset + row * hc_mult + col;
        const float value = mixes[idx] * hc_scale[2] + hc_base[idx];
        cm[row][col] = value;
        row_max = fmaxf(row_max, value);
      }
      float row_sum = 0.0f;
      for (int col = 0; col < hc_mult; ++col) {
        const float value = expf(cm[row][col] - row_max);
        cm[row][col] = value;
        row_sum += value;
      }
      for (int col = 0; col < hc_mult; ++col) {
        cm[row][col] = cm[row][col] / row_sum + hc_eps;
      }
    }

    for (int col = 0; col < hc_mult; ++col) {
      float col_sum = 0.0f;
      for (int row = 0; row < hc_mult; ++row) {
        col_sum += cm[row][col];
      }
      for (int row = 0; row < hc_mult; ++row) {
        cm[row][col] = cm[row][col] / (col_sum + hc_eps);
      }
    }

    for (int iter = 1; iter < sinkhorn_iters; ++iter) {
      for (int row = 0; row < hc_mult; ++row) {
        float row_sum = 0.0f;
        for (int col = 0; col < hc_mult; ++col) {
          row_sum += cm[row][col];
        }
        for (int col = 0; col < hc_mult; ++col) {
          cm[row][col] = cm[row][col] / (row_sum + hc_eps);
        }
      }
      for (int col = 0; col < hc_mult; ++col) {
        float col_sum = 0.0f;
        for (int row = 0; row < hc_mult; ++row) {
          col_sum += cm[row][col];
        }
        for (int row = 0; row < hc_mult; ++row) {
          cm[row][col] = cm[row][col] / (col_sum + hc_eps);
        }
      }
    }

    for (int row = 0; row < hc_mult; ++row) {
      for (int col = 0; col < hc_mult; ++col) {
        comb[(static_cast<int64_t>(token_idx) * hc_mult + row) * hc_mult +
             col] = cm[row][col];
      }
    }
  }
  __syncthreads();
  if (write_layer_input == 0) {
    return;
  }

  const int64_t residual_base =
      static_cast<int64_t>(token_idx) * static_cast<int64_t>(width);
  for (int h = tid; h < hidden_size; h += blockDim.x) {
    float value = 0.0f;
    for (int i = 0; i < hc_mult; ++i) {
      value += pre_shared[i] *
               __bfloat162float(
                   residual[residual_base + i * hidden_size + h]);
    }
    layer_input[static_cast<int64_t>(token_idx) * hidden_size + h] =
        __float2bfloat16(value);
  }
}

__global__ void sm12x_mhc_pre_split_apply_kernel(
    const nv_bfloat16* __restrict__ residual,
    const float* __restrict__ partials,
    nv_bfloat16* __restrict__ layer_input,
    int num_tokens,
    int num_splits,
    int hc_mult,
    int hidden_size,
    int mix_hc) {
  const int token_idx = blockIdx.x;
  const int chunk_idx = blockIdx.y;
  const int tid = threadIdx.x;
  if (token_idx >= num_tokens) {
    return;
  }

  const int h = chunk_idx * blockDim.x + tid;
  if (h >= hidden_size) {
    return;
  }

  const int width = hc_mult * hidden_size;
  const int64_t residual_base =
      static_cast<int64_t>(token_idx) * static_cast<int64_t>(width);
  const int64_t partial_base =
      static_cast<int64_t>(token_idx) * num_splits * (mix_hc + 1);

  float value = 0.0f;
#pragma unroll
  for (int i = 0; i < kSm12xMhcMaxHc; ++i) {
    if (i < hc_mult) {
      value += partials[partial_base + i] *
               __bfloat162float(
                   residual[residual_base + i * hidden_size + h]);
    }
  }
  layer_input[static_cast<int64_t>(token_idx) * hidden_size + h] =
      __float2bfloat16(value);
}

__global__ void sm12x_mhc_post_kernel(
    const nv_bfloat16* __restrict__ hidden_states,
    const nv_bfloat16* __restrict__ residual,
    const float* __restrict__ post,
    const float* __restrict__ comb,
    nv_bfloat16* __restrict__ output,
    int num_tokens,
    int hc_mult,
    int hidden_size) {
  const int token_idx = blockIdx.x;
  const int out_lane = blockIdx.y;
  const int tid = threadIdx.x;
  if (token_idx >= num_tokens || out_lane >= hc_mult) {
    return;
  }

  const int64_t token_hidden_base =
      static_cast<int64_t>(token_idx) * hidden_size;
  const int64_t token_hc_base =
      static_cast<int64_t>(token_idx) * hc_mult * hidden_size;
  const int64_t comb_base =
      static_cast<int64_t>(token_idx) * hc_mult * hc_mult;
  const float post_value =
      post[static_cast<int64_t>(token_idx) * hc_mult + out_lane];

  for (int h = tid; h < hidden_size; h += blockDim.x) {
    float value =
        post_value * __bfloat162float(hidden_states[token_hidden_base + h]);
    for (int in_lane = 0; in_lane < hc_mult; ++in_lane) {
      value += comb[comb_base + in_lane * hc_mult + out_lane] *
               __bfloat162float(
                   residual[token_hc_base + in_lane * hidden_size + h]);
    }
    output[token_hc_base + out_lane * hidden_size + h] =
        __float2bfloat16(value);
  }
}

}  // namespace

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
    int64_t sinkhorn_iters) {
  CHECK_CUDA(residual);
  CHECK_CUDA(fn);
  CHECK_CUDA(hc_scale);
  CHECK_CUDA(hc_base);
  CHECK_CUDA(layer_input);
  CHECK_CUDA(post);
  CHECK_CUDA(comb);
  CHECK_DIM(3, residual);
  CHECK_DIM(2, fn);
  CHECK_DIM(1, hc_scale);
  CHECK_DIM(1, hc_base);
  CHECK_DIM(2, layer_input);
  CHECK_DIM(3, post);
  CHECK_DIM(3, comb);

  TVM_FFI_ICHECK(residual.IsContiguous()) << "residual must be contiguous";
  TVM_FFI_ICHECK(fn.IsContiguous()) << "fn must be contiguous";
  TVM_FFI_ICHECK(hc_scale.IsContiguous()) << "hc_scale must be contiguous";
  TVM_FFI_ICHECK(hc_base.IsContiguous()) << "hc_base must be contiguous";
  TVM_FFI_ICHECK(layer_input.IsContiguous()) << "layer_input must be contiguous";
  TVM_FFI_ICHECK(post.IsContiguous()) << "post must be contiguous";
  TVM_FFI_ICHECK(comb.IsContiguous()) << "comb must be contiguous";
  TVM_FFI_ICHECK(residual.dtype() == dl_bfloat16)
      << "residual must be bfloat16";
  TVM_FFI_ICHECK(layer_input.dtype() == dl_bfloat16)
      << "layer_input must be bfloat16";
  TVM_FFI_ICHECK(fn.dtype() == dl_float32) << "fn must be float32";
  TVM_FFI_ICHECK(hc_scale.dtype() == dl_float32)
      << "hc_scale must be float32";
  TVM_FFI_ICHECK(hc_base.dtype() == dl_float32)
      << "hc_base must be float32";
  TVM_FFI_ICHECK(post.dtype() == dl_float32) << "post must be float32";
  TVM_FFI_ICHECK(comb.dtype() == dl_float32) << "comb must be float32";

  const int num_tokens = static_cast<int>(residual.size(0));
  const int hc_mult = static_cast<int>(residual.size(1));
  const int hidden_size = static_cast<int>(residual.size(2));
  const int mix_hc = hc_mult * (2 + hc_mult);
  TVM_FFI_ICHECK(num_tokens <= 16)
      << "SM12x mHC pre native path is decode-only for <=16 tokens";
  TVM_FFI_ICHECK(hc_mult > 0 && hc_mult <= kSm12xMhcMaxHc)
      << "unsupported hc_mult";
  TVM_FFI_ICHECK(mix_hc <= kSm12xMhcMaxMix) << "unsupported mHC mix width";
  TVM_FFI_ICHECK(hidden_size > 0) << "hidden_size must be positive";
  TVM_FFI_ICHECK(fn.size(0) == mix_hc) << "fn row count mismatch";
  TVM_FFI_ICHECK(fn.size(1) == hc_mult * hidden_size)
      << "fn input width mismatch";
  TVM_FFI_ICHECK(hc_scale.size(0) >= 3) << "hc_scale must have at least 3 rows";
  TVM_FFI_ICHECK(hc_base.size(0) == mix_hc) << "hc_base size mismatch";
  TVM_FFI_ICHECK(layer_input.size(0) == num_tokens)
      << "layer_input token count mismatch";
  TVM_FFI_ICHECK(layer_input.size(1) == hidden_size)
      << "layer_input hidden size mismatch";
  TVM_FFI_ICHECK(post.size(0) == num_tokens && post.size(1) == hc_mult &&
                 post.size(2) == 1)
      << "post shape mismatch";
  TVM_FFI_ICHECK(comb.size(0) == num_tokens && comb.size(1) == hc_mult &&
                 comb.size(2) == hc_mult)
      << "comb shape mismatch";
  TVM_FFI_ICHECK(sinkhorn_iters >= 1) << "sinkhorn_iters must be >= 1";
  if (num_tokens == 0) {
    return;
  }

  cudaSetDevice(residual.device().device_id);
  const cudaStream_t stream = get_stream(residual.device());
  const int threads = kThreads;
  const size_t shared_bytes =
      static_cast<size_t>((mix_hc + 1) * threads + hc_mult) * sizeof(float);
  sm12x_mhc_pre_kernel<<<num_tokens, threads, shared_bytes, stream>>>(
      static_cast<const nv_bfloat16*>(residual.data_ptr()),
      static_cast<const float*>(fn.data_ptr()),
      static_cast<const float*>(hc_scale.data_ptr()),
      static_cast<const float*>(hc_base.data_ptr()),
      static_cast<nv_bfloat16*>(layer_input.data_ptr()),
      static_cast<float*>(post.data_ptr()),
      static_cast<float*>(comb.data_ptr()), num_tokens, hc_mult, hidden_size,
      mix_hc, static_cast<float>(rms_eps), static_cast<float>(hc_eps),
      static_cast<int>(sinkhorn_iters));
  cudaError_t status = cudaGetLastError();
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "deepseek_v4_sm12x_mhc_pre_cuda failed: "
      << cudaGetErrorString(status);
}

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
    int64_t sinkhorn_iters) {
  CHECK_CUDA(residual);
  CHECK_CUDA(fn);
  CHECK_CUDA(hc_scale);
  CHECK_CUDA(hc_base);
  CHECK_CUDA(layer_input);
  CHECK_CUDA(post);
  CHECK_CUDA(comb);
  CHECK_CUDA(partials);
  CHECK_DIM(3, residual);
  CHECK_DIM(2, fn);
  CHECK_DIM(1, hc_scale);
  CHECK_DIM(1, hc_base);
  CHECK_DIM(2, layer_input);
  CHECK_DIM(3, post);
  CHECK_DIM(3, comb);
  CHECK_DIM(3, partials);

  TVM_FFI_ICHECK(residual.IsContiguous()) << "residual must be contiguous";
  TVM_FFI_ICHECK(fn.IsContiguous()) << "fn must be contiguous";
  TVM_FFI_ICHECK(hc_scale.IsContiguous()) << "hc_scale must be contiguous";
  TVM_FFI_ICHECK(hc_base.IsContiguous()) << "hc_base must be contiguous";
  TVM_FFI_ICHECK(layer_input.IsContiguous()) << "layer_input must be contiguous";
  TVM_FFI_ICHECK(post.IsContiguous()) << "post must be contiguous";
  TVM_FFI_ICHECK(comb.IsContiguous()) << "comb must be contiguous";
  TVM_FFI_ICHECK(partials.IsContiguous()) << "partials must be contiguous";
  TVM_FFI_ICHECK(residual.dtype() == dl_bfloat16)
      << "residual must be bfloat16";
  TVM_FFI_ICHECK(layer_input.dtype() == dl_bfloat16)
      << "layer_input must be bfloat16";
  TVM_FFI_ICHECK(fn.dtype() == dl_float32) << "fn must be float32";
  TVM_FFI_ICHECK(hc_scale.dtype() == dl_float32)
      << "hc_scale must be float32";
  TVM_FFI_ICHECK(hc_base.dtype() == dl_float32)
      << "hc_base must be float32";
  TVM_FFI_ICHECK(post.dtype() == dl_float32) << "post must be float32";
  TVM_FFI_ICHECK(comb.dtype() == dl_float32) << "comb must be float32";
  TVM_FFI_ICHECK(partials.dtype() == dl_float32)
      << "partials must be float32";

  const int num_tokens = static_cast<int>(residual.size(0));
  const int hc_mult = static_cast<int>(residual.size(1));
  const int hidden_size = static_cast<int>(residual.size(2));
  const int mix_hc = hc_mult * (2 + hc_mult);
  const int num_splits = static_cast<int>(partials.size(1));
  TVM_FFI_ICHECK(num_tokens <= 16)
      << "SM12x mHC pre native path is decode-only for <=16 tokens";
  TVM_FFI_ICHECK(hc_mult > 0 && hc_mult <= kSm12xMhcMaxHc)
      << "unsupported hc_mult";
  TVM_FFI_ICHECK(mix_hc <= kSm12xMhcMaxMix) << "unsupported mHC mix width";
  TVM_FFI_ICHECK(hidden_size > 0) << "hidden_size must be positive";
  TVM_FFI_ICHECK(num_splits > 0 && num_splits <= 16)
      << "unsupported mHC split count";
  TVM_FFI_ICHECK(fn.size(0) == mix_hc) << "fn row count mismatch";
  TVM_FFI_ICHECK(fn.size(1) == hc_mult * hidden_size)
      << "fn input width mismatch";
  TVM_FFI_ICHECK(hc_scale.size(0) >= 3) << "hc_scale must have at least 3 rows";
  TVM_FFI_ICHECK(hc_base.size(0) == mix_hc) << "hc_base size mismatch";
  TVM_FFI_ICHECK(layer_input.size(0) == num_tokens)
      << "layer_input token count mismatch";
  TVM_FFI_ICHECK(layer_input.size(1) == hidden_size)
      << "layer_input hidden size mismatch";
  TVM_FFI_ICHECK(post.size(0) == num_tokens && post.size(1) == hc_mult &&
                 post.size(2) == 1)
      << "post shape mismatch";
  TVM_FFI_ICHECK(comb.size(0) == num_tokens && comb.size(1) == hc_mult &&
                 comb.size(2) == hc_mult)
      << "comb shape mismatch";
  TVM_FFI_ICHECK(partials.size(0) == num_tokens &&
                 partials.size(2) == mix_hc + 1)
      << "partials shape mismatch";
  TVM_FFI_ICHECK(sinkhorn_iters >= 1) << "sinkhorn_iters must be >= 1";
  if (num_tokens == 0) {
    return;
  }

  cudaSetDevice(residual.device().device_id);
  const cudaStream_t stream = get_stream(residual.device());
  const int threads = kThreads;
  const size_t partial_shared_bytes =
      static_cast<size_t>((mix_hc + 1) * threads) * sizeof(float);
  dim3 partial_grid(num_tokens, num_splits);
  sm12x_mhc_pre_split_partial_kernel<<<
      partial_grid, threads, partial_shared_bytes, stream>>>(
      static_cast<const nv_bfloat16*>(residual.data_ptr()),
      static_cast<const float*>(fn.data_ptr()),
      static_cast<float*>(partials.data_ptr()), num_tokens, num_splits,
      hc_mult, hidden_size, mix_hc);
  cudaError_t status = cudaGetLastError();
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "deepseek_v4_sm12x_mhc_pre_split partial failed: "
      << cudaGetErrorString(status);

  const size_t finalize_shared_bytes =
      static_cast<size_t>(hc_mult) * sizeof(float);
  const bool parallel_apply =
      hc_mult == 4 && hidden_size >= 2048 && num_splits >= 8;
  sm12x_mhc_pre_split_finalize_kernel<<<
      num_tokens, threads, finalize_shared_bytes, stream>>>(
      static_cast<const nv_bfloat16*>(residual.data_ptr()),
      static_cast<float*>(partials.data_ptr()),
      static_cast<const float*>(hc_scale.data_ptr()),
      static_cast<const float*>(hc_base.data_ptr()),
      static_cast<nv_bfloat16*>(layer_input.data_ptr()),
      static_cast<float*>(post.data_ptr()),
      static_cast<float*>(comb.data_ptr()), num_tokens, num_splits, hc_mult,
      hidden_size, mix_hc, static_cast<float>(rms_eps),
      static_cast<float>(hc_eps), static_cast<int>(sinkhorn_iters),
      parallel_apply ? 0 : 1);
  status = cudaGetLastError();
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "deepseek_v4_sm12x_mhc_pre_split finalize failed: "
      << cudaGetErrorString(status);

  if (parallel_apply) {
    const dim3 apply_grid(
        num_tokens,
        (hidden_size + threads - 1) / threads);
    sm12x_mhc_pre_split_apply_kernel<<<apply_grid, threads, 0, stream>>>(
        static_cast<const nv_bfloat16*>(residual.data_ptr()),
        static_cast<const float*>(partials.data_ptr()),
        static_cast<nv_bfloat16*>(layer_input.data_ptr()), num_tokens,
        num_splits, hc_mult, hidden_size, mix_hc);
    status = cudaGetLastError();
    TVM_FFI_ICHECK(status == cudaSuccess)
        << "deepseek_v4_sm12x_mhc_pre_split apply failed: "
        << cudaGetErrorString(status);
  }
}

void deepseek_v4_sm12x_mhc_post_cuda(
    TensorView hidden_states,
    TensorView residual,
    TensorView post,
    TensorView comb,
    TensorView output) {
  CHECK_CUDA(hidden_states);
  CHECK_CUDA(residual);
  CHECK_CUDA(post);
  CHECK_CUDA(comb);
  CHECK_CUDA(output);
  CHECK_DIM(2, hidden_states);
  CHECK_DIM(3, residual);
  CHECK_DIM(3, post);
  CHECK_DIM(3, comb);
  CHECK_DIM(3, output);

  TVM_FFI_ICHECK(hidden_states.IsContiguous())
      << "hidden_states must be contiguous";
  TVM_FFI_ICHECK(residual.IsContiguous()) << "residual must be contiguous";
  TVM_FFI_ICHECK(post.IsContiguous()) << "post must be contiguous";
  TVM_FFI_ICHECK(comb.IsContiguous()) << "comb must be contiguous";
  TVM_FFI_ICHECK(output.IsContiguous()) << "output must be contiguous";
  TVM_FFI_ICHECK(hidden_states.dtype() == dl_bfloat16)
      << "hidden_states must be bfloat16";
  TVM_FFI_ICHECK(residual.dtype() == dl_bfloat16)
      << "residual must be bfloat16";
  TVM_FFI_ICHECK(output.dtype() == dl_bfloat16)
      << "output must be bfloat16";
  TVM_FFI_ICHECK(post.dtype() == dl_float32) << "post must be float32";
  TVM_FFI_ICHECK(comb.dtype() == dl_float32) << "comb must be float32";

  const int num_tokens = static_cast<int>(residual.size(0));
  const int hc_mult = static_cast<int>(residual.size(1));
  const int hidden_size = static_cast<int>(residual.size(2));
  TVM_FFI_ICHECK(num_tokens <= 16)
      << "SM12x mHC post native path is decode-only for <=16 tokens";
  TVM_FFI_ICHECK(hc_mult > 0 && hc_mult <= kSm12xMhcMaxHc)
      << "unsupported hc_mult";
  TVM_FFI_ICHECK(hidden_size > 0) << "hidden_size must be positive";
  TVM_FFI_ICHECK(hidden_states.size(0) == num_tokens)
      << "hidden_states token count mismatch";
  TVM_FFI_ICHECK(hidden_states.size(1) == hidden_size)
      << "hidden_states hidden size mismatch";
  TVM_FFI_ICHECK(post.size(0) == num_tokens && post.size(1) == hc_mult &&
                 post.size(2) == 1)
      << "post shape mismatch";
  TVM_FFI_ICHECK(comb.size(0) == num_tokens && comb.size(1) == hc_mult &&
                 comb.size(2) == hc_mult)
      << "comb shape mismatch";
  TVM_FFI_ICHECK(output.size(0) == num_tokens && output.size(1) == hc_mult &&
                 output.size(2) == hidden_size)
      << "output shape mismatch";
  if (num_tokens == 0) {
    return;
  }

  cudaSetDevice(residual.device().device_id);
  const cudaStream_t stream = get_stream(residual.device());
  dim3 grid(num_tokens, hc_mult);
  sm12x_mhc_post_kernel<<<grid, kThreads, 0, stream>>>(
      static_cast<const nv_bfloat16*>(hidden_states.data_ptr()),
      static_cast<const nv_bfloat16*>(residual.data_ptr()),
      static_cast<const float*>(post.data_ptr()),
      static_cast<const float*>(comb.data_ptr()),
      static_cast<nv_bfloat16*>(output.data_ptr()), num_tokens, hc_mult,
      hidden_size);
  cudaError_t status = cudaGetLastError();
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "deepseek_v4_sm12x_mhc_post_cuda failed: "
      << cudaGetErrorString(status);
}

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
    int64_t compress_ratio) {
  CHECK_CUDA(state_cache);
  CHECK_CUDA(token_to_req_indices);
  CHECK_CUDA(positions);
  CHECK_CUDA(compressor_slot_mapping);
  CHECK_CUDA(block_table);
  CHECK_CUDA(rms_norm_weight);
  CHECK_CUDA(cos_sin_cache);
  CHECK_CUDA(kv_cache);
  CHECK_CUDA(kv_slot_mapping);
  CHECK_DIM(3, state_cache);
  CHECK_DIM(1, token_to_req_indices);
  CHECK_DIM(1, positions);
  CHECK_DIM(1, compressor_slot_mapping);
  CHECK_DIM(2, block_table);
  CHECK_DIM(1, rms_norm_weight);
  CHECK_DIM(2, cos_sin_cache);
  CHECK_DIM(2, kv_cache);
  CHECK_DIM(1, kv_slot_mapping);

  TVM_FFI_ICHECK(state_cache.dtype() == dl_float32)
      << "state_cache must be float32";
  TVM_FFI_ICHECK(token_to_req_indices.dtype() == dl_int32)
      << "token_to_req_indices must be int32";
  TVM_FFI_ICHECK(positions.dtype() == dl_int64) << "positions must be int64";
  TVM_FFI_ICHECK(compressor_slot_mapping.dtype() == dl_int64)
      << "compressor_slot_mapping must be int64";
  TVM_FFI_ICHECK(block_table.dtype() == dl_int32) << "block_table must be int32";
  TVM_FFI_ICHECK(rms_norm_weight.dtype() == dl_float32)
      << "rms_norm_weight must be float32";
  TVM_FFI_ICHECK(cos_sin_cache.dtype() == dl_float32)
      << "cos_sin_cache must be float32";
  TVM_FFI_ICHECK(kv_cache.dtype() == dl_uint8) << "kv_cache must be uint8";
  TVM_FFI_ICHECK(kv_slot_mapping.dtype() == dl_int64)
      << "kv_slot_mapping must be int64";
  TVM_FFI_ICHECK(compress_ratio == 4 || compress_ratio == 128)
      << "compress_ratio must be 4 or 128";
  const int64_t expected_state_width =
      compress_ratio == 4 ? kHeadDim * 2 : kHeadDim;
  TVM_FFI_ICHECK(state_cache.size(2) == expected_state_width * 2)
      << "state_cache last dim does not match compressed KV state width";
  TVM_FFI_ICHECK(rms_norm_weight.size(0) == kHeadDim)
      << "rms_norm_weight must have 512 values";
  TVM_FFI_ICHECK(cos_sin_cache.size(1) >= kRopeDim)
      << "cos_sin_cache must have at least 64 columns";
  TVM_FFI_ICHECK(compressor_block_size == state_cache.size(1))
      << "compressor_block_size must match state_cache page size";
  TVM_FFI_ICHECK(kv_cache.size(1) >=
                 kv_cache_block_size *
                     (kTokenDataBytes + kScaleBytesPerToken))
      << "kv_cache block stride is too small for DeepSeek V4 rows";
  TVM_FFI_ICHECK(positions.size(0) <= token_to_req_indices.size(0))
      << "token_to_req_indices must cover positions";
  TVM_FFI_ICHECK(positions.size(0) <= compressor_slot_mapping.size(0))
      << "compressor_slot_mapping must cover positions";
  TVM_FFI_ICHECK(positions.size(0) <= kv_slot_mapping.size(0))
      << "kv_slot_mapping must cover positions";
  TVM_FFI_ICHECK(compressor_block_size > 0)
      << "compressor_block_size must be positive";
  TVM_FFI_ICHECK(kv_cache_block_size > 0)
      << "kv_cache_block_size must be positive";

  cudaSetDevice(state_cache.device().device_id);
  const cudaStream_t stream = get_stream(state_cache.device());
  const int num_tokens = static_cast<int>(positions.size(0));
  if (num_tokens == 0) {
    return;
  }

  launch_compressed_kv_cache_insert(
      static_cast<const float*>(state_cache.data_ptr()),
      static_cast<const int32_t*>(token_to_req_indices.data_ptr()),
      static_cast<const int64_t*>(positions.data_ptr()),
      static_cast<const int64_t*>(compressor_slot_mapping.data_ptr()),
      static_cast<const int32_t*>(block_table.data_ptr()),
      static_cast<const float*>(rms_norm_weight.data_ptr()),
      static_cast<const float*>(cos_sin_cache.data_ptr()),
      static_cast<uint8_t*>(kv_cache.data_ptr()),
      static_cast<const int64_t*>(kv_slot_mapping.data_ptr()),
      static_cast<float>(rms_norm_eps), num_tokens,
      static_cast<int>(compressor_block_size), state_cache.stride(0),
      state_cache.stride(1), static_cast<int>(block_table.size(1)),
      static_cast<int>(kv_cache_block_size), kv_cache.stride(0),
      cos_sin_cache.stride(0), static_cast<int>(compress_ratio),
      static_cast<int>(expected_state_width), stream);

  cudaError_t status = cudaGetLastError();
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "deepseek_v4_compressed_kv_cache_insert_cuda failed: "
      << cudaGetErrorString(status);
}

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
    int64_t kv_cache_block_size) {
  CHECK_CUDA(state_cache);
  CHECK_CUDA(token_to_req_indices);
  CHECK_CUDA(positions);
  CHECK_CUDA(compressor_slot_mapping);
  CHECK_CUDA(block_table);
  CHECK_CUDA(rms_norm_weight);
  CHECK_CUDA(cos_sin_cache);
  CHECK_CUDA(kv_cache);
  CHECK_CUDA(kv_slot_mapping);
  CHECK_DIM(3, state_cache);
  CHECK_DIM(1, token_to_req_indices);
  CHECK_DIM(1, positions);
  CHECK_DIM(1, compressor_slot_mapping);
  CHECK_DIM(2, block_table);
  CHECK_DIM(1, rms_norm_weight);
  CHECK_DIM(2, cos_sin_cache);
  CHECK_DIM(2, kv_cache);
  CHECK_DIM(1, kv_slot_mapping);

  TVM_FFI_ICHECK(state_cache.dtype() == dl_float32)
      << "state_cache must be float32";
  TVM_FFI_ICHECK(token_to_req_indices.dtype() == dl_int32)
      << "token_to_req_indices must be int32";
  TVM_FFI_ICHECK(positions.dtype() == dl_int64) << "positions must be int64";
  TVM_FFI_ICHECK(compressor_slot_mapping.dtype() == dl_int64)
      << "compressor_slot_mapping must be int64";
  TVM_FFI_ICHECK(block_table.dtype() == dl_int32) << "block_table must be int32";
  TVM_FFI_ICHECK(rms_norm_weight.dtype() == dl_float32)
      << "rms_norm_weight must be float32";
  TVM_FFI_ICHECK(cos_sin_cache.dtype() == dl_float32)
      << "cos_sin_cache must be float32";
  TVM_FFI_ICHECK(kv_cache.dtype() == dl_uint8) << "kv_cache must be uint8";
  TVM_FFI_ICHECK(kv_slot_mapping.dtype() == dl_int64)
      << "kv_slot_mapping must be int64";
  TVM_FFI_ICHECK(state_cache.size(2) == kIndexerStateWidth * 2)
      << "state_cache last dim must be 512 for CSA indexer";
  TVM_FFI_ICHECK(rms_norm_weight.size(0) == kIndexerDim)
      << "rms_norm_weight must have 128 values";
  TVM_FFI_ICHECK(cos_sin_cache.size(1) >= kIndexerRopeDim)
      << "cos_sin_cache must have at least 64 columns";
  TVM_FFI_ICHECK(compressor_block_size == state_cache.size(1))
      << "compressor_block_size must match state_cache page size";
  TVM_FFI_ICHECK(kv_cache.size(1) >=
                 kv_cache_block_size *
                     (kIndexerCacheValueBytes + kIndexerCacheScaleBytes))
      << "kv_cache block stride is too small for FP8 indexer rows";
  TVM_FFI_ICHECK(positions.size(0) <= token_to_req_indices.size(0))
      << "token_to_req_indices must cover positions";
  TVM_FFI_ICHECK(positions.size(0) <= compressor_slot_mapping.size(0))
      << "compressor_slot_mapping must cover positions";
  TVM_FFI_ICHECK(positions.size(0) <= kv_slot_mapping.size(0))
      << "kv_slot_mapping must cover positions";
  TVM_FFI_ICHECK(compressor_block_size > 0)
      << "compressor_block_size must be positive";
  TVM_FFI_ICHECK(kv_cache_block_size > 0)
      << "kv_cache_block_size must be positive";

  cudaSetDevice(state_cache.device().device_id);
  const cudaStream_t stream = get_stream(state_cache.device());
  const int num_tokens = static_cast<int>(positions.size(0));
  if (num_tokens == 0) {
    return;
  }

  csa_indexer_cache_insert_fp8_kernel<<<num_tokens, kThreads, 0, stream>>>(
      static_cast<const float*>(state_cache.data_ptr()),
      static_cast<const int32_t*>(token_to_req_indices.data_ptr()),
      static_cast<const int64_t*>(positions.data_ptr()),
      static_cast<const int64_t*>(compressor_slot_mapping.data_ptr()),
      static_cast<const int32_t*>(block_table.data_ptr()),
      static_cast<const float*>(rms_norm_weight.data_ptr()),
      static_cast<const float*>(cos_sin_cache.data_ptr()),
      static_cast<uint8_t*>(kv_cache.data_ptr()),
      static_cast<const int64_t*>(kv_slot_mapping.data_ptr()),
      static_cast<float>(rms_norm_eps),
      num_tokens,
      static_cast<int>(compressor_block_size),
      state_cache.stride(0),
      state_cache.stride(1),
      static_cast<int>(block_table.size(1)),
      static_cast<int>(kv_cache_block_size),
      kv_cache.stride(0),
      cos_sin_cache.stride(0));

  cudaError_t status = cudaGetLastError();
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "deepseek_v4_csa_indexer_cache_insert_fp8_cuda failed: "
      << cudaGetErrorString(status);
}

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
    int64_t full_candidate_max_len) {
  CHECK_CUDA(positions);
  CHECK_CUDA(token_to_req_indices);
  CHECK_CUDA(block_table);
  CHECK_CUDA(topk_indices);
  CHECK_CUDA(swa_indices);
  CHECK_CUDA(swa_lens);
  CHECK_CUDA(extra_indices);
  CHECK_CUDA(extra_lens);
  CHECK_DIM(1, positions);
  CHECK_DIM(1, token_to_req_indices);
  CHECK_DIM(2, block_table);
  CHECK_DIM(2, topk_indices);
  CHECK_DIM(2, swa_indices);
  CHECK_DIM(1, swa_lens);
  CHECK_DIM(2, extra_indices);
  CHECK_DIM(1, extra_lens);

  TVM_FFI_ICHECK(positions.IsContiguous()) << "positions must be contiguous";
  TVM_FFI_ICHECK(token_to_req_indices.IsContiguous())
      << "token_to_req_indices must be contiguous";
  TVM_FFI_ICHECK(block_table.IsContiguous()) << "block_table must be contiguous";
  TVM_FFI_ICHECK(topk_indices.size(1) == 0 || topk_indices.IsContiguous())
      << "topk_indices must be contiguous";
  TVM_FFI_ICHECK(swa_indices.IsContiguous()) << "swa_indices must be contiguous";
  TVM_FFI_ICHECK(swa_lens.IsContiguous()) << "swa_lens must be contiguous";
  TVM_FFI_ICHECK(extra_indices.size(1) == 0 || extra_indices.IsContiguous())
      << "extra_indices must be contiguous";
  TVM_FFI_ICHECK(extra_lens.IsContiguous()) << "extra_lens must be contiguous";
  TVM_FFI_ICHECK(positions.dtype() == dl_int64) << "positions must be int64";
  TVM_FFI_ICHECK(token_to_req_indices.dtype() == dl_int32)
      << "token_to_req_indices must be int32";
  TVM_FFI_ICHECK(block_table.dtype() == dl_int32) << "block_table must be int32";
  TVM_FFI_ICHECK(topk_indices.dtype() == dl_int32)
      << "topk_indices must be int32";
  TVM_FFI_ICHECK(swa_indices.dtype() == dl_int32)
      << "swa_indices must be int32";
  TVM_FFI_ICHECK(swa_lens.dtype() == dl_int32) << "swa_lens must be int32";
  TVM_FFI_ICHECK(extra_indices.dtype() == dl_int32)
      << "extra_indices must be int32";
  TVM_FFI_ICHECK(extra_lens.dtype() == dl_int32) << "extra_lens must be int32";
  TVM_FFI_ICHECK(token_to_req_indices.size(0) >= positions.size(0))
      << "token_to_req_indices must cover positions";
  TVM_FFI_ICHECK(topk_indices.size(0) == positions.size(0))
      << "topk_indices token count must match positions";
  TVM_FFI_ICHECK(swa_indices.size(0) == positions.size(0))
      << "swa_indices token count must match positions";
  TVM_FFI_ICHECK(swa_lens.size(0) == positions.size(0))
      << "swa_lens token count must match positions";
  TVM_FFI_ICHECK(extra_indices.size(0) == positions.size(0))
      << "extra_indices token count must match positions";
  TVM_FFI_ICHECK(extra_lens.size(0) == positions.size(0))
      << "extra_lens token count must match positions";
  TVM_FFI_ICHECK(window_size > 0) << "window_size must be positive";
  TVM_FFI_ICHECK(swa_block_size > 0) << "swa_block_size must be positive";
  TVM_FFI_ICHECK(compress_ratio >= 1) << "compress_ratio must be positive";
  TVM_FFI_ICHECK(compressed_block_size > 0)
      << "compressed_block_size must be positive";

  cudaSetDevice(positions.device().device_id);
  const cudaStream_t stream = get_stream(positions.device());
  const int num_tokens = static_cast<int>(positions.size(0));
  if (num_tokens == 0) {
    return;
  }

  launch_decode_indices(
      static_cast<const int64_t*>(positions.data_ptr()),
      static_cast<const int32_t*>(token_to_req_indices.data_ptr()),
      static_cast<const int32_t*>(block_table.data_ptr()),
      static_cast<const int32_t*>(topk_indices.data_ptr()),
      static_cast<int32_t*>(swa_indices.data_ptr()),
      static_cast<int32_t*>(swa_lens.data_ptr()),
      static_cast<int32_t*>(extra_indices.data_ptr()),
      static_cast<int32_t*>(extra_lens.data_ptr()), num_tokens,
      static_cast<int>(block_table.size(1)), static_cast<int>(window_size),
      static_cast<int>(swa_indices.size(1)), static_cast<int>(swa_block_size),
      static_cast<int>(compress_ratio), static_cast<int>(extra_indices.size(1)),
      static_cast<int>(compressed_block_size),
      static_cast<int>(topk_indices.size(1)),
      static_cast<int>(full_candidate_max_len), stream);

  cudaError_t status = cudaGetLastError();
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "deepseek_v4_decode_indices_cuda failed: "
      << cudaGetErrorString(status);
}

void deepseek_v4_full_candidate_topk_cuda(
    TensorView positions,
    TensorView topk,
    int64_t compress_ratio) {
  CHECK_CUDA(positions);
  CHECK_CUDA(topk);
  CHECK_DIM(1, positions);
  CHECK_DIM(2, topk);

  TVM_FFI_ICHECK(positions.IsContiguous()) << "positions must be contiguous";
  TVM_FFI_ICHECK(topk.IsContiguous()) << "topk must be contiguous";
  TVM_FFI_ICHECK(positions.dtype() == dl_int64) << "positions must be int64";
  TVM_FFI_ICHECK(topk.dtype() == dl_int32) << "topk must be int32";
  TVM_FFI_ICHECK(topk.size(0) == positions.size(0))
      << "topk token count must match positions";
  TVM_FFI_ICHECK(compress_ratio > 0) << "compress_ratio must be positive";

  cudaSetDevice(positions.device().device_id);
  const cudaStream_t stream = get_stream(positions.device());
  const int num_tokens = static_cast<int>(positions.size(0));
  const int width = static_cast<int>(topk.size(1));
  if (num_tokens == 0 || width == 0) {
    return;
  }

  launch_full_candidate_topk(
      static_cast<const int64_t*>(positions.data_ptr()),
      static_cast<int32_t*>(topk.data_ptr()), num_tokens, width,
      static_cast<int>(compress_ratio), stream);

  cudaError_t status = cudaGetLastError();
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "deepseek_v4_full_candidate_topk_cuda failed: "
      << cudaGetErrorString(status);
}

void deepseek_v4_save_compressor_state_cuda(
    TensorView kv,
    TensorView score,
    TensorView ape,
    TensorView state_cache,
    TensorView slot_mapping,
    TensorView positions,
    int64_t block_size,
    int64_t compress_ratio) {
  CHECK_CUDA(kv);
  CHECK_CUDA(score);
  CHECK_CUDA(ape);
  CHECK_CUDA(state_cache);
  CHECK_CUDA(slot_mapping);
  CHECK_CUDA(positions);
  CHECK_DIM(2, kv);
  CHECK_DIM(2, score);
  CHECK_DIM(2, ape);
  CHECK_DIM(3, state_cache);
  CHECK_DIM(1, slot_mapping);
  CHECK_DIM(1, positions);

  TVM_FFI_ICHECK(kv.IsContiguous()) << "kv must be contiguous";
  TVM_FFI_ICHECK(score.IsContiguous()) << "score must be contiguous";
  TVM_FFI_ICHECK(ape.IsContiguous()) << "ape must be contiguous";
  TVM_FFI_ICHECK(state_cache.stride(2) == 1)
      << "state_cache last dim must be contiguous";
  TVM_FFI_ICHECK(slot_mapping.IsContiguous())
      << "slot_mapping must be contiguous";
  TVM_FFI_ICHECK(positions.IsContiguous()) << "positions must be contiguous";
  TVM_FFI_ICHECK(kv.dtype() == dl_float32) << "kv must be float32";
  TVM_FFI_ICHECK(score.dtype() == dl_float32) << "score must be float32";
  TVM_FFI_ICHECK(ape.dtype() == dl_float32) << "ape must be float32";
  TVM_FFI_ICHECK(state_cache.dtype() == dl_float32)
      << "state_cache must be float32";
  TVM_FFI_ICHECK(slot_mapping.dtype() == dl_int64)
      << "slot_mapping must be int64";
  TVM_FFI_ICHECK(positions.dtype() == dl_int64) << "positions must be int64";
  TVM_FFI_ICHECK(kv.size(0) == score.size(0))
      << "kv and score token counts must match";
  TVM_FFI_ICHECK(kv.size(1) == score.size(1))
      << "kv and score widths must match";
  TVM_FFI_ICHECK(slot_mapping.size(0) >= kv.size(0))
      << "slot_mapping must cover kv rows";
  TVM_FFI_ICHECK(positions.size(0) >= kv.size(0))
      << "positions must cover kv rows";
  TVM_FFI_ICHECK(ape.size(0) == compress_ratio)
      << "ape first dim must match compress_ratio";
  TVM_FFI_ICHECK(ape.size(1) == kv.size(1))
      << "ape width must match kv width";
  TVM_FFI_ICHECK(state_cache.size(2) == kv.size(1) * 2)
      << "state_cache last dim must be 2 * kv width";
  TVM_FFI_ICHECK(block_size == state_cache.size(1))
      << "block_size must match state_cache page size";
  TVM_FFI_ICHECK(block_size > 0) << "block_size must be positive";
  TVM_FFI_ICHECK(compress_ratio > 0) << "compress_ratio must be positive";

  cudaSetDevice(kv.device().device_id);
  const cudaStream_t stream = get_stream(kv.device());
  const int num_tokens = static_cast<int>(kv.size(0));
  if (num_tokens == 0) {
    return;
  }

  launch_save_compressor_state(
      static_cast<const float*>(kv.data_ptr()),
      static_cast<const float*>(score.data_ptr()),
      static_cast<const float*>(ape.data_ptr()),
      static_cast<float*>(state_cache.data_ptr()),
      static_cast<const int64_t*>(slot_mapping.data_ptr()),
      static_cast<const int64_t*>(positions.data_ptr()), num_tokens,
      static_cast<int>(kv.size(1)), static_cast<int>(block_size),
      state_cache.stride(0), state_cache.stride(1),
      static_cast<int>(compress_ratio), stream);

  cudaError_t status = cudaGetLastError();
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "deepseek_v4_save_compressor_state_cuda failed: "
      << cudaGetErrorString(status);
}

void fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
    TensorView q,
    TensorView kv,
    TensorView k_cache,
    TensorView slot_mapping,
    TensorView positions,
    TensorView cos_sin_cache,
    double rms_norm_eps,
    int64_t cache_block_size) {
  CHECK_CUDA(q);
  CHECK_CUDA(kv);
  CHECK_CUDA(k_cache);
  CHECK_CUDA(slot_mapping);
  CHECK_CUDA(positions);
  CHECK_CUDA(cos_sin_cache);
  CHECK_DIM(3, q);
  CHECK_DIM(2, kv);
  CHECK_DIM(2, k_cache);
  CHECK_DIM(1, slot_mapping);
  CHECK_DIM(1, positions);
  CHECK_DIM(2, cos_sin_cache);

  TVM_FFI_ICHECK(q.IsContiguous()) << "q must be contiguous";
  TVM_FFI_ICHECK(kv.IsContiguous()) << "kv must be contiguous";
  TVM_FFI_ICHECK(k_cache.stride(1) == 1) << "k_cache last dim must be contiguous";
  TVM_FFI_ICHECK(slot_mapping.IsContiguous()) << "slot_mapping must be contiguous";
  TVM_FFI_ICHECK(positions.IsContiguous()) << "positions must be contiguous";
  TVM_FFI_ICHECK(cos_sin_cache.IsContiguous()) << "cos_sin_cache must be contiguous";
  TVM_FFI_ICHECK(q.dtype() == kv.dtype()) << "q and kv dtype must match";
  TVM_FFI_ICHECK(k_cache.dtype() == dl_uint8) << "k_cache must be uint8";
  TVM_FFI_ICHECK(slot_mapping.dtype() == dl_int64) << "slot_mapping must be int64";
  TVM_FFI_ICHECK(positions.dtype() == dl_int64) << "positions must be int64";
  TVM_FFI_ICHECK(cos_sin_cache.dtype() == dl_float32)
      << "cos_sin_cache must be float32";
  TVM_FFI_ICHECK(q.size(2) == kHeadDim) << "q must have head_dim=512";
  TVM_FFI_ICHECK(kv.size(1) == kHeadDim) << "kv must have dim=512";
  TVM_FFI_ICHECK(kv.size(0) == q.size(0)) << "q and kv token counts must match";
  TVM_FFI_ICHECK(positions.size(0) == q.size(0))
      << "positions must cover all q rows";
  TVM_FFI_ICHECK(slot_mapping.size(0) <= q.size(0))
      << "slot_mapping cannot be longer than q";
  TVM_FFI_ICHECK(cos_sin_cache.size(1) == kRopeDim)
      << "cos_sin_cache must have width 64";
  TVM_FFI_ICHECK(cache_block_size > 0) << "cache_block_size must be positive";
  TVM_FFI_ICHECK(k_cache.size(1) >= cache_block_size * (kTokenDataBytes + kScaleBytesPerToken))
      << "k_cache block stride is too small for DeepSeek V4 SWA rows";

  cudaSetDevice(q.device().device_id);
  const cudaStream_t stream = get_stream(q.device());
  const int num_tokens_full = static_cast<int>(q.size(0));
  const int num_tokens_insert = static_cast<int>(slot_mapping.size(0));
  const int num_heads = static_cast<int>(q.size(1));
  const int64_t cache_block_stride = k_cache.stride(0);

  if (q.dtype() == dl_float16) {
    launch_fused_qnorm_rope_kv_insert<half>(
        static_cast<half*>(q.data_ptr()), static_cast<const half*>(kv.data_ptr()),
        static_cast<uint8_t*>(k_cache.data_ptr()),
        static_cast<const int64_t*>(slot_mapping.data_ptr()),
        static_cast<const int64_t*>(positions.data_ptr()),
        static_cast<const float*>(cos_sin_cache.data_ptr()),
        static_cast<float>(rms_norm_eps), num_tokens_full, num_tokens_insert,
        num_heads, static_cast<int>(cache_block_size), cache_block_stride, stream);
  } else if (q.dtype() == dl_bfloat16) {
    launch_fused_qnorm_rope_kv_insert<nv_bfloat16>(
        static_cast<nv_bfloat16*>(q.data_ptr()),
        static_cast<const nv_bfloat16*>(kv.data_ptr()),
        static_cast<uint8_t*>(k_cache.data_ptr()),
        static_cast<const int64_t*>(slot_mapping.data_ptr()),
        static_cast<const int64_t*>(positions.data_ptr()),
        static_cast<const float*>(cos_sin_cache.data_ptr()),
        static_cast<float>(rms_norm_eps), num_tokens_full, num_tokens_insert,
        num_heads, static_cast<int>(cache_block_size), cache_block_stride, stream);
  } else {
    TVM_FFI_ICHECK(false) << "q/kv dtype must be float16 or bfloat16";
  }

  cudaError_t status = cudaGetLastError();
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert failed: "
      << cudaGetErrorString(status);
}

void deepseek_v4_sparse_mla_cuda(
    TensorView q,
    TensorView kv,
    TensorView indices,
    TensorView topk_length,
    TensorView attn_sink,
    TensorView output,
    double sm_scale) {
  CHECK_CUDA(q);
  CHECK_CUDA(kv);
  CHECK_CUDA(indices);
  CHECK_CUDA(topk_length);
  CHECK_CUDA(attn_sink);
  CHECK_CUDA(output);
  CHECK_DIM(3, q);
  CHECK_DIM(2, kv);
  CHECK_DIM(2, indices);
  CHECK_DIM(1, topk_length);
  CHECK_DIM(1, attn_sink);
  CHECK_DIM(3, output);

  TVM_FFI_ICHECK(q.IsContiguous()) << "q must be contiguous";
  TVM_FFI_ICHECK(kv.IsContiguous()) << "kv must be contiguous";
  TVM_FFI_ICHECK(indices.IsContiguous()) << "indices must be contiguous";
  TVM_FFI_ICHECK(topk_length.IsContiguous()) << "topk_length must be contiguous";
  TVM_FFI_ICHECK(attn_sink.IsContiguous()) << "attn_sink must be contiguous";
  TVM_FFI_ICHECK(output.IsContiguous()) << "output must be contiguous";
  TVM_FFI_ICHECK(q.dtype() == kv.dtype()) << "q and kv dtype must match";
  TVM_FFI_ICHECK(output.dtype() == q.dtype()) << "output dtype must match q";
  TVM_FFI_ICHECK(indices.dtype() == dl_int32) << "indices must be int32";
  TVM_FFI_ICHECK(topk_length.dtype() == dl_int64) << "topk_length must be int64";
  TVM_FFI_ICHECK(attn_sink.dtype() == dl_float32) << "attn_sink must be float32";
  TVM_FFI_ICHECK(q.size(2) == kHeadDim) << "q must have head_dim=512";
  TVM_FFI_ICHECK(kv.size(1) == kHeadDim) << "kv must have dim=512";
  TVM_FFI_ICHECK(indices.size(0) == q.size(0))
      << "indices token count must match q";
  TVM_FFI_ICHECK(topk_length.size(0) == q.size(0))
      << "topk_length token count must match q";
  TVM_FFI_ICHECK(output.size(0) == q.size(0)) << "output tokens must match q";
  TVM_FFI_ICHECK(output.size(1) == q.size(1)) << "output heads must match q";
  TVM_FFI_ICHECK(output.size(2) == kHeadDim) << "output head_dim must be 512";

  cudaSetDevice(q.device().device_id);
  const cudaStream_t stream = get_stream(q.device());
  const int num_tokens = static_cast<int>(q.size(0));
  const int num_heads = static_cast<int>(q.size(1));
  const int topk_width = static_cast<int>(indices.size(1));
  const int attn_sink_count = static_cast<int>(attn_sink.size(0));

  if (q.dtype() == dl_float16) {
    launch_sparse_mla<half>(
        static_cast<const half*>(q.data_ptr()), static_cast<const half*>(kv.data_ptr()),
        static_cast<const int32_t*>(indices.data_ptr()),
        static_cast<const int64_t*>(topk_length.data_ptr()),
        static_cast<const float*>(attn_sink.data_ptr()),
        static_cast<half*>(output.data_ptr()), static_cast<float>(sm_scale),
        num_tokens, num_heads, topk_width, attn_sink_count, stream);
  } else if (q.dtype() == dl_bfloat16) {
    launch_sparse_mla<nv_bfloat16>(
        static_cast<const nv_bfloat16*>(q.data_ptr()),
        static_cast<const nv_bfloat16*>(kv.data_ptr()),
        static_cast<const int32_t*>(indices.data_ptr()),
        static_cast<const int64_t*>(topk_length.data_ptr()),
        static_cast<const float*>(attn_sink.data_ptr()),
        static_cast<nv_bfloat16*>(output.data_ptr()), static_cast<float>(sm_scale),
        num_tokens, num_heads, topk_width, attn_sink_count, stream);
  } else {
    TVM_FFI_ICHECK(false) << "q/kv dtype must be float16 or bfloat16";
  }

  cudaError_t status = cudaGetLastError();
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "deepseek_v4_sparse_mla_cuda failed: " << cudaGetErrorString(status);
}

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
    int64_t k_split) {
  CHECK_CUDA(q);
  CHECK_CUDA(swa_cache);
  CHECK_CUDA(swa_indices);
  CHECK_CUDA(swa_lens);
  CHECK_CUDA(compressed_cache);
  CHECK_CUDA(extra_indices);
  CHECK_CUDA(extra_lens);
  CHECK_CUDA(attn_sink);
  CHECK_CUDA(output);
  CHECK_DIM(3, q);
  CHECK_DIM(2, swa_cache);
  CHECK_DIM(2, swa_indices);
  CHECK_DIM(1, swa_lens);
  CHECK_DIM(2, compressed_cache);
  CHECK_DIM(2, extra_indices);
  CHECK_DIM(1, extra_lens);
  CHECK_DIM(1, attn_sink);
  CHECK_DIM(3, output);

  TVM_FFI_ICHECK(q.IsContiguous()) << "q must be contiguous";
  TVM_FFI_ICHECK(swa_cache.stride(1) == 1)
      << "swa_cache last dim must be contiguous";
  TVM_FFI_ICHECK(swa_indices.IsContiguous()) << "swa_indices must be contiguous";
  TVM_FFI_ICHECK(swa_lens.IsContiguous()) << "swa_lens must be contiguous";
  TVM_FFI_ICHECK(compressed_cache.stride(1) == 1)
      << "compressed_cache last dim must be contiguous";
  TVM_FFI_ICHECK(extra_indices.size(1) == 0 || extra_indices.IsContiguous())
      << "extra_indices must be contiguous";
  TVM_FFI_ICHECK(extra_lens.IsContiguous()) << "extra_lens must be contiguous";
  TVM_FFI_ICHECK(attn_sink.IsContiguous()) << "attn_sink must be contiguous";
  TVM_FFI_ICHECK(output.IsContiguous()) << "output must be contiguous";
  TVM_FFI_ICHECK(swa_cache.dtype() == dl_uint8) << "swa_cache must be uint8";
  TVM_FFI_ICHECK(compressed_cache.dtype() == dl_uint8)
      << "compressed_cache must be uint8";
  TVM_FFI_ICHECK(swa_indices.dtype() == dl_int32)
      << "swa_indices must be int32";
  TVM_FFI_ICHECK(extra_indices.dtype() == dl_int32)
      << "extra_indices must be int32";
  TVM_FFI_ICHECK(swa_lens.dtype() == dl_int64) << "swa_lens must be int64";
  TVM_FFI_ICHECK(extra_lens.dtype() == dl_int64) << "extra_lens must be int64";
  TVM_FFI_ICHECK(attn_sink.dtype() == dl_float32) << "attn_sink must be float32";
  TVM_FFI_ICHECK(output.dtype() == q.dtype()) << "output dtype must match q";
  TVM_FFI_ICHECK(q.size(2) == kHeadDim) << "q must have head_dim=512";
  TVM_FFI_ICHECK(output.size(0) == q.size(0)) << "output tokens must match q";
  TVM_FFI_ICHECK(output.size(1) == q.size(1)) << "output heads must match q";
  TVM_FFI_ICHECK(output.size(2) == kHeadDim) << "output head_dim must be 512";
  TVM_FFI_ICHECK(swa_indices.size(0) == q.size(0))
      << "swa_indices token count must match q";
  TVM_FFI_ICHECK(swa_lens.size(0) == q.size(0))
      << "swa_lens token count must match q";
  TVM_FFI_ICHECK(extra_indices.size(0) == q.size(0))
      << "extra_indices token count must match q";
  TVM_FFI_ICHECK(extra_lens.size(0) == q.size(0))
      << "extra_lens token count must match q";
  TVM_FFI_ICHECK(swa_block_size > 0) << "swa_block_size must be positive";
  TVM_FFI_ICHECK(compressed_block_size > 0)
      << "compressed_block_size must be positive";
  TVM_FFI_ICHECK(swa_cache.size(1) >=
                 swa_block_size * (kTokenDataBytes + kScaleBytesPerToken))
      << "swa_cache block stride is too small for DeepSeek V4 rows";
  TVM_FFI_ICHECK(
      compressed_cache.size(1) == 0 ||
      compressed_cache.size(1) >=
          compressed_block_size * (kTokenDataBytes + kScaleBytesPerToken))
      << "compressed_cache block stride is too small for DeepSeek V4 rows";

  cudaSetDevice(q.device().device_id);
  const cudaStream_t stream = get_stream(q.device());
  const int num_tokens = static_cast<int>(q.size(0));
  const int num_heads = static_cast<int>(q.size(1));
  const int swa_width = static_cast<int>(swa_indices.size(1));
  const int extra_width = static_cast<int>(extra_indices.size(1));
  const int attn_sink_count = static_cast<int>(attn_sink.size(0));
  const int64_t swa_cache_block_stride = swa_cache.stride(0);
  const int64_t compressed_cache_block_stride = compressed_cache.stride(0);

  // ``k_split`` opts the K-axis-split path on. Caller passes a non-zero
  // ``partials`` scratch buffer big enough for
  // ``num_tokens * num_heads * k_split * (kHeadDim + 2)`` floats. When
  // ``k_split <= 1`` we go through the existing single-block path and
  // ``partials`` is ignored (may be an empty tensor).
  const int k_split_value = static_cast<int>(k_split);
  TVM_FFI_ICHECK(k_split_value >= 1) << "k_split must be >= 1";
  TVM_FFI_ICHECK(k_split_value <= 16)
      << "k_split must be <= 16 (reduce kernel shared_rescales bound)";
  float* partials_ptr = nullptr;
  if (k_split_value > 1) {
    CHECK_CUDA(partials);
    CHECK_DEVICE(output, partials);
    TVM_FFI_ICHECK(partials.IsContiguous()) << "partials must be contiguous";
    TVM_FFI_ICHECK(partials.dtype() == dl_float32)
        << "partials must be float32";
    int64_t const required =
        static_cast<int64_t>(num_tokens) * num_heads * k_split_value *
        (static_cast<int64_t>(kHeadDim) + 2);
    TVM_FFI_ICHECK(partials.numel() >= required)
        << "partials scratch too small: have " << partials.numel()
        << " elements, need at least " << required;
    partials_ptr = static_cast<float*>(partials.data_ptr());
  }

  if (q.dtype() == dl_float16) {
    launch_sparse_mla_fp8_cache<half>(
        static_cast<const half*>(q.data_ptr()),
        static_cast<const uint8_t*>(swa_cache.data_ptr()),
        static_cast<const int32_t*>(swa_indices.data_ptr()),
        static_cast<const int64_t*>(swa_lens.data_ptr()),
        static_cast<const uint8_t*>(compressed_cache.data_ptr()),
        static_cast<const int32_t*>(extra_indices.data_ptr()),
        static_cast<const int64_t*>(extra_lens.data_ptr()),
        static_cast<const float*>(attn_sink.data_ptr()),
        static_cast<half*>(output.data_ptr()),
        partials_ptr,
        static_cast<float>(sm_scale),
        num_tokens, num_heads, swa_width, extra_width,
        static_cast<int>(swa_block_size),
        static_cast<int>(compressed_block_size), swa_cache_block_stride,
        compressed_cache_block_stride, attn_sink_count, online_softmax != 0,
        k_split_value, stream);
  } else if (q.dtype() == dl_bfloat16) {
    launch_sparse_mla_fp8_cache<nv_bfloat16>(
        static_cast<const nv_bfloat16*>(q.data_ptr()),
        static_cast<const uint8_t*>(swa_cache.data_ptr()),
        static_cast<const int32_t*>(swa_indices.data_ptr()),
        static_cast<const int64_t*>(swa_lens.data_ptr()),
        static_cast<const uint8_t*>(compressed_cache.data_ptr()),
        static_cast<const int32_t*>(extra_indices.data_ptr()),
        static_cast<const int64_t*>(extra_lens.data_ptr()),
        static_cast<const float*>(attn_sink.data_ptr()),
        static_cast<nv_bfloat16*>(output.data_ptr()),
        partials_ptr,
        static_cast<float>(sm_scale),
        num_tokens, num_heads, swa_width, extra_width,
        static_cast<int>(swa_block_size),
        static_cast<int>(compressed_block_size), swa_cache_block_stride,
        compressed_cache_block_stride, attn_sink_count, online_softmax != 0,
        k_split_value, stream);
  } else {
    TVM_FFI_ICHECK(false) << "q dtype must be float16 or bfloat16";
  }

  cudaError_t status = cudaGetLastError();
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "deepseek_v4_sparse_mla_fp8_cache_cuda failed: "
      << cudaGetErrorString(status);
}
