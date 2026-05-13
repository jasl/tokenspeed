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

// First native SM12x MXFP4 MoE path. It consumes canonical checkpoint layout:
//   w13=[E, 2I, H/2], w13_scale=[E, 2I, H/32]
//   w2 =[E, H,  I/2], w2_scale =[E, H,  I/32]
// and dequantizes MXFP4 on the fly, avoiding whole-weight swizzles at load time.

#include <algorithm>
#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "tvm_ffi_utils.h"

using tvm::ffi::TensorView;

namespace {

constexpr int kThreads = 256;
constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = kThreads / kWarpSize;
constexpr int kMxfp4Block = 32;
constexpr int kMaxWarpBlocks = 262144;

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

template <typename scalar_t>
__device__ __forceinline__ scalar_t float_to_scalar(float value);

template <>
__device__ __forceinline__ float float_to_scalar<float>(float value) {
  return value;
}

template <>
__device__ __forceinline__ half float_to_scalar<half>(float value) {
  return __float2half_rn(value);
}

template <>
__device__ __forceinline__ nv_bfloat16 float_to_scalar<nv_bfloat16>(
    float value) {
  return __float2bfloat16(value);
}

__device__ __forceinline__ float e2m1_to_float(uint8_t nibble) {
  float magnitude = 0.0f;
  switch (nibble & 0x7) {
    case 0x0:
      magnitude = 0.0f;
      break;
    case 0x1:
      magnitude = 0.5f;
      break;
    case 0x2:
      magnitude = 1.0f;
      break;
    case 0x3:
      magnitude = 1.5f;
      break;
    case 0x4:
      magnitude = 2.0f;
      break;
    case 0x5:
      magnitude = 3.0f;
      break;
    case 0x6:
      magnitude = 4.0f;
      break;
    case 0x7:
      magnitude = 6.0f;
      break;
  }
  return (nibble & 0x8) ? -magnitude : magnitude;
}

__device__ __forceinline__ float load_mxfp4_value(
    const uint8_t* __restrict__ packed,
    const uint8_t* __restrict__ scale,
    int64_t packed_row_offset,
    int64_t scale_row_offset,
    int k) {
  uint8_t const byte = packed[packed_row_offset + k / 2];
  uint8_t const nibble = (k & 1) ? ((byte >> 4) & 0x0F) : (byte & 0x0F);
  uint8_t const encoded_scale = scale[scale_row_offset + k / kMxfp4Block];
  return e2m1_to_float(nibble) * exp2f(static_cast<float>(encoded_scale) - 127.0f);
}

__device__ __forceinline__ int local_expert_id(
    int expert_id,
    int num_local_experts,
    int ep_rank,
    int ep_size) {
  if (expert_id < 0) {
    return -1;
  }
  if (ep_size <= 1) {
    return expert_id < num_local_experts ? expert_id : -1;
  }

  int const local_start = ep_rank * num_local_experts;
  int const local_end = local_start + num_local_experts;
  if (expert_id < local_start || expert_id >= local_end) {
    return -1;
  }
  return expert_id - local_start;
}

__device__ __forceinline__ float apply_swiglu(
    float gate,
    float up,
    bool use_swiglu,
    float swiglu_alpha,
    bool has_swiglu_alpha,
    float swiglu_limit,
    bool has_swiglu_limit,
    float swiglu_beta,
    bool has_swiglu_beta) {
  if (has_swiglu_limit) {
    gate = fminf(gate, swiglu_limit);
    up = fminf(fmaxf(up, -swiglu_limit), swiglu_limit);
  }

  float activated = 0.0f;
  if (use_swiglu && has_swiglu_alpha) {
    activated = gate / (1.0f + expf(-swiglu_alpha * gate));
  } else {
    activated = gate / (1.0f + expf(-gate));
  }

  float const up_term = has_swiglu_beta ? up + swiglu_beta : up;
  return activated * up_term;
}

__device__ __forceinline__ float warp_sum(float value) {
  unsigned int const mask = 0xFFFFFFFFu;
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    value += __shfl_down_sync(mask, value, offset);
  }
  return value;
}

__device__ __forceinline__ float quad_max(float value) {
  unsigned int const mask = 0xFFFFFFFFu;
  value = fmaxf(value, __shfl_xor_sync(mask, value, 1));
  value = fmaxf(value, __shfl_xor_sync(mask, value, 2));
  return value;
}

__device__ __forceinline__ float mxfp8_quant_exponent(float max_abs) {
  if (max_abs <= 0.0f) {
    return 0.0f;
  }
  float const safe = fmaxf(max_abs / 448.0f, 1.1754943508222875e-38f);
  float exponent = ceilf(log2f(safe));
  return fminf(fmaxf(exponent, -127.0f), 127.0f);
}

__device__ __forceinline__ void ldmatrix_x4_b16(
    uint32_t& d0,
    uint32_t& d1,
    uint32_t& d2,
    uint32_t& d3,
    void* smem) {
  asm volatile(
      "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      : "l"(__cvta_generic_to_shared(smem))
      : "memory");
}

__device__ __forceinline__ void ldmatrix_x2_b16(
    uint32_t& d0,
    uint32_t& d1,
    void* smem) {
  asm volatile(
      "ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n"
      : "=r"(d0), "=r"(d1)
      : "l"(__cvta_generic_to_shared(smem))
      : "memory");
}

__device__ __forceinline__ void ldmatrix_m8n16_x2_b4x16_p64(
    uint32_t& d0,
    uint32_t& d1,
    void* smem) {
  asm volatile(
      "ldmatrix.sync.aligned.m8n16.x2.shared.b8x16.b4x16_p64 {%0, %1}, [%2];\n"
      : "=r"(d0), "=r"(d1)
      : "l"(__cvta_generic_to_shared(smem))
      : "memory");
}

__device__ __forceinline__ void sm120_mxfp4_mxfp8_mma(
    float (&d)[4],
    const uint32_t (&a)[4],
    const uint32_t (&b)[2],
    uint8_t sfa,
    uint8_t sfb) {
  asm volatile(
      "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e2m1.e4m3.f32.ue8m0 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, "
      "{%10}, {%11, %12}, {%13}, {%14, %15};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]),
        "r"(b[1]), "r"(static_cast<uint32_t>(sfa)),
        "n"(static_cast<uint16_t>(0)), "n"(static_cast<uint16_t>(0)),
        "r"(static_cast<uint32_t>(sfb)), "n"(static_cast<uint16_t>(0)),
        "n"(static_cast<uint16_t>(0)));
}

__device__ __forceinline__ void sm120_fp8_mxfp4_mma(
    float (&d)[4],
    const uint32_t (&a)[4],
    const uint32_t (&b)[2],
    uint8_t sfa,
    uint8_t sfb) {
  asm volatile(
      "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e2m1.f32.ue8m0 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, "
      "{%10}, {%11, %12}, {%13}, {%14, %15};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]),
        "r"(b[1]), "r"(static_cast<uint32_t>(sfa)),
        "n"(static_cast<uint16_t>(0)), "n"(static_cast<uint16_t>(0)),
        "r"(static_cast<uint32_t>(sfb)), "n"(static_cast<uint16_t>(0)),
        "n"(static_cast<uint16_t>(0)));
}

__global__ void mxfp8_mxfp4_mma_tile_kernel(
    float* __restrict__ output,
    const uint8_t* __restrict__ activations,
    const uint8_t* __restrict__ activation_scale,
    const uint8_t* __restrict__ weight,
    const uint8_t* __restrict__ weight_scale) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200)
  __shared__ __align__(128) uint8_t smem_a[16 * 32];
  // .b4x16_p64 shared-memory layout: each 16 FP4 values occupies
  // 8 packed data bytes plus 8 pad bytes, so K=32 has a 32-byte row stride.
  __shared__ __align__(128) uint8_t smem_b[8 * 32];

  int const lane = threadIdx.x & (kWarpSize - 1);
  for (int i = lane; i < 16 * 32; i += kWarpSize) {
    smem_a[i] = activations[i];
  }
  for (int i = lane; i < 8 * 32; i += kWarpSize) {
    smem_b[i] = 0;
  }
  for (int i = lane; i < 8 * 16; i += kWarpSize) {
    int const row = i / 16;
    int const col = i % 16;
    int const group = col / 8;
    int const byte_in_group = col % 8;
    smem_b[row * 32 + group * 16 + byte_in_group] = weight[i];
  }
  __syncwarp();

  uint32_t a_frag[4];
  uint32_t b_frag[2];
  int const a_row = (lane & 7) + ((lane >> 3) & 1) * 8;
  int const a_col = (lane >> 4) * 16;
  ldmatrix_x4_b16(
      a_frag[0], a_frag[1], a_frag[2], a_frag[3],
      smem_a + a_row * 32 + a_col);

  int const b_row = lane & 7;
  int const b_col = ((lane >> 3) & 1) * 16;
  ldmatrix_m8n16_x2_b4x16_p64(
      b_frag[0], b_frag[1], smem_b + b_row * 32 + b_col);
  b_frag[0] <<= 2;
  b_frag[1] <<= 2;

  int const group_id = lane / 4;
  int const thread_id = lane % 4;
  uint8_t const sfa = activation_scale[group_id + (thread_id & 1) * 8];
  uint8_t const sfb = weight_scale[group_id];
  float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  sm120_fp8_mxfp4_mma(d, a_frag, b_frag, sfa, sfb);

  int const col = thread_id * 2;
  int const row0 = group_id;
  int const row1 = group_id + 8;
  output[row0 * 8 + col] = d[0];
  output[row0 * 8 + col + 1] = d[1];
  output[row1 * 8 + col] = d[2];
  output[row1 * 8 + col + 1] = d[3];
#else
  if (threadIdx.x == 0) {
    output[0] = 0.0f;
  }
#endif
}

__global__ void mxfp4_mxfp8_mma_tile_kernel(
    float* __restrict__ output,
    const uint8_t* __restrict__ weight,
    const uint8_t* __restrict__ weight_scale,
    const uint8_t* __restrict__ activations,
    const uint8_t* __restrict__ activation_scale) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200)
  __shared__ __align__(128) uint8_t smem_a[16 * 32];
  __shared__ __align__(128) uint8_t smem_b[8 * 32];

  int const lane = threadIdx.x & (kWarpSize - 1);
  for (int i = lane; i < 16 * 32; i += kWarpSize) {
    int const row = i / 32;
    int const col = i & 31;
    uint8_t const packed = weight[row * 16 + col / 2];
    uint8_t const nibble =
        (col & 1) ? ((packed >> 4) & 0x0F) : (packed & 0x0F);
    smem_a[i] = nibble << 2;
  }
  for (int i = lane; i < 8 * 32; i += kWarpSize) {
    smem_b[i] = activations[i];
  }
  __syncwarp();

  uint32_t a_frag[4];
  uint32_t b_frag[2];
  int const a_row = (lane & 7) + ((lane >> 3) & 1) * 8;
  int const a_col = (lane >> 4) * 16;
  ldmatrix_x4_b16(
      a_frag[0], a_frag[1], a_frag[2], a_frag[3],
      smem_a + a_row * 32 + a_col);

  int const b_row = lane & 7;
  int const b_col = ((lane >> 3) & 1) * 16;
  ldmatrix_x2_b16(b_frag[0], b_frag[1], smem_b + b_row * 32 + b_col);

  int const group_id = lane / 4;
  int const thread_id = lane % 4;
  uint8_t const sfa = weight_scale[group_id + (thread_id & 1) * 8];
  uint8_t const sfb = activation_scale[group_id];
  float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  sm120_mxfp4_mxfp8_mma(d, a_frag, b_frag, sfa, sfb);

  int const col = thread_id * 2;
  int const row0 = group_id;
  int const row1 = group_id + 8;
  output[row0 * 8 + col] = d[0];
  output[row0 * 8 + col + 1] = d[1];
  output[row1 * 8 + col] = d[2];
  output[row1 * 8 + col + 1] = d[3];
#else
  if (threadIdx.x == 0) {
    output[0] = 0.0f;
  }
#endif
}

__global__ void mxfp4_mxfp8_dense_kernel(
    float* __restrict__ output,
    const uint8_t* __restrict__ weight,
    const uint8_t* __restrict__ weight_scale,
    const uint8_t* __restrict__ activations,
    const uint8_t* __restrict__ activation_scale,
    int m_dim,
    int n_dim,
    int k_dim,
    int weight_packed_k,
    int weight_scale_k,
    int activation_scale_k) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200)
  __shared__ __align__(128) uint8_t smem_a[16 * 32];
  __shared__ __align__(128) uint8_t smem_b[8 * 32];

  int const lane = threadIdx.x & (kWarpSize - 1);
  int const n_base = blockIdx.x * 16;
  int const m_base = blockIdx.y * 8;
  float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  for (int k_block = 0; k_block < k_dim / 32; ++k_block) {
    for (int i = lane; i < 16 * 32; i += kWarpSize) {
      int const row = i / 32;
      int const col = i & 31;
      uint8_t const packed =
          weight[(n_base + row) * weight_packed_k + k_block * 16 + col / 2];
      uint8_t const nibble =
          (col & 1) ? ((packed >> 4) & 0x0F) : (packed & 0x0F);
      smem_a[i] = nibble << 2;
    }
    for (int i = lane; i < 8 * 32; i += kWarpSize) {
      int const row = i / 32;
      int const col = i & 31;
      smem_b[i] = activations[(m_base + row) * k_dim + k_block * 32 + col];
    }
    __syncwarp();

    uint32_t a_frag[4];
    uint32_t b_frag[2];
    int const a_row = (lane & 7) + ((lane >> 3) & 1) * 8;
    int const a_col = (lane >> 4) * 16;
    ldmatrix_x4_b16(
        a_frag[0], a_frag[1], a_frag[2], a_frag[3],
        smem_a + a_row * 32 + a_col);

    int const b_row = lane & 7;
    int const b_col = ((lane >> 3) & 1) * 16;
    ldmatrix_x2_b16(b_frag[0], b_frag[1], smem_b + b_row * 32 + b_col);

    int const group_id = lane / 4;
    int const thread_id = lane % 4;
    uint8_t const sfa =
        weight_scale[(n_base + group_id + (thread_id & 1) * 8) *
                         weight_scale_k +
                     k_block];
    uint8_t const sfb =
        activation_scale[(m_base + group_id) * activation_scale_k + k_block];
    sm120_mxfp4_mxfp8_mma(d, a_frag, b_frag, sfa, sfb);
    __syncwarp();
  }

  int const group_id = lane / 4;
  int const thread_id = lane % 4;
  int const row0 = n_base + group_id;
  int const row1 = n_base + group_id + 8;
  int const col = m_base + thread_id * 2;
  output[col * n_dim + row0] = d[0];
  output[(col + 1) * n_dim + row0] = d[1];
  output[col * n_dim + row1] = d[2];
  output[(col + 1) * n_dim + row1] = d[3];
#else
  if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
    output[0] = 0.0f;
  }
#endif
}

__global__ void mxfp8_mxfp4_dense_kernel(
    float* __restrict__ output,
    const uint8_t* __restrict__ activations,
    const uint8_t* __restrict__ activation_scale,
    const uint8_t* __restrict__ weight,
    const uint8_t* __restrict__ weight_scale,
    int m_dim,
    int n_dim,
    int k_dim,
    int activation_scale_k,
    int weight_packed_k,
    int weight_scale_k) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200)
  __shared__ __align__(128) uint8_t smem_a[16 * 32];
  __shared__ __align__(128) uint8_t smem_b[8 * 32];

  int const lane = threadIdx.x & (kWarpSize - 1);
  int const m_base = blockIdx.y * 16;
  int const n_base = blockIdx.x * 8;
  float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  for (int k_block = 0; k_block < k_dim / 32; ++k_block) {
    for (int i = lane; i < 16 * 32; i += kWarpSize) {
      int const row = i / 32;
      int const col = i % 32;
      smem_a[i] = activations[(m_base + row) * k_dim + k_block * 32 + col];
    }
    for (int i = lane; i < 8 * 32; i += kWarpSize) {
      smem_b[i] = 0;
    }
    for (int i = lane; i < 8 * 16; i += kWarpSize) {
      int const row = i / 16;
      int const col = i % 16;
      int const group = col / 8;
      int const byte_in_group = col % 8;
      smem_b[row * 32 + group * 16 + byte_in_group] =
          weight[(n_base + row) * weight_packed_k + k_block * 16 + col];
    }
    __syncwarp();

    uint32_t a_frag[4];
    uint32_t b_frag[2];
    int const a_row = (lane & 7) + ((lane >> 3) & 1) * 8;
    int const a_col = (lane >> 4) * 16;
    ldmatrix_x4_b16(
        a_frag[0], a_frag[1], a_frag[2], a_frag[3],
        smem_a + a_row * 32 + a_col);

    int const b_row = lane & 7;
    int const b_col = ((lane >> 3) & 1) * 16;
    ldmatrix_m8n16_x2_b4x16_p64(
        b_frag[0], b_frag[1], smem_b + b_row * 32 + b_col);
    b_frag[0] <<= 2;
    b_frag[1] <<= 2;

    int const group_id = lane / 4;
    int const thread_id = lane % 4;
    uint8_t const sfa =
        activation_scale[(m_base + group_id + (thread_id & 1) * 8) *
                             activation_scale_k +
                         k_block];
    uint8_t const sfb =
        weight_scale[(n_base + group_id) * weight_scale_k + k_block];
    sm120_fp8_mxfp4_mma(d, a_frag, b_frag, sfa, sfb);
    __syncwarp();
  }

  int const group_id = lane / 4;
  int const thread_id = lane % 4;
  int const col = n_base + thread_id * 2;
  int const row0 = m_base + group_id;
  int const row1 = m_base + group_id + 8;
  output[row0 * n_dim + col] = d[0];
  output[row0 * n_dim + col + 1] = d[1];
  output[row1 * n_dim + col] = d[2];
  output[row1 * n_dim + col + 1] = d[3];
#else
  if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
    output[0] = 0.0f;
  }
#endif
}

template <typename scalar_t>
__global__ void gate_activation_warp_kernel(
    float* __restrict__ intermediate,
    const scalar_t* __restrict__ hidden_states,
    const int* __restrict__ topk_ids,
    const uint8_t* __restrict__ w13_weight,
    const uint8_t* __restrict__ w13_scale,
    const float* __restrict__ w13_bias,
    bool has_w13_bias,
    int num_tokens,
    int top_k,
    int hidden_dim,
    int intermediate_size,
    int num_local_experts,
    int w13_packed_k,
    int w13_scale_k,
    int ep_rank,
    int ep_size,
    bool use_swiglu,
    float swiglu_alpha,
    bool has_swiglu_alpha,
    float swiglu_limit,
    bool has_swiglu_limit,
    float swiglu_beta,
    bool has_swiglu_beta) {
  int const lane = threadIdx.x & (kWarpSize - 1);
  int const warp_in_block = threadIdx.x / kWarpSize;
  int64_t const total =
      static_cast<int64_t>(num_tokens) * top_k * intermediate_size;
  int64_t const stride = static_cast<int64_t>(gridDim.x) * kWarpsPerBlock;

  for (int64_t idx = (static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock) +
                     warp_in_block;
       idx < total; idx += stride) {
    int const inter_idx = static_cast<int>(idx % intermediate_size);
    int64_t const tk = idx / intermediate_size;
    int const choice_idx = static_cast<int>(tk % top_k);
    int const token_idx = static_cast<int>(tk / top_k);
    int const expert_id = topk_ids[token_idx * top_k + choice_idx];
    int const local_id =
        local_expert_id(expert_id, num_local_experts, ep_rank, ep_size);

    if (local_id < 0) {
      if (lane == 0) {
        intermediate[idx] = 0.0f;
      }
      continue;
    }

    int const gate_row = inter_idx;
    int const up_row = intermediate_size + inter_idx;
    int64_t const gate_packed_offset =
        (static_cast<int64_t>(local_id) * (2 * intermediate_size) + gate_row) *
        w13_packed_k;
    int64_t const up_packed_offset =
        (static_cast<int64_t>(local_id) * (2 * intermediate_size) + up_row) *
        w13_packed_k;
    int64_t const gate_scale_offset =
        (static_cast<int64_t>(local_id) * (2 * intermediate_size) + gate_row) *
        w13_scale_k;
    int64_t const up_scale_offset =
        (static_cast<int64_t>(local_id) * (2 * intermediate_size) + up_row) *
        w13_scale_k;
    int64_t const hidden_offset = static_cast<int64_t>(token_idx) * hidden_dim;

    float gate = 0.0f;
    float up = 0.0f;
    for (int k = lane; k < hidden_dim; k += kWarpSize) {
      float const x = scalar_to_float(hidden_states[hidden_offset + k]);
      gate += x * load_mxfp4_value(
                      w13_weight, w13_scale, gate_packed_offset,
                      gate_scale_offset, k);
      up += x * load_mxfp4_value(
                  w13_weight, w13_scale, up_packed_offset, up_scale_offset, k);
    }

    gate = warp_sum(gate);
    up = warp_sum(up);
    if (lane == 0) {
      if (has_w13_bias) {
        gate += w13_bias[static_cast<int64_t>(local_id) *
                             (2 * intermediate_size) +
                         gate_row];
        up += w13_bias[static_cast<int64_t>(local_id) *
                           (2 * intermediate_size) +
                       up_row];
      }
      intermediate[idx] = apply_swiglu(
          gate, up, use_swiglu, swiglu_alpha, has_swiglu_alpha, swiglu_limit,
          has_swiglu_limit, swiglu_beta, has_swiglu_beta);
    }
  }
}

template <typename scalar_t>
__global__ void gate_activation_warp_ds4_decode_kernel(
    float* __restrict__ intermediate,
    const scalar_t* __restrict__ hidden_states,
    const int* __restrict__ topk_ids,
    const uint8_t* __restrict__ w13_weight,
    const uint8_t* __restrict__ w13_scale,
    const float* __restrict__ w13_bias,
    bool has_w13_bias,
    int num_tokens,
    int num_local_experts,
    int ep_rank,
    int ep_size,
    bool use_swiglu,
    float swiglu_alpha,
    bool has_swiglu_alpha,
    float swiglu_limit,
    bool has_swiglu_limit,
    float swiglu_beta,
    bool has_swiglu_beta) {
  constexpr int kDs4TopK = 6;
  constexpr int kDs4Hidden = 4096;
  constexpr int kDs4Intermediate = 2048;
  constexpr int kDs4W13PackedK = kDs4Hidden / 2;
  constexpr int kDs4W13ScaleK = kDs4Hidden / kMxfp4Block;

  int const lane = threadIdx.x & (kWarpSize - 1);
  int const warp_in_block = threadIdx.x / kWarpSize;
  int64_t const total =
      static_cast<int64_t>(num_tokens) * kDs4TopK * kDs4Intermediate;
  int64_t const stride = static_cast<int64_t>(gridDim.x) * kWarpsPerBlock;

  for (int64_t idx = (static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock) +
                     warp_in_block;
       idx < total; idx += stride) {
    int const inter_idx = static_cast<int>(idx & (kDs4Intermediate - 1));
    int64_t const tk = idx >> 11;
    int const choice_idx = static_cast<int>(tk % kDs4TopK);
    int const token_idx = static_cast<int>(tk / kDs4TopK);
    int const expert_id = topk_ids[token_idx * kDs4TopK + choice_idx];
    int const local_id =
        local_expert_id(expert_id, num_local_experts, ep_rank, ep_size);

    if (local_id < 0) {
      if (lane == 0) {
        intermediate[idx] = 0.0f;
      }
      continue;
    }

    int const gate_row = inter_idx;
    int const up_row = kDs4Intermediate + inter_idx;
    int64_t const gate_packed_offset =
        (static_cast<int64_t>(local_id) * (2 * kDs4Intermediate) + gate_row) *
        kDs4W13PackedK;
    int64_t const up_packed_offset =
        (static_cast<int64_t>(local_id) * (2 * kDs4Intermediate) + up_row) *
        kDs4W13PackedK;
    int64_t const gate_scale_offset =
        (static_cast<int64_t>(local_id) * (2 * kDs4Intermediate) + gate_row) *
        kDs4W13ScaleK;
    int64_t const up_scale_offset =
        (static_cast<int64_t>(local_id) * (2 * kDs4Intermediate) + up_row) *
        kDs4W13ScaleK;
    int64_t const hidden_offset = static_cast<int64_t>(token_idx) * kDs4Hidden;

    float gate = 0.0f;
    float up = 0.0f;
    for (int k = lane; k < kDs4Hidden; k += kWarpSize) {
      float const x = scalar_to_float(hidden_states[hidden_offset + k]);
      gate += x * load_mxfp4_value(
                      w13_weight, w13_scale, gate_packed_offset,
                      gate_scale_offset, k);
      up += x * load_mxfp4_value(
                  w13_weight, w13_scale, up_packed_offset, up_scale_offset, k);
    }

    gate = warp_sum(gate);
    up = warp_sum(up);
    if (lane == 0) {
      if (has_w13_bias) {
        gate += w13_bias[static_cast<int64_t>(local_id) *
                             (2 * kDs4Intermediate) +
                         gate_row];
        up += w13_bias[static_cast<int64_t>(local_id) *
                           (2 * kDs4Intermediate) +
                       up_row];
      }
      intermediate[idx] = apply_swiglu(
          gate, up, use_swiglu, swiglu_alpha, has_swiglu_alpha, swiglu_limit,
          has_swiglu_limit, swiglu_beta, has_swiglu_beta);
    }
  }
}

template <typename scalar_t>
__global__ void down_warp_kernel(
    scalar_t* __restrict__ output,
    const float* __restrict__ intermediate,
    const float* __restrict__ topk_weights,
    const int* __restrict__ topk_ids,
    const uint8_t* __restrict__ w2_weight,
    const uint8_t* __restrict__ w2_scale,
    const float* __restrict__ w2_bias,
    bool has_w2_bias,
    int num_tokens,
    int top_k,
    int hidden_dim,
    int intermediate_size,
    int num_local_experts,
    int w2_rows,
    int w2_packed_k,
    int w2_scale_k,
    int ep_rank,
    int ep_size) {
  int const lane = threadIdx.x & (kWarpSize - 1);
  int const warp_in_block = threadIdx.x / kWarpSize;
  int64_t const total = static_cast<int64_t>(num_tokens) * hidden_dim;
  int64_t const stride = static_cast<int64_t>(gridDim.x) * kWarpsPerBlock;

  for (int64_t idx = (static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock) +
                     warp_in_block;
       idx < total; idx += stride) {
    int const hidden_idx = static_cast<int>(idx % hidden_dim);
    int const token_idx = static_cast<int>(idx / hidden_dim);
    float acc = 0.0f;

    for (int choice_idx = 0; choice_idx < top_k; ++choice_idx) {
      int const expert_id = topk_ids[token_idx * top_k + choice_idx];
      int const local_id =
          local_expert_id(expert_id, num_local_experts, ep_rank, ep_size);
      if (local_id < 0) {
        continue;
      }

      int64_t const row_offset =
          (static_cast<int64_t>(local_id) * w2_rows + hidden_idx) * w2_packed_k;
      int64_t const scale_offset =
          (static_cast<int64_t>(local_id) * w2_rows + hidden_idx) * w2_scale_k;
      int64_t const intermediate_offset =
          (static_cast<int64_t>(token_idx) * top_k + choice_idx) *
          intermediate_size;

      float expert_out = 0.0f;
      for (int i = lane; i < intermediate_size; i += kWarpSize) {
        expert_out += intermediate[intermediate_offset + i] *
                      load_mxfp4_value(
                          w2_weight, w2_scale, row_offset, scale_offset, i);
      }
      expert_out = warp_sum(expert_out);
      if (lane == 0) {
        if (has_w2_bias) {
          expert_out +=
              w2_bias[static_cast<int64_t>(local_id) * w2_rows + hidden_idx];
        }
        acc += expert_out * topk_weights[token_idx * top_k + choice_idx];
      }
    }

    if (lane == 0) {
      output[idx] = float_to_scalar<scalar_t>(acc);
    }
  }
}

template <typename scalar_t>
__global__ void down_warp_ds4_decode_kernel(
    scalar_t* __restrict__ output,
    const float* __restrict__ intermediate,
    const float* __restrict__ topk_weights,
    const int* __restrict__ topk_ids,
    const uint8_t* __restrict__ w2_weight,
    const uint8_t* __restrict__ w2_scale,
    const float* __restrict__ w2_bias,
    bool has_w2_bias,
    int num_tokens,
    int num_local_experts,
    int ep_rank,
    int ep_size) {
  constexpr int kDs4TopK = 6;
  constexpr int kDs4Hidden = 4096;
  constexpr int kDs4Intermediate = 2048;
  constexpr int kDs4W2PackedK = kDs4Intermediate / 2;
  constexpr int kDs4W2ScaleK = kDs4Intermediate / kMxfp4Block;

  int const lane = threadIdx.x & (kWarpSize - 1);
  int const warp_in_block = threadIdx.x / kWarpSize;
  int64_t const total = static_cast<int64_t>(num_tokens) * kDs4Hidden;
  int64_t const stride = static_cast<int64_t>(gridDim.x) * kWarpsPerBlock;

  for (int64_t idx = (static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock) +
                     warp_in_block;
       idx < total; idx += stride) {
    int const hidden_idx = static_cast<int>(idx & (kDs4Hidden - 1));
    int const token_idx = static_cast<int>(idx >> 12);
    float acc = 0.0f;

#pragma unroll
    for (int choice_idx = 0; choice_idx < kDs4TopK; ++choice_idx) {
      int const expert_id = topk_ids[token_idx * kDs4TopK + choice_idx];
      int const local_id =
          local_expert_id(expert_id, num_local_experts, ep_rank, ep_size);
      if (local_id < 0) {
        continue;
      }

      int64_t const row_offset =
          (static_cast<int64_t>(local_id) * kDs4Hidden + hidden_idx) *
          kDs4W2PackedK;
      int64_t const scale_offset =
          (static_cast<int64_t>(local_id) * kDs4Hidden + hidden_idx) *
          kDs4W2ScaleK;
      int64_t const intermediate_offset =
          (static_cast<int64_t>(token_idx) * kDs4TopK + choice_idx) *
          kDs4Intermediate;

      float expert_out = 0.0f;
      for (int i = lane; i < kDs4Intermediate; i += kWarpSize) {
        expert_out += intermediate[intermediate_offset + i] *
                      load_mxfp4_value(
                          w2_weight, w2_scale, row_offset, scale_offset, i);
      }
      expert_out = warp_sum(expert_out);
      if (lane == 0) {
        if (has_w2_bias) {
          expert_out +=
              w2_bias[static_cast<int64_t>(local_id) * kDs4Hidden + hidden_idx];
        }
        acc += expert_out * topk_weights[token_idx * kDs4TopK + choice_idx];
      }
    }

    if (lane == 0) {
      output[idx] = float_to_scalar<scalar_t>(acc);
    }
  }
}

template <typename scalar_t>
__global__ void mxfp8_quantize_kernel(
    uint8_t* __restrict__ output,
    uint8_t* __restrict__ output_scale,
    const scalar_t* __restrict__ values,
    int rows,
    int hidden_dim,
    int scale_k) {
  int const lane = threadIdx.x;
  int const row = blockIdx.y;
  int const block = blockIdx.x;
  int const col = block * kMxfp4Block + lane;
  if (row >= rows || lane >= kMxfp4Block) {
    return;
  }

  int64_t const idx = static_cast<int64_t>(row) * hidden_dim + col;
  float const value = scalar_to_float<scalar_t>(values[idx]);
  float max_abs = fabsf(value);
  unsigned int const mask = 0xFFFFFFFFu;
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    max_abs = fmaxf(max_abs, __shfl_down_sync(mask, max_abs, offset));
  }
  max_abs = __shfl_sync(mask, max_abs, 0);

  float exponent = 0.0f;
  if (max_abs > 0.0f) {
    float const safe = fmaxf(max_abs / 448.0f, 1.1754943508222875e-38f);
    exponent = ceilf(log2f(safe));
    exponent = fminf(fmaxf(exponent, -127.0f), 127.0f);
  }
  if (lane == 0) {
    output_scale[row * scale_k + block] =
        static_cast<uint8_t>(static_cast<int>(exponent) + 127);
  }

  float const scaled = value / exp2f(exponent);
  reinterpret_cast<__nv_fp8_e4m3*>(output)[idx] = __nv_fp8_e4m3(scaled);
}

__global__ void swiglu_mxfp8_quantize_kernel(
    uint8_t* __restrict__ output,
    uint8_t* __restrict__ output_scale,
    const float* __restrict__ gate_up,
    const int* __restrict__ expert_ids,
    const float* __restrict__ w13_bias,
    int total_rows,
    int intermediate_size,
    int scale_k,
    int bias_stride,
    bool has_w13_bias,
    bool use_swiglu,
    float swiglu_alpha,
    bool has_swiglu_alpha,
    float swiglu_limit,
    bool has_swiglu_limit,
    float swiglu_beta,
    bool has_swiglu_beta) {
  int const lane = threadIdx.x;
  int const row = blockIdx.y;
  int const block = blockIdx.x;
  int const col = block * kMxfp4Block + lane;
  if (row >= total_rows || lane >= kMxfp4Block) {
    return;
  }

  int64_t const gate_up_base =
      static_cast<int64_t>(row) * (2 * static_cast<int64_t>(intermediate_size));
  float gate = gate_up[gate_up_base + col];
  float up = gate_up[gate_up_base + intermediate_size + col];
  if (has_w13_bias) {
    int const expert_id = expert_ids[row];
    int64_t const bias_base =
        static_cast<int64_t>(expert_id) * static_cast<int64_t>(bias_stride);
    gate += w13_bias[bias_base + col];
    up += w13_bias[bias_base + intermediate_size + col];
  }

  float const value = apply_swiglu(
      gate, up, use_swiglu, swiglu_alpha, has_swiglu_alpha, swiglu_limit,
      has_swiglu_limit, swiglu_beta, has_swiglu_beta);

  float max_abs = fabsf(value);
  unsigned int const mask = 0xFFFFFFFFu;
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    max_abs = fmaxf(max_abs, __shfl_down_sync(mask, max_abs, offset));
  }
  max_abs = __shfl_sync(mask, max_abs, 0);

  float exponent = 0.0f;
  if (max_abs > 0.0f) {
    float const safe = fmaxf(max_abs / 448.0f, 1.1754943508222875e-38f);
    exponent = ceilf(log2f(safe));
    exponent = fminf(fmaxf(exponent, -127.0f), 127.0f);
  }
  if (lane == 0) {
    output_scale[row * scale_k + block] =
        static_cast<uint8_t>(static_cast<int>(exponent) + 127);
  }

  float const scaled = value / exp2f(exponent);
  reinterpret_cast<__nv_fp8_e4m3*>(output)
      [static_cast<int64_t>(row) * intermediate_size + col] =
          __nv_fp8_e4m3(scaled);
}

int launch_blocks(int64_t total) {
  int64_t const blocks = (total + kThreads - 1) / kThreads;
  return static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(blocks, 65535)));
}

int launch_warp_blocks(int64_t total) {
  int64_t const blocks = (total + kWarpsPerBlock - 1) / kWarpsPerBlock;
  return static_cast<int>(
      std::max<int64_t>(1, std::min<int64_t>(blocks, kMaxWarpBlocks)));
}

template <typename scalar_t>
void launch_sm12x_mxfp4_moe_forward(
    TensorView output,
    TensorView intermediate,
    TensorView hidden_states,
    TensorView topk_weights,
    TensorView topk_ids,
    TensorView w13_weight,
    TensorView w13_scale,
    TensorView w2_weight,
    TensorView w2_scale,
    TensorView w13_bias,
    TensorView w2_bias,
    int ep_rank,
    int ep_size,
    bool use_swiglu,
    float swiglu_alpha,
    bool has_swiglu_alpha,
    float swiglu_limit,
    bool has_swiglu_limit,
    float swiglu_beta,
    bool has_swiglu_beta,
    cudaStream_t stream) {
  int const num_tokens = static_cast<int>(hidden_states.size(0));
  int const hidden_dim = static_cast<int>(hidden_states.size(1));
  int const top_k = static_cast<int>(topk_ids.size(1));
  int const num_local_experts = static_cast<int>(w13_weight.size(0));
  int const intermediate_size = static_cast<int>(w2_weight.size(2)) * 2;
  int const w13_packed_k = static_cast<int>(w13_weight.size(2));
  int const w13_scale_k = static_cast<int>(w13_scale.size(2));
  int const w2_rows = static_cast<int>(w2_weight.size(1));
  int const w2_packed_k = static_cast<int>(w2_weight.size(2));
  int const w2_scale_k = static_cast<int>(w2_scale.size(2));
  bool const has_w13_bias = w13_bias.numel() > 0;
  bool const has_w2_bias = w2_bias.numel() > 0;
  bool const use_ds4_decode_shape =
      hidden_dim == 4096 && intermediate_size == 2048 && top_k == 6 &&
      w13_packed_k == 2048 && w13_scale_k == 128 && w2_rows == 4096 &&
      w2_packed_k == 1024 && w2_scale_k == 64;

  int64_t const gate_total =
      static_cast<int64_t>(num_tokens) * top_k * intermediate_size;
  if (use_ds4_decode_shape) {
    gate_activation_warp_ds4_decode_kernel<scalar_t>
        <<<launch_warp_blocks(gate_total), kThreads, 0, stream>>>(
            static_cast<float*>(intermediate.data_ptr()),
            static_cast<const scalar_t*>(hidden_states.data_ptr()),
            static_cast<const int*>(topk_ids.data_ptr()),
            static_cast<const uint8_t*>(w13_weight.data_ptr()),
            static_cast<const uint8_t*>(w13_scale.data_ptr()),
            static_cast<const float*>(w13_bias.data_ptr()),
            has_w13_bias,
            num_tokens,
            num_local_experts,
            ep_rank,
            ep_size,
            use_swiglu,
            swiglu_alpha,
            has_swiglu_alpha,
            swiglu_limit,
            has_swiglu_limit,
            swiglu_beta,
            has_swiglu_beta);
  } else {
    gate_activation_warp_kernel<scalar_t>
        <<<launch_warp_blocks(gate_total), kThreads, 0, stream>>>(
            static_cast<float*>(intermediate.data_ptr()),
            static_cast<const scalar_t*>(hidden_states.data_ptr()),
            static_cast<const int*>(topk_ids.data_ptr()),
            static_cast<const uint8_t*>(w13_weight.data_ptr()),
            static_cast<const uint8_t*>(w13_scale.data_ptr()),
            static_cast<const float*>(w13_bias.data_ptr()),
            has_w13_bias,
            num_tokens,
            top_k,
            hidden_dim,
            intermediate_size,
            num_local_experts,
            w13_packed_k,
            w13_scale_k,
            ep_rank,
            ep_size,
            use_swiglu,
            swiglu_alpha,
            has_swiglu_alpha,
            swiglu_limit,
            has_swiglu_limit,
            swiglu_beta,
            has_swiglu_beta);
  }

  cudaError_t status = cudaGetLastError();
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "sm12x_mxfp4 gate activation launch failed: "
      << cudaGetErrorString(status);

  int64_t const down_total = static_cast<int64_t>(num_tokens) * hidden_dim;
  if (use_ds4_decode_shape) {
    down_warp_ds4_decode_kernel<scalar_t>
        <<<launch_warp_blocks(down_total), kThreads, 0, stream>>>(
            static_cast<scalar_t*>(output.data_ptr()),
            static_cast<const float*>(intermediate.data_ptr()),
            static_cast<const float*>(topk_weights.data_ptr()),
            static_cast<const int*>(topk_ids.data_ptr()),
            static_cast<const uint8_t*>(w2_weight.data_ptr()),
            static_cast<const uint8_t*>(w2_scale.data_ptr()),
            static_cast<const float*>(w2_bias.data_ptr()),
            has_w2_bias,
            num_tokens,
            num_local_experts,
            ep_rank,
            ep_size);
  } else {
    down_warp_kernel<scalar_t><<<launch_warp_blocks(down_total), kThreads, 0, stream>>>(
        static_cast<scalar_t*>(output.data_ptr()),
        static_cast<const float*>(intermediate.data_ptr()),
        static_cast<const float*>(topk_weights.data_ptr()),
        static_cast<const int*>(topk_ids.data_ptr()),
        static_cast<const uint8_t*>(w2_weight.data_ptr()),
        static_cast<const uint8_t*>(w2_scale.data_ptr()),
        static_cast<const float*>(w2_bias.data_ptr()),
        has_w2_bias,
        num_tokens,
        top_k,
        hidden_dim,
        intermediate_size,
        num_local_experts,
        w2_rows,
        w2_packed_k,
        w2_scale_k,
        ep_rank,
        ep_size);
  }

  status = cudaGetLastError();
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "sm12x_mxfp4 down projection launch failed: "
      << cudaGetErrorString(status);
}

// Tensorcore W13 GEMM for MoE decode: per (token, top_k) pair compute
// gate_up = w13[expert] @ hidden_fp8[token], using the SM120 mxf4xfp8
// block-scaled MMA. Bias and SwiGLU live in the downstream
// `sm12x_mxfp4_swiglu_mxfp8_quantize` helper -- this kernel is a pure GEMM.
//
// Grid: (num_pairs, gate_up_dim / 16). One pair = (token_idx, top_k_idx).
// Block: 32 threads (single warp). The MMA shape is m16n8k32; here m=16 maps
// to the weight's output-channel direction (16 outputs per block) and n=8
// maps to the batch (activation) direction. Each pair has batch=1, so the
// activation tile is padded with zeros for n=1..7. Compute the full m16xn8
// MMA anyway -- the tensor unit is fast enough that the 7/8 column waste is
// invisible relative to the weight bandwidth cost. Only thread_id==0 owns
// the m=0 (real) outputs and writes them to global memory.
//
// gate_up output is fp32 in [num_tokens, top_k, gate_up_dim] layout so the
// downstream SwiGLU+quantize kernel can consume it without a transpose.
__global__ void mxfp4_moe_w13_tensorcore_kernel(
    float* __restrict__ gate_up,
    const uint8_t* __restrict__ hidden_fp8,
    const uint8_t* __restrict__ hidden_scale,
    const int* __restrict__ topk_ids,
    const uint8_t* __restrict__ w13_weight,
    const uint8_t* __restrict__ w13_scale,
    int num_tokens,
    int top_k,
    int hidden,
    int gate_up_dim,
    int num_local_experts,
    int weight_packed_k,
    int weight_scale_k,
    int hidden_scale_k,
    int ep_rank) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200)
  int const pair_idx = blockIdx.x;
  int const n_tile_idx = blockIdx.y;
  int const lane = threadIdx.x & (kWarpSize - 1);
  if (pair_idx >= num_tokens * top_k) {
    return;
  }
  int const token_idx = pair_idx / top_k;
  int const top_k_idx = pair_idx % top_k;
  int const n_base = n_tile_idx * 16;
  int64_t const out_base =
      (static_cast<int64_t>(token_idx) * top_k + top_k_idx) *
      static_cast<int64_t>(gate_up_dim);

  int const global_expert = topk_ids[pair_idx];
  int const expert_lo = ep_rank * num_local_experts;
  int const expert_hi = expert_lo + num_local_experts;
  bool const local_expert =
      (global_expert >= expert_lo) && (global_expert < expert_hi);

  int const group_id = lane / 4;
  int const thread_id = lane % 4;

  if (!local_expert) {
    // Out-of-rank pair: contribute zero so the downstream SwiGLU+quantize and
    // W2 see a stable buffer.
    if (thread_id == 0) {
      int const row0 = n_base + group_id;
      int const row1 = n_base + group_id + 8;
      gate_up[out_base + row0] = 0.0f;
      gate_up[out_base + row1] = 0.0f;
    }
    return;
  }
  int const local_expert_id = global_expert - expert_lo;

  __shared__ __align__(128) uint8_t smem_a[16 * 32];
  __shared__ __align__(128) uint8_t smem_b[8 * 32];

  float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  int64_t const expert_packed_base =
      static_cast<int64_t>(local_expert_id) * gate_up_dim;
  int64_t const expert_scale_base = expert_packed_base;

  for (int k_block = 0; k_block < hidden / 32; ++k_block) {
    // Load weight [16 N rows, 32 K] from packed FP4 -> nibble-expanded fp8.
    for (int i = lane; i < 16 * 32; i += kWarpSize) {
      int const row = i / 32;
      int const col = i & 31;
      uint8_t const packed =
          w13_weight[(expert_packed_base + n_base + row) * weight_packed_k +
                     k_block * 16 + col / 2];
      uint8_t const nibble =
          (col & 1) ? ((packed >> 4) & 0x0F) : (packed & 0x0F);
      smem_a[i] = nibble << 2;
    }
    // Load activation [8 M rows, 32 K]; only M=0 has real fp8 data.
    int64_t const a_base =
        static_cast<int64_t>(token_idx) * hidden +
        static_cast<int64_t>(k_block) * 32;
    for (int i = lane; i < 8 * 32; i += kWarpSize) {
      int const row = i / 32;
      int const col = i & 31;
      smem_b[i] = (row == 0) ? hidden_fp8[a_base + col] : 0;
    }
    __syncwarp();

    uint32_t a_frag[4];
    uint32_t b_frag[2];
    int const a_row = (lane & 7) + ((lane >> 3) & 1) * 8;
    int const a_col = (lane >> 4) * 16;
    ldmatrix_x4_b16(
        a_frag[0], a_frag[1], a_frag[2], a_frag[3],
        smem_a + a_row * 32 + a_col);

    int const b_row = lane & 7;
    int const b_col = ((lane >> 3) & 1) * 16;
    ldmatrix_x2_b16(b_frag[0], b_frag[1], smem_b + b_row * 32 + b_col);

    uint8_t const sfa =
        w13_scale[(expert_scale_base + n_base + group_id +
                   (thread_id & 1) * 8) *
                      weight_scale_k +
                  k_block];
    // Replicate the real token scale to all m-rows; m=1..7 rows have zero
    // activations so the scale value does not affect their MMA contribution
    // and a broadcast load is cheaper than a per-row gather.
    uint8_t const sfb = hidden_scale[token_idx * hidden_scale_k + k_block];
    sm120_mxfp4_mxfp8_mma(d, a_frag, b_frag, sfa, sfb);
    __syncwarp();
  }

  // Only thread_id==0 owns the m=0 outputs; other thread_ids hold m=1..7
  // which are zero by construction.
  if (thread_id == 0) {
    int const row0 = n_base + group_id;
    int const row1 = n_base + group_id + 8;
    gate_up[out_base + row0] = d[0];
    gate_up[out_base + row1] = d[2];
  }
#else
  if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
    gate_up[0] = 0.0f;
  }
#endif
}

// Tensorcore W2 GEMM + weighted-scatter for MoE decode.
//
// Companion to ``mxfp4_moe_w13_tensorcore_kernel``: per (token, top_k) pair,
// run the SM120 mxf4xfp8 block-scaled m16n8k32 MMA over the selected expert's
// W2 weights and the post-SwiGLU FP8 intermediate, then write the per-pair
// output to a scratch buffer. A separate weighted-reduce kernel finishes the
// MoE step by summing the top_k contributions with their topk_weights into
// the final per-token hidden_dim output.
//
// Grid: (num_pairs, hidden / 16). Block: 32 threads. The activation tile is
// padded M=1 -> M=8 with zeros and broadcasts the per-pair ue8m0 scale.
// Out-of-rank pairs (filtered by ep_rank) write zeros so the downstream
// reduce sees a stable buffer.
__global__ void mxfp4_moe_w2_tensorcore_kernel(
    float* __restrict__ per_pair_out,
    const uint8_t* __restrict__ intermediate_fp8,
    const uint8_t* __restrict__ intermediate_scale,
    const int* __restrict__ topk_ids,
    const uint8_t* __restrict__ w2_weight,
    const uint8_t* __restrict__ w2_scale,
    const float* __restrict__ w2_bias,
    int num_tokens,
    int top_k,
    int intermediate,
    int hidden,
    int num_local_experts,
    int weight_packed_k,
    int weight_scale_k,
    int intermediate_scale_k,
    int ep_rank) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200)
  int const pair_idx = blockIdx.x;
  int const n_tile_idx = blockIdx.y;
  int const lane = threadIdx.x & (kWarpSize - 1);
  if (pair_idx >= num_tokens * top_k) {
    return;
  }
  int const n_base = n_tile_idx * 16;
  int64_t const out_base =
      static_cast<int64_t>(pair_idx) * static_cast<int64_t>(hidden);

  int const global_expert = topk_ids[pair_idx];
  int const expert_lo = ep_rank * num_local_experts;
  int const expert_hi = expert_lo + num_local_experts;
  bool const local_expert =
      (global_expert >= expert_lo) && (global_expert < expert_hi);

  int const group_id = lane / 4;
  int const thread_id = lane % 4;

  if (!local_expert) {
    if (thread_id == 0) {
      int const row0 = n_base + group_id;
      int const row1 = n_base + group_id + 8;
      per_pair_out[out_base + row0] = 0.0f;
      per_pair_out[out_base + row1] = 0.0f;
    }
    return;
  }
  int const local_expert_id = global_expert - expert_lo;

  __shared__ __align__(128) uint8_t smem_a[16 * 32];
  __shared__ __align__(128) uint8_t smem_b[8 * 32];

  float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  int64_t const expert_packed_base =
      static_cast<int64_t>(local_expert_id) * hidden;
  int64_t const expert_scale_base = expert_packed_base;

  for (int k_block = 0; k_block < intermediate / 32; ++k_block) {
    for (int i = lane; i < 16 * 32; i += kWarpSize) {
      int const row = i / 32;
      int const col = i & 31;
      uint8_t const packed =
          w2_weight[(expert_packed_base + n_base + row) * weight_packed_k +
                    k_block * 16 + col / 2];
      uint8_t const nibble =
          (col & 1) ? ((packed >> 4) & 0x0F) : (packed & 0x0F);
      smem_a[i] = nibble << 2;
    }
    int64_t const a_base =
        static_cast<int64_t>(pair_idx) * intermediate +
        static_cast<int64_t>(k_block) * 32;
    for (int i = lane; i < 8 * 32; i += kWarpSize) {
      int const row = i / 32;
      int const col = i & 31;
      smem_b[i] = (row == 0) ? intermediate_fp8[a_base + col] : 0;
    }
    __syncwarp();

    uint32_t a_frag[4];
    uint32_t b_frag[2];
    int const a_row = (lane & 7) + ((lane >> 3) & 1) * 8;
    int const a_col = (lane >> 4) * 16;
    ldmatrix_x4_b16(
        a_frag[0], a_frag[1], a_frag[2], a_frag[3],
        smem_a + a_row * 32 + a_col);

    int const b_row = lane & 7;
    int const b_col = ((lane >> 3) & 1) * 16;
    ldmatrix_x2_b16(b_frag[0], b_frag[1], smem_b + b_row * 32 + b_col);

    uint8_t const sfa =
        w2_scale[(expert_scale_base + n_base + group_id +
                  (thread_id & 1) * 8) *
                     weight_scale_k +
                 k_block];
    uint8_t const sfb =
        intermediate_scale[pair_idx * intermediate_scale_k + k_block];
    sm120_mxfp4_mxfp8_mma(d, a_frag, b_frag, sfa, sfb);
    __syncwarp();
  }

  if (thread_id == 0) {
    int const row0 = n_base + group_id;
    int const row1 = n_base + group_id + 8;
    // T3-alpha W2 bias fuse: add per-expert bias at the final write.
    // ``w2_bias`` is ``[num_local_experts, hidden]`` (float32) when present,
    // ``nullptr`` when the caller has signalled ``w2_bias is None``. The
    // bias is per-(local_expert, channel), so we read it at the same
    // (local_expert_id, row) coordinates we already have in scope.
    float const bias0 =
        (w2_bias == nullptr)
            ? 0.0f
            : w2_bias[static_cast<int64_t>(local_expert_id) * hidden + row0];
    float const bias1 =
        (w2_bias == nullptr)
            ? 0.0f
            : w2_bias[static_cast<int64_t>(local_expert_id) * hidden + row1];
    per_pair_out[out_base + row0] = d[0] + bias0;
    per_pair_out[out_base + row1] = d[2] + bias1;
  }
#else
  if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
    per_pair_out[0] = 0.0f;
  }
#endif
}

// Weighted reduce: sum top_k per-pair contributions (each is a row of length
// `hidden`) into the per-token output, applying topk_weights. The reduce is
// trivial enough that one warp covers a 32-wide stripe of channels per token;
// the grid runs num_tokens * cdiv(hidden, kReduceTile) blocks.
template <typename scalar_t>
__global__ void moe_weighted_reduce_kernel(
    scalar_t* __restrict__ output,
    const float* __restrict__ per_pair,
    const float* __restrict__ topk_weights,
    int num_tokens,
    int top_k,
    int hidden) {
  int const token_idx = blockIdx.x;
  int const stripe = blockIdx.y;
  int const lane = threadIdx.x & (kWarpSize - 1);
  int const channel = stripe * kWarpSize + lane;
  if (token_idx >= num_tokens || channel >= hidden) {
    return;
  }
  float accum = 0.0f;
  int64_t const pair_base =
      static_cast<int64_t>(token_idx) * top_k * hidden;
  for (int k = 0; k < top_k; ++k) {
    float const w = topk_weights[token_idx * top_k + k];
    float const v = per_pair[pair_base + static_cast<int64_t>(k) * hidden +
                              channel];
    accum += w * v;
  }
  output[static_cast<int64_t>(token_idx) * hidden + channel] =
      static_cast<scalar_t>(accum);
}

void launch_sm12x_mxfp4_moe_w13_tensorcore(
    TensorView gate_up,
    TensorView hidden_fp8,
    TensorView hidden_scale,
    TensorView topk_ids,
    TensorView w13_weight,
    TensorView w13_scale,
    int ep_rank,
    cudaStream_t stream) {
  int const num_tokens = static_cast<int>(hidden_fp8.size(0));
  int const top_k = static_cast<int>(topk_ids.size(1));
  int const hidden = static_cast<int>(hidden_fp8.size(1));
  int const num_local_experts = static_cast<int>(w13_weight.size(0));
  int const gate_up_dim = static_cast<int>(w13_weight.size(1));
  int const weight_packed_k = static_cast<int>(w13_weight.size(2));
  int const weight_scale_k = static_cast<int>(w13_scale.size(2));
  int const hidden_scale_k = static_cast<int>(hidden_scale.size(1));

  if (num_tokens == 0 || top_k == 0 || gate_up_dim == 0) {
    return;
  }

  dim3 const grid(num_tokens * top_k, gate_up_dim / 16);
  mxfp4_moe_w13_tensorcore_kernel<<<grid, kWarpSize, 0, stream>>>(
      static_cast<float*>(gate_up.data_ptr()),
      static_cast<const uint8_t*>(hidden_fp8.data_ptr()),
      static_cast<const uint8_t*>(hidden_scale.data_ptr()),
      static_cast<const int*>(topk_ids.data_ptr()),
      static_cast<const uint8_t*>(w13_weight.data_ptr()),
      static_cast<const uint8_t*>(w13_scale.data_ptr()),
      num_tokens,
      top_k,
      hidden,
      gate_up_dim,
      num_local_experts,
      weight_packed_k,
      weight_scale_k,
      hidden_scale_k,
      ep_rank);

  cudaError_t const status = cudaGetLastError();
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "sm12x_mxfp4_moe_w13_tensorcore launch failed: "
      << cudaGetErrorString(status);
}

void launch_sm12x_mxfp4_moe_w2_tensorcore(
    TensorView per_pair_out,
    TensorView intermediate_fp8,
    TensorView intermediate_scale,
    TensorView topk_ids,
    TensorView w2_weight,
    TensorView w2_scale,
    const float* w2_bias,
    int ep_rank,
    cudaStream_t stream) {
  int const num_tokens = static_cast<int>(topk_ids.size(0));
  int const top_k = static_cast<int>(topk_ids.size(1));
  int const num_pairs = num_tokens * top_k;
  int const intermediate = static_cast<int>(intermediate_fp8.size(1));
  int const num_local_experts = static_cast<int>(w2_weight.size(0));
  int const hidden = static_cast<int>(w2_weight.size(1));
  int const weight_packed_k = static_cast<int>(w2_weight.size(2));
  int const weight_scale_k = static_cast<int>(w2_scale.size(2));
  int const intermediate_scale_k = static_cast<int>(intermediate_scale.size(1));

  if (num_pairs == 0 || hidden == 0) {
    return;
  }

  dim3 const grid(num_pairs, hidden / 16);
  mxfp4_moe_w2_tensorcore_kernel<<<grid, kWarpSize, 0, stream>>>(
      static_cast<float*>(per_pair_out.data_ptr()),
      static_cast<const uint8_t*>(intermediate_fp8.data_ptr()),
      static_cast<const uint8_t*>(intermediate_scale.data_ptr()),
      static_cast<const int*>(topk_ids.data_ptr()),
      static_cast<const uint8_t*>(w2_weight.data_ptr()),
      static_cast<const uint8_t*>(w2_scale.data_ptr()),
      w2_bias,
      num_tokens,
      top_k,
      intermediate,
      hidden,
      num_local_experts,
      weight_packed_k,
      weight_scale_k,
      intermediate_scale_k,
      ep_rank);

  cudaError_t const status = cudaGetLastError();
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "sm12x_mxfp4_moe_w2_tensorcore launch failed: "
      << cudaGetErrorString(status);
}

template <typename scalar_t>
void launch_moe_weighted_reduce(
    TensorView output,
    TensorView per_pair,
    TensorView topk_weights,
    cudaStream_t stream) {
  int const num_tokens = static_cast<int>(per_pair.size(0));
  int const top_k = static_cast<int>(per_pair.size(1));
  int const hidden = static_cast<int>(per_pair.size(2));
  if (num_tokens == 0 || hidden == 0) {
    return;
  }
  int const stripes = (hidden + kWarpSize - 1) / kWarpSize;
  dim3 const grid(num_tokens, stripes);
  moe_weighted_reduce_kernel<scalar_t><<<grid, kWarpSize, 0, stream>>>(
      static_cast<scalar_t*>(output.data_ptr()),
      static_cast<const float*>(per_pair.data_ptr()),
      static_cast<const float*>(topk_weights.data_ptr()),
      num_tokens,
      top_k,
      hidden);
  cudaError_t const status = cudaGetLastError();
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "sm12x_mxfp4_moe_weighted_reduce launch failed: "
      << cudaGetErrorString(status);
}

}  // namespace

template <typename scalar_t>
void launch_sm12x_mxfp4_mxfp8_quantize(
    TensorView output,
    TensorView output_scale,
    TensorView values,
    cudaStream_t stream) {
  int const rows = static_cast<int>(values.size(0));
  int const hidden_dim = static_cast<int>(values.size(1));
  if (rows == 0) {
    return;
  }
  dim3 const grid(hidden_dim / kMxfp4Block, rows);
  mxfp8_quantize_kernel<scalar_t><<<grid, kWarpSize, 0, stream>>>(
      static_cast<uint8_t*>(output.data_ptr()),
      static_cast<uint8_t*>(output_scale.data_ptr()),
      static_cast<const scalar_t*>(values.data_ptr()),
      rows,
      hidden_dim,
      static_cast<int>(output_scale.size(1)));
  cudaError_t const status = cudaGetLastError();
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "sm12x_mxfp4_mxfp8_quantize launch failed: "
      << cudaGetErrorString(status);
}

void sm12x_mxfp4_mxfp8_quantize(
    TensorView output,
    TensorView output_scale,
    TensorView values) {
  CHECK_INPUT(output);
  CHECK_INPUT(output_scale);
  CHECK_INPUT(values);
  CHECK_DEVICE(output, output_scale);
  CHECK_DEVICE(output, values);

  CHECK_DIM(2, output);
  CHECK_DIM(2, output_scale);
  CHECK_DIM(2, values);
  TVM_FFI_ICHECK(output.IsContiguous()) << "output must be contiguous";
  TVM_FFI_ICHECK(output_scale.IsContiguous()) << "output_scale must be contiguous";
  TVM_FFI_ICHECK(values.IsContiguous()) << "values must be contiguous";
  TVM_FFI_ICHECK_EQ(output.dtype(), dl_float8_e4m3fn);
  TVM_FFI_ICHECK_EQ(output_scale.dtype(), dl_uint8);

  int64_t const rows = values.size(0);
  int64_t const hidden_dim = values.size(1);
  TVM_FFI_ICHECK_GE(rows, 0);
  TVM_FFI_ICHECK_GT(hidden_dim, 0);
  TVM_FFI_ICHECK_EQ(hidden_dim % kMxfp4Block, 0);
  TVM_FFI_ICHECK_EQ(output.size(0), rows);
  TVM_FFI_ICHECK_EQ(output.size(1), hidden_dim);
  TVM_FFI_ICHECK_EQ(output_scale.size(0), rows);
  TVM_FFI_ICHECK_EQ(output_scale.size(1), hidden_dim / kMxfp4Block);

  cudaSetDevice(output.device().device_id);
  cudaStream_t const stream = get_stream(output.device());
  if (values.dtype() == dl_float32) {
    launch_sm12x_mxfp4_mxfp8_quantize<float>(
        output, output_scale, values, stream);
  } else if (values.dtype() == dl_float16) {
    launch_sm12x_mxfp4_mxfp8_quantize<half>(
        output, output_scale, values, stream);
  } else if (values.dtype() == dl_bfloat16) {
    launch_sm12x_mxfp4_mxfp8_quantize<nv_bfloat16>(
        output, output_scale, values, stream);
  } else {
    TVM_FFI_ICHECK(false) << "values dtype must be float32, float16, or bfloat16";
  }
}

void sm12x_mxfp4_swiglu_mxfp8_quantize(
    TensorView output,
    TensorView output_scale,
    TensorView gate_up,
    TensorView expert_ids,
    TensorView w13_bias,
    bool use_swiglu,
    double swiglu_alpha,
    bool has_swiglu_alpha,
    double swiglu_limit,
    bool has_swiglu_limit,
    double swiglu_beta,
    bool has_swiglu_beta) {
  CHECK_INPUT(output);
  CHECK_INPUT(output_scale);
  CHECK_INPUT(gate_up);
  CHECK_INPUT(expert_ids);
  CHECK_CUDA(w13_bias);
  CHECK_DEVICE(output, output_scale);
  CHECK_DEVICE(output, gate_up);
  CHECK_DEVICE(output, expert_ids);
  CHECK_DEVICE(output, w13_bias);

  CHECK_DIM(2, output);
  CHECK_DIM(2, output_scale);
  CHECK_DIM(2, gate_up);
  CHECK_DIM(1, expert_ids);
  TVM_FFI_ICHECK(output.IsContiguous()) << "output must be contiguous";
  TVM_FFI_ICHECK(output_scale.IsContiguous()) << "output_scale must be contiguous";
  TVM_FFI_ICHECK(gate_up.IsContiguous()) << "gate_up must be contiguous";
  TVM_FFI_ICHECK(expert_ids.IsContiguous()) << "expert_ids must be contiguous";
  TVM_FFI_ICHECK(w13_bias.IsContiguous()) << "w13_bias must be contiguous";
  TVM_FFI_ICHECK_EQ(output.dtype(), dl_float8_e4m3fn);
  TVM_FFI_ICHECK_EQ(output_scale.dtype(), dl_uint8);
  TVM_FFI_ICHECK_EQ(gate_up.dtype(), dl_float32);
  TVM_FFI_ICHECK_EQ(expert_ids.dtype(), dl_int32);

  int const total_rows = static_cast<int>(gate_up.size(0));
  int const gate_up_width = static_cast<int>(gate_up.size(1));
  TVM_FFI_ICHECK_GT(total_rows, 0);
  TVM_FFI_ICHECK_GT(gate_up_width, 0);
  TVM_FFI_ICHECK_EQ(gate_up_width % 2, 0);
  int const intermediate_size = gate_up_width / 2;
  TVM_FFI_ICHECK_EQ(intermediate_size % kMxfp4Block, 0);
  TVM_FFI_ICHECK_EQ(output.size(0), total_rows);
  TVM_FFI_ICHECK_EQ(output.size(1), intermediate_size);
  TVM_FFI_ICHECK_EQ(output_scale.size(0), total_rows);
  TVM_FFI_ICHECK_EQ(output_scale.size(1), intermediate_size / kMxfp4Block);
  TVM_FFI_ICHECK_EQ(expert_ids.size(0), total_rows);

  bool const has_w13_bias = w13_bias.numel() > 0;
  int bias_stride = 0;
  if (has_w13_bias) {
    CHECK_DIM(2, w13_bias);
    TVM_FFI_ICHECK_EQ(w13_bias.dtype(), dl_float32);
    TVM_FFI_ICHECK_EQ(w13_bias.size(1), gate_up_width);
    bias_stride = gate_up_width;
  } else {
    CHECK_DIM(1, w13_bias);
  }

  cudaSetDevice(output.device().device_id);
  cudaStream_t const stream = get_stream(output.device());
  dim3 const grid(intermediate_size / kMxfp4Block, total_rows);
  swiglu_mxfp8_quantize_kernel<<<grid, kWarpSize, 0, stream>>>(
      static_cast<uint8_t*>(output.data_ptr()),
      static_cast<uint8_t*>(output_scale.data_ptr()),
      static_cast<const float*>(gate_up.data_ptr()),
      static_cast<const int*>(expert_ids.data_ptr()),
      static_cast<const float*>(w13_bias.data_ptr()),
      total_rows,
      intermediate_size,
      static_cast<int>(output_scale.size(1)),
      bias_stride,
      has_w13_bias,
      use_swiglu,
      static_cast<float>(swiglu_alpha),
      has_swiglu_alpha,
      static_cast<float>(swiglu_limit),
      has_swiglu_limit,
      static_cast<float>(swiglu_beta),
      has_swiglu_beta);
  cudaError_t const status = cudaGetLastError();
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "sm12x_mxfp4_swiglu_mxfp8_quantize launch failed: "
      << cudaGetErrorString(status);
}

void sm12x_mxfp4_moe_forward_impl(
    TensorView output,
    TensorView intermediate,
    TensorView hidden_states,
    TensorView topk_weights,
    TensorView topk_ids,
    TensorView w13_weight,
    TensorView w13_scale,
    TensorView w2_weight,
    TensorView w2_scale,
    TensorView w13_bias,
    TensorView w2_bias,
    int64_t ep_rank,
    int64_t ep_size,
    bool use_swiglu,
    double swiglu_alpha,
    bool has_swiglu_alpha,
    double swiglu_limit,
    bool has_swiglu_limit,
    double swiglu_beta,
    bool has_swiglu_beta) {
  CHECK_CUDA(output);
  CHECK_CUDA(intermediate);
  CHECK_CUDA(hidden_states);
  CHECK_CUDA(topk_weights);
  CHECK_CUDA(topk_ids);
  CHECK_CUDA(w13_weight);
  CHECK_CUDA(w13_scale);
  CHECK_CUDA(w2_weight);
  CHECK_CUDA(w2_scale);
  CHECK_CUDA(w13_bias);
  CHECK_CUDA(w2_bias);
  CHECK_DIM(2, output);
  CHECK_DIM(3, intermediate);
  CHECK_DIM(2, hidden_states);
  CHECK_DIM(2, topk_weights);
  CHECK_DIM(2, topk_ids);
  CHECK_DIM(3, w13_weight);
  CHECK_DIM(3, w13_scale);
  CHECK_DIM(3, w2_weight);
  CHECK_DIM(3, w2_scale);

  TVM_FFI_ICHECK(output.IsContiguous()) << "output must be contiguous";
  TVM_FFI_ICHECK(intermediate.IsContiguous()) << "intermediate must be contiguous";
  TVM_FFI_ICHECK(hidden_states.IsContiguous()) << "hidden_states must be contiguous";
  TVM_FFI_ICHECK(topk_weights.IsContiguous()) << "topk_weights must be contiguous";
  TVM_FFI_ICHECK(topk_ids.IsContiguous()) << "topk_ids must be contiguous";
  TVM_FFI_ICHECK(w13_weight.IsContiguous()) << "w13_weight must be contiguous";
  TVM_FFI_ICHECK(w13_scale.IsContiguous()) << "w13_scale must be contiguous";
  TVM_FFI_ICHECK(w2_weight.IsContiguous()) << "w2_weight must be contiguous";
  TVM_FFI_ICHECK(w2_scale.IsContiguous()) << "w2_scale must be contiguous";

  TVM_FFI_ICHECK(output.dtype() == hidden_states.dtype())
      << "output and hidden_states dtype must match";
  TVM_FFI_ICHECK(intermediate.dtype() == dl_float32)
      << "intermediate must be float32";
  TVM_FFI_ICHECK(topk_weights.dtype() == dl_float32)
      << "topk_weights must be float32";
  TVM_FFI_ICHECK(topk_ids.dtype() == dl_int32) << "topk_ids must be int32";
  TVM_FFI_ICHECK(w13_weight.dtype() == dl_uint8) << "w13_weight must be uint8";
  TVM_FFI_ICHECK(w13_scale.dtype() == dl_uint8) << "w13_scale must be uint8";
  TVM_FFI_ICHECK(w2_weight.dtype() == dl_uint8) << "w2_weight must be uint8";
  TVM_FFI_ICHECK(w2_scale.dtype() == dl_uint8) << "w2_scale must be uint8";
  TVM_FFI_ICHECK(w13_bias.dtype() == dl_float32) << "w13_bias must be float32";
  TVM_FFI_ICHECK(w2_bias.dtype() == dl_float32) << "w2_bias must be float32";

  int64_t const num_tokens = hidden_states.size(0);
  int64_t const hidden_dim = hidden_states.size(1);
  int64_t const top_k = topk_ids.size(1);
  int64_t const intermediate_size = w2_weight.size(2) * 2;
  int64_t const hidden_padded = w13_weight.size(2) * 2;

  TVM_FFI_ICHECK_EQ(output.size(0), num_tokens);
  TVM_FFI_ICHECK_EQ(output.size(1), hidden_dim);
  TVM_FFI_ICHECK_EQ(topk_weights.size(0), num_tokens);
  TVM_FFI_ICHECK_EQ(topk_weights.size(1), top_k);
  TVM_FFI_ICHECK_EQ(intermediate.size(0), num_tokens);
  TVM_FFI_ICHECK_EQ(intermediate.size(1), top_k);
  TVM_FFI_ICHECK_EQ(intermediate.size(2), intermediate_size);
  TVM_FFI_ICHECK_EQ(w13_weight.size(1), 2 * intermediate_size);
  TVM_FFI_ICHECK_EQ(w13_scale.size(0), w13_weight.size(0));
  TVM_FFI_ICHECK_EQ(w13_scale.size(1), w13_weight.size(1));
  TVM_FFI_ICHECK_EQ(w13_scale.size(2), hidden_padded / kMxfp4Block);
  TVM_FFI_ICHECK_EQ(w2_scale.size(0), w2_weight.size(0));
  TVM_FFI_ICHECK_EQ(w2_scale.size(1), w2_weight.size(1));
  TVM_FFI_ICHECK_EQ(w2_scale.size(2), intermediate_size / kMxfp4Block);
  TVM_FFI_ICHECK_EQ(w2_weight.size(0), w13_weight.size(0));
  TVM_FFI_ICHECK_LE(hidden_dim, hidden_padded);
  TVM_FFI_ICHECK_LE(hidden_dim, w2_weight.size(1));
  TVM_FFI_ICHECK_EQ(hidden_padded % kMxfp4Block, 0);
  TVM_FFI_ICHECK_EQ(intermediate_size % kMxfp4Block, 0);
  TVM_FFI_ICHECK_GE(ep_rank, 0);
  TVM_FFI_ICHECK_GE(ep_size, 1);

  if (w13_bias.numel() > 0) {
    CHECK_DIM(2, w13_bias);
    TVM_FFI_ICHECK(w13_bias.IsContiguous()) << "w13_bias must be contiguous";
    TVM_FFI_ICHECK_EQ(w13_bias.size(0), w13_weight.size(0));
    TVM_FFI_ICHECK_EQ(w13_bias.size(1), w13_weight.size(1));
  }
  if (w2_bias.numel() > 0) {
    CHECK_DIM(2, w2_bias);
    TVM_FFI_ICHECK(w2_bias.IsContiguous()) << "w2_bias must be contiguous";
    TVM_FFI_ICHECK_EQ(w2_bias.size(0), w2_weight.size(0));
    TVM_FFI_ICHECK_GE(w2_bias.size(1), hidden_dim);
  }

  cudaSetDevice(output.device().device_id);
  cudaStream_t const stream = get_stream(output.device());
  if (hidden_states.dtype() == dl_float32) {
    launch_sm12x_mxfp4_moe_forward<float>(
        output, intermediate, hidden_states, topk_weights, topk_ids, w13_weight,
        w13_scale, w2_weight, w2_scale, w13_bias, w2_bias,
        static_cast<int>(ep_rank), static_cast<int>(ep_size), use_swiglu,
        static_cast<float>(swiglu_alpha), has_swiglu_alpha,
        static_cast<float>(swiglu_limit), has_swiglu_limit,
        static_cast<float>(swiglu_beta), has_swiglu_beta, stream);
  } else if (hidden_states.dtype() == dl_float16) {
    launch_sm12x_mxfp4_moe_forward<half>(
        output, intermediate, hidden_states, topk_weights, topk_ids, w13_weight,
        w13_scale, w2_weight, w2_scale, w13_bias, w2_bias,
        static_cast<int>(ep_rank), static_cast<int>(ep_size), use_swiglu,
        static_cast<float>(swiglu_alpha), has_swiglu_alpha,
        static_cast<float>(swiglu_limit), has_swiglu_limit,
        static_cast<float>(swiglu_beta), has_swiglu_beta, stream);
  } else if (hidden_states.dtype() == dl_bfloat16) {
    launch_sm12x_mxfp4_moe_forward<nv_bfloat16>(
        output, intermediate, hidden_states, topk_weights, topk_ids, w13_weight,
        w13_scale, w2_weight, w2_scale, w13_bias, w2_bias,
        static_cast<int>(ep_rank), static_cast<int>(ep_size), use_swiglu,
        static_cast<float>(swiglu_alpha), has_swiglu_alpha,
        static_cast<float>(swiglu_limit), has_swiglu_limit,
        static_cast<float>(swiglu_beta), has_swiglu_beta, stream);
  } else {
    TVM_FFI_ICHECK(false)
        << "hidden_states dtype must be float32, float16, or bfloat16";
  }
}

void sm12x_mxfp4_moe_forward_warp(
    TensorView output,
    TensorView intermediate,
    TensorView hidden_states,
    TensorView topk_weights,
    TensorView topk_ids,
    TensorView w13_weight,
    TensorView w13_scale,
    TensorView w2_weight,
    TensorView w2_scale,
    TensorView w13_bias,
    TensorView w2_bias,
    int64_t ep_rank,
    int64_t ep_size,
    bool use_swiglu,
    double swiglu_alpha,
    bool has_swiglu_alpha,
    double swiglu_limit,
    bool has_swiglu_limit,
    double swiglu_beta,
    bool has_swiglu_beta) {
  sm12x_mxfp4_moe_forward_impl(
      output, intermediate, hidden_states, topk_weights, topk_ids, w13_weight,
      w13_scale, w2_weight, w2_scale, w13_bias, w2_bias, ep_rank, ep_size,
      use_swiglu, swiglu_alpha, has_swiglu_alpha, swiglu_limit,
      has_swiglu_limit, swiglu_beta, has_swiglu_beta);
}

void sm12x_mxfp8_mxfp4_mma_tile(
    TensorView output,
    TensorView activations,
    TensorView activation_scale,
    TensorView weight,
    TensorView weight_scale) {
  CHECK_INPUT(output);
  CHECK_INPUT(activations);
  CHECK_INPUT(activation_scale);
  CHECK_INPUT(weight);
  CHECK_INPUT(weight_scale);
  CHECK_DEVICE(output, activations);
  CHECK_DEVICE(output, activation_scale);
  CHECK_DEVICE(output, weight);
  CHECK_DEVICE(output, weight_scale);

  CHECK_DIM(2, output);
  CHECK_DIM(2, activations);
  CHECK_DIM(2, activation_scale);
  CHECK_DIM(2, weight);
  CHECK_DIM(2, weight_scale);
  TVM_FFI_ICHECK_EQ(output.dtype(), dl_float32);
  TVM_FFI_ICHECK_EQ(activations.dtype(), dl_float8_e4m3fn);
  TVM_FFI_ICHECK_EQ(activation_scale.dtype(), dl_uint8);
  TVM_FFI_ICHECK_EQ(weight.dtype(), dl_uint8);
  TVM_FFI_ICHECK_EQ(weight_scale.dtype(), dl_uint8);
  TVM_FFI_ICHECK_EQ(output.size(0), 16);
  TVM_FFI_ICHECK_EQ(output.size(1), 8);
  TVM_FFI_ICHECK_EQ(activations.size(0), 16);
  TVM_FFI_ICHECK_EQ(activations.size(1), 32);
  TVM_FFI_ICHECK_EQ(activation_scale.size(0), 16);
  TVM_FFI_ICHECK_EQ(activation_scale.size(1), 1);
  TVM_FFI_ICHECK_EQ(weight.size(0), 8);
  TVM_FFI_ICHECK_EQ(weight.size(1), 16);
  TVM_FFI_ICHECK_EQ(weight_scale.size(0), 8);
  TVM_FFI_ICHECK_EQ(weight_scale.size(1), 1);

  cudaSetDevice(output.device().device_id);
  cudaStream_t const stream = get_stream(output.device());
  mxfp8_mxfp4_mma_tile_kernel<<<1, kWarpSize, 0, stream>>>(
      static_cast<float*>(output.data_ptr()),
      static_cast<const uint8_t*>(activations.data_ptr()),
      static_cast<const uint8_t*>(activation_scale.data_ptr()),
      static_cast<const uint8_t*>(weight.data_ptr()),
      static_cast<const uint8_t*>(weight_scale.data_ptr()));
  cudaError_t const status = cudaGetLastError();
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "sm12x_mxfp8_mxfp4_mma_tile launch failed: "
      << cudaGetErrorString(status);
}

void sm12x_mxfp4_mxfp8_mma_tile(
    TensorView output,
    TensorView weight,
    TensorView weight_scale,
    TensorView activations,
    TensorView activation_scale) {
  CHECK_CUDA(output);
  CHECK_CUDA(weight);
  CHECK_CUDA(weight_scale);
  CHECK_CUDA(activations);
  CHECK_CUDA(activation_scale);
  CHECK_DEVICE(output, weight);
  CHECK_DEVICE(output, weight_scale);
  CHECK_DEVICE(output, activations);
  CHECK_DEVICE(output, activation_scale);
  TVM_FFI_ICHECK_EQ(output.dtype(), dl_float32);
  TVM_FFI_ICHECK_EQ(weight.dtype(), dl_uint8);
  TVM_FFI_ICHECK_EQ(weight_scale.dtype(), dl_uint8);
  TVM_FFI_ICHECK_EQ(activations.dtype(), dl_float8_e4m3fn);
  TVM_FFI_ICHECK_EQ(activation_scale.dtype(), dl_uint8);
  TVM_FFI_ICHECK_EQ(output.size(0), 16);
  TVM_FFI_ICHECK_EQ(output.size(1), 8);
  TVM_FFI_ICHECK_EQ(weight.size(0), 16);
  TVM_FFI_ICHECK_EQ(weight.size(1), 16);
  TVM_FFI_ICHECK_EQ(weight_scale.size(0), 16);
  TVM_FFI_ICHECK_EQ(weight_scale.size(1), 1);
  TVM_FFI_ICHECK_EQ(activations.size(0), 8);
  TVM_FFI_ICHECK_EQ(activations.size(1), 32);
  TVM_FFI_ICHECK_EQ(activation_scale.size(0), 8);
  TVM_FFI_ICHECK_EQ(activation_scale.size(1), 1);

  cudaSetDevice(output.device().device_id);
  cudaStream_t const stream = get_stream(output.device());
  mxfp4_mxfp8_mma_tile_kernel<<<1, kWarpSize, 0, stream>>>(
      static_cast<float*>(output.data_ptr()),
      static_cast<const uint8_t*>(weight.data_ptr()),
      static_cast<const uint8_t*>(weight_scale.data_ptr()),
      static_cast<const uint8_t*>(activations.data_ptr()),
      static_cast<const uint8_t*>(activation_scale.data_ptr()));
  cudaError_t status = cudaGetLastError();
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "sm12x_mxfp4_mxfp8_mma_tile launch failed: "
      << cudaGetErrorString(status);
}

void sm12x_mxfp4_mxfp8_dense(
    TensorView output,
    TensorView weight,
    TensorView weight_scale,
    TensorView activations,
    TensorView activation_scale) {
  CHECK_CUDA(output);
  CHECK_CUDA(weight);
  CHECK_CUDA(weight_scale);
  CHECK_CUDA(activations);
  CHECK_CUDA(activation_scale);
  CHECK_DEVICE(output, weight);
  CHECK_DEVICE(output, weight_scale);
  CHECK_DEVICE(output, activations);
  CHECK_DEVICE(output, activation_scale);

  CHECK_DIM(2, output);
  CHECK_DIM(2, weight);
  CHECK_DIM(2, weight_scale);
  CHECK_DIM(2, activations);
  CHECK_DIM(2, activation_scale);
  TVM_FFI_ICHECK_EQ(output.dtype(), dl_float32);
  TVM_FFI_ICHECK_EQ(weight.dtype(), dl_uint8);
  TVM_FFI_ICHECK_EQ(weight_scale.dtype(), dl_uint8);
  TVM_FFI_ICHECK_EQ(activations.dtype(), dl_float8_e4m3fn);
  TVM_FFI_ICHECK_EQ(activation_scale.dtype(), dl_uint8);

  int const m_dim = static_cast<int>(activations.size(0));
  int const k_dim = static_cast<int>(activations.size(1));
  int const n_dim = static_cast<int>(weight.size(0));
  TVM_FFI_ICHECK_GT(m_dim, 0);
  TVM_FFI_ICHECK_GT(n_dim, 0);
  TVM_FFI_ICHECK_GT(k_dim, 0);
  TVM_FFI_ICHECK_EQ(m_dim % 8, 0);
  TVM_FFI_ICHECK_EQ(n_dim % 16, 0);
  TVM_FFI_ICHECK_EQ(k_dim % kMxfp4Block, 0);
  TVM_FFI_ICHECK_EQ(output.size(0), m_dim);
  TVM_FFI_ICHECK_EQ(output.size(1), n_dim);
  TVM_FFI_ICHECK_EQ(weight.size(1), k_dim / 2);
  TVM_FFI_ICHECK_EQ(weight_scale.size(0), n_dim);
  TVM_FFI_ICHECK_EQ(weight_scale.size(1), k_dim / kMxfp4Block);
  TVM_FFI_ICHECK_EQ(activation_scale.size(0), m_dim);
  TVM_FFI_ICHECK_EQ(activation_scale.size(1), k_dim / kMxfp4Block);

  cudaSetDevice(output.device().device_id);
  cudaStream_t const stream = get_stream(output.device());
  dim3 const grid(n_dim / 16, m_dim / 8);
  mxfp4_mxfp8_dense_kernel<<<grid, kWarpSize, 0, stream>>>(
      static_cast<float*>(output.data_ptr()),
      static_cast<const uint8_t*>(weight.data_ptr()),
      static_cast<const uint8_t*>(weight_scale.data_ptr()),
      static_cast<const uint8_t*>(activations.data_ptr()),
      static_cast<const uint8_t*>(activation_scale.data_ptr()),
      m_dim,
      n_dim,
      k_dim,
      static_cast<int>(weight.size(1)),
      static_cast<int>(weight_scale.size(1)),
      static_cast<int>(activation_scale.size(1)));
  cudaError_t status = cudaGetLastError();
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "sm12x_mxfp4_mxfp8_dense launch failed: "
      << cudaGetErrorString(status);
}

void sm12x_mxfp8_mxfp4_dense(
    TensorView output,
    TensorView activations,
    TensorView activation_scale,
    TensorView weight,
    TensorView weight_scale) {
  CHECK_INPUT(output);
  CHECK_INPUT(activations);
  CHECK_INPUT(activation_scale);
  CHECK_INPUT(weight);
  CHECK_INPUT(weight_scale);
  CHECK_DEVICE(output, activations);
  CHECK_DEVICE(output, activation_scale);
  CHECK_DEVICE(output, weight);
  CHECK_DEVICE(output, weight_scale);

  CHECK_DIM(2, output);
  CHECK_DIM(2, activations);
  CHECK_DIM(2, activation_scale);
  CHECK_DIM(2, weight);
  CHECK_DIM(2, weight_scale);
  TVM_FFI_ICHECK_EQ(output.dtype(), dl_float32);
  TVM_FFI_ICHECK_EQ(activations.dtype(), dl_float8_e4m3fn);
  TVM_FFI_ICHECK_EQ(activation_scale.dtype(), dl_uint8);
  TVM_FFI_ICHECK_EQ(weight.dtype(), dl_uint8);
  TVM_FFI_ICHECK_EQ(weight_scale.dtype(), dl_uint8);

  int const m_dim = static_cast<int>(activations.size(0));
  int const k_dim = static_cast<int>(activations.size(1));
  int const n_dim = static_cast<int>(weight.size(0));
  TVM_FFI_ICHECK_GT(m_dim, 0);
  TVM_FFI_ICHECK_GT(n_dim, 0);
  TVM_FFI_ICHECK_GT(k_dim, 0);
  TVM_FFI_ICHECK_EQ(m_dim % 16, 0);
  TVM_FFI_ICHECK_EQ(n_dim % 8, 0);
  TVM_FFI_ICHECK_EQ(k_dim % kMxfp4Block, 0);
  TVM_FFI_ICHECK_EQ(output.size(0), m_dim);
  TVM_FFI_ICHECK_EQ(output.size(1), n_dim);
  TVM_FFI_ICHECK_EQ(weight.size(1), k_dim / 2);
  TVM_FFI_ICHECK_EQ(activation_scale.size(0), m_dim);
  TVM_FFI_ICHECK_EQ(activation_scale.size(1), k_dim / kMxfp4Block);
  TVM_FFI_ICHECK_EQ(weight_scale.size(0), n_dim);
  TVM_FFI_ICHECK_EQ(weight_scale.size(1), k_dim / kMxfp4Block);

  cudaSetDevice(output.device().device_id);
  cudaStream_t const stream = get_stream(output.device());
  dim3 const grid(n_dim / 8, m_dim / 16);
  mxfp8_mxfp4_dense_kernel<<<grid, kWarpSize, 0, stream>>>(
      static_cast<float*>(output.data_ptr()),
      static_cast<const uint8_t*>(activations.data_ptr()),
      static_cast<const uint8_t*>(activation_scale.data_ptr()),
      static_cast<const uint8_t*>(weight.data_ptr()),
      static_cast<const uint8_t*>(weight_scale.data_ptr()),
      m_dim,
      n_dim,
      k_dim,
      static_cast<int>(activation_scale.size(1)),
      static_cast<int>(weight.size(1)),
      static_cast<int>(weight_scale.size(1)));
  cudaError_t const status = cudaGetLastError();
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "sm12x_mxfp8_mxfp4_dense launch failed: "
      << cudaGetErrorString(status);
}

void sm12x_mxfp4_moe_w13_tensorcore(
    TensorView gate_up,
    TensorView hidden_fp8,
    TensorView hidden_scale,
    TensorView topk_ids,
    TensorView w13_weight,
    TensorView w13_scale,
    int64_t ep_rank) {
  CHECK_CUDA(gate_up);
  CHECK_CUDA(hidden_fp8);
  CHECK_CUDA(hidden_scale);
  CHECK_CUDA(topk_ids);
  CHECK_CUDA(w13_weight);
  CHECK_CUDA(w13_scale);
  CHECK_DEVICE(gate_up, hidden_fp8);
  CHECK_DEVICE(gate_up, hidden_scale);
  CHECK_DEVICE(gate_up, topk_ids);
  CHECK_DEVICE(gate_up, w13_weight);
  CHECK_DEVICE(gate_up, w13_scale);

  CHECK_DIM(3, gate_up);
  CHECK_DIM(2, hidden_fp8);
  CHECK_DIM(2, hidden_scale);
  CHECK_DIM(2, topk_ids);
  CHECK_DIM(3, w13_weight);
  CHECK_DIM(3, w13_scale);

  TVM_FFI_ICHECK(gate_up.IsContiguous()) << "gate_up must be contiguous";
  TVM_FFI_ICHECK(hidden_fp8.IsContiguous()) << "hidden_fp8 must be contiguous";
  TVM_FFI_ICHECK(hidden_scale.IsContiguous())
      << "hidden_scale must be contiguous";
  TVM_FFI_ICHECK(topk_ids.IsContiguous()) << "topk_ids must be contiguous";
  TVM_FFI_ICHECK(w13_weight.IsContiguous()) << "w13_weight must be contiguous";
  TVM_FFI_ICHECK(w13_scale.IsContiguous()) << "w13_scale must be contiguous";

  TVM_FFI_ICHECK_EQ(gate_up.dtype(), dl_float32);
  TVM_FFI_ICHECK_EQ(hidden_fp8.dtype(), dl_float8_e4m3fn);
  TVM_FFI_ICHECK_EQ(hidden_scale.dtype(), dl_uint8);
  TVM_FFI_ICHECK_EQ(topk_ids.dtype(), dl_int32);
  TVM_FFI_ICHECK_EQ(w13_weight.dtype(), dl_uint8);
  TVM_FFI_ICHECK_EQ(w13_scale.dtype(), dl_uint8);

  int const num_tokens = static_cast<int>(hidden_fp8.size(0));
  int const top_k = static_cast<int>(topk_ids.size(1));
  int const hidden = static_cast<int>(hidden_fp8.size(1));
  int const num_local_experts = static_cast<int>(w13_weight.size(0));
  int const gate_up_dim = static_cast<int>(w13_weight.size(1));
  int const weight_packed_k = static_cast<int>(w13_weight.size(2));
  int const weight_scale_k = static_cast<int>(w13_scale.size(2));
  int const hidden_scale_k = static_cast<int>(hidden_scale.size(1));

  TVM_FFI_ICHECK_EQ(topk_ids.size(0), num_tokens);
  TVM_FFI_ICHECK_EQ(gate_up.size(0), num_tokens);
  TVM_FFI_ICHECK_EQ(gate_up.size(1), top_k);
  TVM_FFI_ICHECK_EQ(gate_up.size(2), gate_up_dim);
  TVM_FFI_ICHECK_EQ(hidden_scale.size(0), num_tokens);
  TVM_FFI_ICHECK_EQ(w13_scale.size(0), num_local_experts);
  TVM_FFI_ICHECK_EQ(w13_scale.size(1), gate_up_dim);
  TVM_FFI_ICHECK_EQ(weight_packed_k, hidden / 2);
  TVM_FFI_ICHECK_EQ(weight_scale_k, hidden / kMxfp4Block);
  TVM_FFI_ICHECK_EQ(hidden_scale_k, hidden / kMxfp4Block);
  TVM_FFI_ICHECK_EQ(hidden % kMxfp4Block, 0);
  TVM_FFI_ICHECK_EQ(gate_up_dim % 16, 0);
  TVM_FFI_ICHECK_GE(ep_rank, 0);

  cudaSetDevice(gate_up.device().device_id);
  cudaStream_t const stream = get_stream(gate_up.device());
  launch_sm12x_mxfp4_moe_w13_tensorcore(
      gate_up, hidden_fp8, hidden_scale, topk_ids, w13_weight, w13_scale,
      static_cast<int>(ep_rank), stream);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(sm12x_mxfp4_moe_forward_warp,
                              sm12x_mxfp4_moe_forward_warp);
void sm12x_mxfp4_moe_w2_tensorcore(
    TensorView per_pair_out,
    TensorView intermediate_fp8,
    TensorView intermediate_scale,
    TensorView topk_ids,
    TensorView w2_weight,
    TensorView w2_scale,
    TensorView w2_bias,
    int64_t ep_rank) {
  CHECK_CUDA(per_pair_out);
  CHECK_CUDA(intermediate_fp8);
  CHECK_CUDA(intermediate_scale);
  CHECK_CUDA(topk_ids);
  CHECK_CUDA(w2_weight);
  CHECK_CUDA(w2_scale);
  CHECK_DEVICE(per_pair_out, intermediate_fp8);
  CHECK_DEVICE(per_pair_out, intermediate_scale);
  CHECK_DEVICE(per_pair_out, topk_ids);
  CHECK_DEVICE(per_pair_out, w2_weight);
  CHECK_DEVICE(per_pair_out, w2_scale);

  CHECK_DIM(3, per_pair_out);
  CHECK_DIM(2, intermediate_fp8);
  CHECK_DIM(2, intermediate_scale);
  CHECK_DIM(2, topk_ids);
  CHECK_DIM(3, w2_weight);
  CHECK_DIM(3, w2_scale);

  TVM_FFI_ICHECK(per_pair_out.IsContiguous()) << "per_pair_out must be contiguous";
  TVM_FFI_ICHECK(intermediate_fp8.IsContiguous())
      << "intermediate_fp8 must be contiguous";
  TVM_FFI_ICHECK(intermediate_scale.IsContiguous())
      << "intermediate_scale must be contiguous";
  TVM_FFI_ICHECK(topk_ids.IsContiguous()) << "topk_ids must be contiguous";
  TVM_FFI_ICHECK(w2_weight.IsContiguous()) << "w2_weight must be contiguous";
  TVM_FFI_ICHECK(w2_scale.IsContiguous()) << "w2_scale must be contiguous";

  TVM_FFI_ICHECK_EQ(per_pair_out.dtype(), dl_float32);
  TVM_FFI_ICHECK_EQ(intermediate_fp8.dtype(), dl_float8_e4m3fn);
  TVM_FFI_ICHECK_EQ(intermediate_scale.dtype(), dl_uint8);
  TVM_FFI_ICHECK_EQ(topk_ids.dtype(), dl_int32);
  TVM_FFI_ICHECK_EQ(w2_weight.dtype(), dl_uint8);
  TVM_FFI_ICHECK_EQ(w2_scale.dtype(), dl_uint8);

  int const num_tokens = static_cast<int>(topk_ids.size(0));
  int const top_k = static_cast<int>(topk_ids.size(1));
  int const num_pairs = num_tokens * top_k;
  int const intermediate = static_cast<int>(intermediate_fp8.size(1));
  int const num_local_experts = static_cast<int>(w2_weight.size(0));
  int const hidden = static_cast<int>(w2_weight.size(1));

  TVM_FFI_ICHECK_EQ(intermediate_fp8.size(0), num_pairs);
  TVM_FFI_ICHECK_EQ(intermediate_scale.size(0), num_pairs);
  TVM_FFI_ICHECK_EQ(per_pair_out.size(0), num_tokens);
  TVM_FFI_ICHECK_EQ(per_pair_out.size(1), top_k);
  TVM_FFI_ICHECK_EQ(per_pair_out.size(2), hidden);
  TVM_FFI_ICHECK_EQ(w2_scale.size(0), num_local_experts);
  TVM_FFI_ICHECK_EQ(w2_scale.size(1), hidden);
  TVM_FFI_ICHECK_EQ(w2_weight.size(2), intermediate / 2);
  TVM_FFI_ICHECK_EQ(w2_scale.size(2), intermediate / kMxfp4Block);
  TVM_FFI_ICHECK_EQ(intermediate_scale.size(1), intermediate / kMxfp4Block);
  TVM_FFI_ICHECK_EQ(intermediate % kMxfp4Block, 0);
  TVM_FFI_ICHECK_EQ(hidden % 16, 0);
  TVM_FFI_ICHECK_GE(ep_rank, 0);

  // ``w2_bias`` is signalled as an empty 0-D tensor (Python side passes
  // ``torch.empty(0, dtype=float32)`` when there is no bias) so the FFI
  // contract doesn't have to be made optional. A real bias is
  // ``[num_local_experts, hidden]`` float32.
  float const* w2_bias_ptr = nullptr;
  if (w2_bias.ndim() > 0 && w2_bias.numel() > 0) {
    CHECK_CUDA(w2_bias);
    CHECK_DEVICE(per_pair_out, w2_bias);
    CHECK_DIM(2, w2_bias);
    TVM_FFI_ICHECK(w2_bias.IsContiguous()) << "w2_bias must be contiguous";
    TVM_FFI_ICHECK_EQ(w2_bias.dtype(), dl_float32);
    TVM_FFI_ICHECK_EQ(w2_bias.size(0), num_local_experts);
    TVM_FFI_ICHECK_EQ(w2_bias.size(1), hidden);
    w2_bias_ptr = static_cast<const float*>(w2_bias.data_ptr());
  }

  cudaSetDevice(per_pair_out.device().device_id);
  cudaStream_t const stream = get_stream(per_pair_out.device());
  launch_sm12x_mxfp4_moe_w2_tensorcore(
      per_pair_out, intermediate_fp8, intermediate_scale, topk_ids, w2_weight,
      w2_scale, w2_bias_ptr, static_cast<int>(ep_rank), stream);
}

void sm12x_mxfp4_moe_weighted_reduce(
    TensorView output,
    TensorView per_pair,
    TensorView topk_weights) {
  CHECK_CUDA(output);
  CHECK_CUDA(per_pair);
  CHECK_CUDA(topk_weights);
  CHECK_DEVICE(output, per_pair);
  CHECK_DEVICE(output, topk_weights);
  CHECK_DIM(2, output);
  CHECK_DIM(3, per_pair);
  CHECK_DIM(2, topk_weights);
  TVM_FFI_ICHECK(output.IsContiguous()) << "output must be contiguous";
  TVM_FFI_ICHECK(per_pair.IsContiguous()) << "per_pair must be contiguous";
  TVM_FFI_ICHECK(topk_weights.IsContiguous())
      << "topk_weights must be contiguous";
  TVM_FFI_ICHECK_EQ(per_pair.dtype(), dl_float32);
  TVM_FFI_ICHECK_EQ(topk_weights.dtype(), dl_float32);

  int const num_tokens = static_cast<int>(per_pair.size(0));
  int const top_k = static_cast<int>(per_pair.size(1));
  int const hidden = static_cast<int>(per_pair.size(2));
  TVM_FFI_ICHECK_EQ(output.size(0), num_tokens);
  TVM_FFI_ICHECK_EQ(output.size(1), hidden);
  TVM_FFI_ICHECK_EQ(topk_weights.size(0), num_tokens);
  TVM_FFI_ICHECK_EQ(topk_weights.size(1), top_k);

  cudaSetDevice(output.device().device_id);
  cudaStream_t const stream = get_stream(output.device());
  if (output.dtype() == dl_float32) {
    launch_moe_weighted_reduce<float>(output, per_pair, topk_weights, stream);
  } else if (output.dtype() == dl_float16) {
    launch_moe_weighted_reduce<half>(output, per_pair, topk_weights, stream);
  } else if (output.dtype() == dl_bfloat16) {
    launch_moe_weighted_reduce<nv_bfloat16>(
        output, per_pair, topk_weights, stream);
  } else {
    TVM_FFI_ICHECK(false)
        << "output dtype must be float32, float16, or bfloat16";
  }
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(sm12x_mxfp4_moe_w13_tensorcore,
                              sm12x_mxfp4_moe_w13_tensorcore);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(sm12x_mxfp4_moe_w2_tensorcore,
                              sm12x_mxfp4_moe_w2_tensorcore);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(sm12x_mxfp4_moe_weighted_reduce,
                              sm12x_mxfp4_moe_weighted_reduce);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(sm12x_mxfp4_mxfp8_quantize,
                              sm12x_mxfp4_mxfp8_quantize);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(sm12x_mxfp4_swiglu_mxfp8_quantize,
                              sm12x_mxfp4_swiglu_mxfp8_quantize);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(sm12x_mxfp8_mxfp4_mma_tile,
                              sm12x_mxfp8_mxfp4_mma_tile);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(sm12x_mxfp4_mxfp8_mma_tile,
                              sm12x_mxfp4_mxfp8_mma_tile);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(sm12x_mxfp4_mxfp8_dense,
                              sm12x_mxfp4_mxfp8_dense);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(sm12x_mxfp8_mxfp4_dense,
                              sm12x_mxfp8_mxfp4_dense);
