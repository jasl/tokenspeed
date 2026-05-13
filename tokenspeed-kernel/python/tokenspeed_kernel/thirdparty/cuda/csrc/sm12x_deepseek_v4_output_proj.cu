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
// Two earlier hand-written designs were rejected and one cooperative-tile
// MMA design only reached ~0.5x of the contemporaneous Triton baseline
// (see docs/notes/2026-05-09-ds4-sm12x-rejected-experiments.md). The
// per-thread GEMV-style design ported here side-steps the tensor-core
// path entirely -- at decode shape (small `tokens` batch) the M axis is
// too small for MMA tiles to pay for themselves; a thread-per-output-cell
// GEMV with fp8x4 loads and L1/L2-friendly weight access wins.

#include <cmath>
#include <cstdint>
#include <type_traits>

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

// =============================================================================
// Fused inverse-RoPE + FP8 quant (DSv4 attention output projection prelude)
// =============================================================================
//
// Per DeepSeek V4 attention, the attention output for each (token, head)
// has its last `rope_dim=64` elements rotated by the inverse RoPE and then
// the full `head_dim=512` is quantized into FP8 with one UE8M0-style
// fp32 block-128 scale per `quant_group_size=128` chunk. The downstream
// `bhr,hdr->bhd` einsum consumes this output.
//
// Schedule: one CUDA block per (token, head). The block has 512 threads
// (one per head_dim element). Per-block work:
//   1. Load `o[token, head, tid]` as fp32.
//   2. If `tid` falls in the rope region (last `rope_dim` elements),
//      apply inverse RoPE by swapping with the partner element
//      (`tid ^ 1`) and multiplying by cos / -sin from `cos_sin_cache`.
//   3. Compute block-128 absolute max via warp shuffles + cross-warp
//      shared-memory reduction. 4 chunks per head -> 4 chunk_max slots.
//   4. UE8M0-style fp32 scale = `exp2(ceil(log2(absmax / fp8_max)))`.
//   5. Quantize: clamp(x / scale, ±fp8_max) -> fp8 e4m3.
//   6. Write fp8 (per-thread) and scale (one thread per chunk) into the
//      `[G, T_aligned, H]` storage layout consumed by the einsum.
//
// T_aligned padding (multiple-of-4) is the caller's responsibility --
// this kernel skips the data write and zeros the scale for padded
// tokens so the downstream einsum sees stable zero contributions there.

constexpr int kQuantGroupSize = 128;
constexpr int kHeadDim = 512;
constexpr int kChunksPerHead = kHeadDim / kQuantGroupSize;            // 4
constexpr int kBlockThreadsForInvRopeQuant = kHeadDim;                // 512
constexpr int kWarpsForInvRopeQuant = kBlockThreadsForInvRopeQuant / 32;  // 16
constexpr float kFp8E4m3Max = 448.0f;
constexpr float kInvRopeQuantEps = 1.0e-10f;

template <typename input_t>
__global__ void deepseek_v4_inv_rope_fp8_quant_kernel(
    __nv_fp8_e4m3* __restrict__ fp8_out,
    float* __restrict__ scale_out,
    const input_t* __restrict__ o,
    const int64_t* __restrict__ positions,
    const float* __restrict__ cos_sin_cache,
    int num_tokens,
    int heads_per_group,
    int rope_abs_start,
    int half_rope,
    int64_t o_stride_token,
    int64_t o_stride_head,
    int64_t cache_stride_pos,
    int64_t fp8_stride_group,
    int64_t fp8_stride_token,
    int64_t scale_stride_group,
    int64_t scale_stride_block) {
  int const token_id = blockIdx.x;
  int const head_id = blockIdx.y;
  int const tid = threadIdx.x;

  int const group = head_id / heads_per_group;
  int const head_in_group = head_id % heads_per_group;
  int const chunk_id = tid / kQuantGroupSize;          // 0..3
  int const scale_block_idx = head_in_group * kChunksPerHead + chunk_id;

  // Padded-token slot: just zero the scale and bail.
  if (token_id >= num_tokens) {
    if (tid < kChunksPerHead) {
      int const sb = head_in_group * kChunksPerHead + tid;
      scale_out[group * scale_stride_group
                + token_id
                + sb * scale_stride_block] = 0.0f;
    }
    return;
  }

  // Load this thread's element of `o[token, head, :]`.
  const input_t* input_base = o + static_cast<int64_t>(token_id) * o_stride_token
                              + static_cast<int64_t>(head_id) * o_stride_head;
  float x;
  if constexpr (std::is_same_v<input_t, nv_bfloat16>) {
    x = __bfloat162float(input_base[tid]);
  } else {
    x = static_cast<float>(input_base[tid]);
  }

  // Inverse RoPE for the last `rope_dim` elements.
  if (tid >= rope_abs_start) {
    int const rope_local = tid - rope_abs_start;
    int const partner_idx = tid ^ 1;
    float partner;
    if constexpr (std::is_same_v<input_t, nv_bfloat16>) {
      partner = __bfloat162float(input_base[partner_idx]);
    } else {
      partner = static_cast<float>(input_base[partner_idx]);
    }
    int const cs_idx = rope_local >> 1;
    int64_t const pos = positions[token_id];
    float const cos_v = cos_sin_cache[pos * cache_stride_pos + cs_idx];
    float const sin_v = cos_sin_cache[pos * cache_stride_pos + half_rope + cs_idx];
    if ((rope_local & 1) == 0) {
      x = x * cos_v + partner * sin_v;
    } else {
      x = x * cos_v - partner * sin_v;
    }
  }

  // Per-chunk absolute max via warp + cross-warp reductions.
  // 4 warps per chunk (128 threads per chunk, 32 threads per warp).
  float ax = fabsf(x);
  unsigned int const full_mask = 0xFFFFFFFFu;
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    ax = fmaxf(ax, __shfl_down_sync(full_mask, ax, offset));
  }
  // Lane 0 of each warp now holds its warp's max.

  __shared__ float warp_max_shared[kWarpsForInvRopeQuant];
  __shared__ float chunk_max[kChunksPerHead];
  int const lane = tid & 31;
  int const global_warp = tid >> 5;
  int const warp_in_chunk = global_warp & 3;  // 4 warps per chunk
  if (lane == 0) {
    warp_max_shared[global_warp] = ax;
  }
  __syncthreads();

  if (warp_in_chunk == 0) {
    int const chunk_warp_base = chunk_id * 4;
    float cm = (lane < 4) ? warp_max_shared[chunk_warp_base + lane] : 0.0f;
    cm = fmaxf(cm, __shfl_down_sync(full_mask, cm, 2));
    cm = fmaxf(cm, __shfl_down_sync(full_mask, cm, 1));
    if (lane == 0) {
      chunk_max[chunk_id] = cm;
    }
  }
  __syncthreads();

  // UE8M0 scale: exp2(ceil(log2(absmax / fp8_max))).
  float const absmax = fmaxf(chunk_max[chunk_id], kInvRopeQuantEps);
  float const scale = exp2f(ceilf(log2f(absmax * (1.0f / kFp8E4m3Max))));
  float const x_quant_f = fminf(fmaxf(x / scale, -kFp8E4m3Max), kFp8E4m3Max);

  int64_t const fp8_offset = static_cast<int64_t>(group) * fp8_stride_group
                             + static_cast<int64_t>(token_id) * fp8_stride_token
                             + static_cast<int64_t>(head_in_group) * kHeadDim
                             + tid;
  fp8_out[fp8_offset] = __nv_fp8_e4m3(x_quant_f);

  // One thread per chunk writes the scale.
  if ((tid & (kQuantGroupSize - 1)) == 0) {
    int64_t const scale_offset = static_cast<int64_t>(group) * scale_stride_group
                                 + token_id
                                 + static_cast<int64_t>(scale_block_idx)
                                       * scale_stride_block;
    scale_out[scale_offset] = scale;
  }
}

template <typename input_t>
void launch_deepseek_v4_inv_rope_fp8_quant(
    TensorView fp8_out,
    TensorView scale_out,
    TensorView o,
    TensorView positions,
    TensorView cos_sin_cache,
    int num_tokens,
    int aligned_tokens,
    int num_heads,
    int heads_per_group,
    int rope_abs_start,
    int half_rope,
    cudaStream_t stream) {
  if (aligned_tokens == 0 || num_heads == 0) {
    return;
  }
  dim3 const grid(aligned_tokens, num_heads);
  deepseek_v4_inv_rope_fp8_quant_kernel<input_t>
      <<<grid, kBlockThreadsForInvRopeQuant, 0, stream>>>(
          static_cast<__nv_fp8_e4m3*>(fp8_out.data_ptr()),
          static_cast<float*>(scale_out.data_ptr()),
          static_cast<const input_t*>(o.data_ptr()),
          static_cast<const int64_t*>(positions.data_ptr()),
          static_cast<const float*>(cos_sin_cache.data_ptr()),
          num_tokens,
          heads_per_group,
          rope_abs_start,
          half_rope,
          o.stride(0),
          o.stride(1),
          cos_sin_cache.stride(0),
          fp8_out.stride(0),
          fp8_out.stride(1),
          scale_out.stride(0),
          scale_out.stride(2));
  cudaError_t const status = cudaGetLastError();
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "sm12x_deepseek_v4_inv_rope_fp8_quant launch failed: "
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

void sm12x_deepseek_v4_inv_rope_fp8_quant(
    TensorView fp8_out,
    TensorView scale_out,
    TensorView o,
    TensorView positions,
    TensorView cos_sin_cache,
    int64_t heads_per_group,
    int64_t nope_dim,
    int64_t rope_dim) {
  // `fp8_out`: [n_groups, T_aligned, hidden] (contig) -- caller computes
  //            T_aligned = round_up(num_tokens, 4) and passes the
  //            underlying [G, T_aligned, H] storage so the kernel's
  //            T_aligned slot for padded tokens can have its scale
  //            zeroed deterministically.
  // `scale_out`: strided view (n_groups, num_tokens, scale_blocks) with
  //              strides (scale_blocks * T_aligned, 1, T_aligned) -- the
  //              `stride_token = 1` axis is what lets the downstream FP8
  //              einsum walk per-token scales with a single contiguous
  //              load.
  // `o`: [num_tokens, num_heads, head_dim] bfloat16 (CUDA tensor; arbitrary
  //      strides for token/head, hidden contig).
  // `positions`: [num_tokens] int64.
  // `cos_sin_cache`: [max_pos, >= rope_dim] fp32.
  CHECK_CUDA(fp8_out);
  CHECK_CUDA(scale_out);
  CHECK_CUDA(o);
  CHECK_CUDA(positions);
  CHECK_CUDA(cos_sin_cache);
  CHECK_DEVICE(fp8_out, scale_out);
  CHECK_DEVICE(fp8_out, o);
  CHECK_DEVICE(fp8_out, positions);
  CHECK_DEVICE(fp8_out, cos_sin_cache);
  CHECK_DIM(3, fp8_out);
  CHECK_DIM(3, scale_out);
  CHECK_DIM(3, o);
  CHECK_DIM(1, positions);
  CHECK_DIM(2, cos_sin_cache);
  TVM_FFI_ICHECK_EQ(fp8_out.dtype(), dl_float8_e4m3fn);
  TVM_FFI_ICHECK_EQ(scale_out.dtype(), dl_float32);
  TVM_FFI_ICHECK_EQ(o.dtype(), dl_bfloat16);
  TVM_FFI_ICHECK_EQ(positions.dtype(), dl_int64);
  TVM_FFI_ICHECK_EQ(cos_sin_cache.dtype(), dl_float32);

  int64_t const num_tokens = o.size(0);
  int64_t const num_heads = o.size(1);
  int64_t const head_dim = o.size(2);
  TVM_FFI_ICHECK_EQ(head_dim, nope_dim + rope_dim)
      << "head_dim must equal nope_dim + rope_dim";
  TVM_FFI_ICHECK_EQ(head_dim, kHeadDim)
      << "kernel hard-codes head_dim=" << kHeadDim;
  TVM_FFI_ICHECK(rope_dim > 0 && rope_dim % 2 == 0)
      << "rope_dim must be even and positive";
  TVM_FFI_ICHECK_EQ(head_dim % kQuantGroupSize, 0)
      << "head_dim must be divisible by " << kQuantGroupSize;
  TVM_FFI_ICHECK(heads_per_group > 0);
  TVM_FFI_ICHECK_EQ(num_heads % heads_per_group, 0)
      << "num_heads must be divisible by heads_per_group";

  TVM_FFI_ICHECK_EQ(o.stride(2), 1)
      << "o hidden dim must be contiguous";
  TVM_FFI_ICHECK_EQ(cos_sin_cache.size(1) >= rope_dim, true);

  int64_t const n_groups = num_heads / heads_per_group;
  int64_t const hidden = heads_per_group * head_dim;
  int64_t const scale_blocks = hidden / kQuantGroupSize;

  TVM_FFI_ICHECK_EQ(fp8_out.size(0), n_groups);
  // fp8_out.size(1) is T_aligned (>= num_tokens).
  int64_t const aligned_tokens = fp8_out.size(1);
  TVM_FFI_ICHECK_GE(aligned_tokens, num_tokens);
  TVM_FFI_ICHECK_EQ(aligned_tokens % 4, 0)
      << "T_aligned must be a multiple of 4";
  TVM_FFI_ICHECK_EQ(fp8_out.size(2), hidden);

  TVM_FFI_ICHECK_EQ(scale_out.size(0), n_groups);
  TVM_FFI_ICHECK_EQ(scale_out.size(1), num_tokens);
  TVM_FFI_ICHECK_EQ(scale_out.size(2), scale_blocks);

  TVM_FFI_ICHECK_EQ(positions.size(0), num_tokens);

  // rope_abs_start: the last `rope_dim` elements of head_dim must land
  // in the last quant chunk's tail so a single contiguous block-128
  // scale covers the rope-rotated region.
  int const rope_start_in_chunk = static_cast<int>(nope_dim % kQuantGroupSize);
  TVM_FFI_ICHECK_EQ(rope_start_in_chunk + rope_dim, kQuantGroupSize)
      << "DeepSeek V4 inverse-RoPE block layout is unsupported";
  int const rope_abs_start = (kChunksPerHead - 1) * kQuantGroupSize
                             + rope_start_in_chunk;
  int const half_rope = static_cast<int>(rope_dim / 2);

  if (num_heads == 0) {
    return;
  }

  cudaSetDevice(fp8_out.device().device_id);
  cudaStream_t const stream = get_stream(fp8_out.device());
  launch_deepseek_v4_inv_rope_fp8_quant<nv_bfloat16>(
      fp8_out,
      scale_out,
      o,
      positions,
      cos_sin_cache,
      static_cast<int>(num_tokens),
      static_cast<int>(aligned_tokens),
      static_cast<int>(num_heads),
      static_cast<int>(heads_per_group),
      rope_abs_start,
      half_rope,
      stream);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(sm12x_deepseek_v4_inv_rope_fp8_quant,
                              sm12x_deepseek_v4_inv_rope_fp8_quant);
