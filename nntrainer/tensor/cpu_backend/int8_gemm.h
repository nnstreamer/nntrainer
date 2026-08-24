// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   int8_gemm.h
 * @date   24 August 2026
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  W8A8 GEMM with channel-wise weight scales and a tensor-wise
 * activation scale.
 *
 * The weight is a Q8_0 byte stream whose blocks all carry the same scale within
 * an output channel (written by nntr_quantize_q8_0_per_channel), so there is
 * exactly one weight scale per channel. The activation is quantized with a
 * single scale for the whole tensor. Together those two properties make the
 * inner loop a pure int8 dot product accumulating into int32, with one scalar
 * multiply per output element at the end -- no per-block scale arithmetic,
 * which is what the block-scaled Q8_0 x Q8_0 kernel has to carry.
 */

#ifndef __NNTR_INT8_GEMM_H__
#define __NNTR_INT8_GEMM_H__

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>

#include <thread_manager.h>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

namespace nntrainer {
namespace int8_gemm {

/** Q8_0 block geometry: fp16 scale followed by 32 int8 quants. */
constexpr unsigned int QK = 32;
constexpr unsigned int BLOCK_BYTES = 2 + QK;

/**
 * @brief Decode an fp16 bit pattern to FP32.
 */
inline float fp16BitsToFp32(uint16_t h) {
  const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
  const uint32_t exp = (h >> 10) & 0x1fu;
  const uint32_t man = h & 0x3ffu;
  uint32_t f;
  if (exp == 0)
    f = sign | (man ? (0x38800000u | (man << 13)) : 0u);
  else if (exp == 31)
    f = sign | 0x7f800000u | (man << 13);
  else
    f = sign | ((exp + 112u) << 23) | (man << 13);
  float out;
  std::memcpy(&out, &f, 4);
  return out;
}

/**
 * @brief A weight prepared once for the channel-wise kernel.
 */
struct PerChannelWeight {
  bool usable = false;      /**< false when the stream is not channel-wise */
  std::vector<int8_t> q;    /**< [N, K] int8, scale bytes stripped out */
  std::vector<float> scale; /**< [N] one scale per output channel */
};

/**
 * @brief Prepare (and cache) a weight for the channel-wise kernel.
 *
 * Two things happen once per weight rather than once per call: the per-channel
 * scale is read out, and the int8 values are gathered into a contiguous run per
 * channel so the dot product does not have to step over the 2-byte scale every
 * 32 values. Cached by weight pointer, the same pattern the conv layer uses for
 * its repacked filters.
 *
 * `usable` is false when any row's blocks disagree on their scale, i.e. the
 * stream is a genuine per-block Q8_0 file. Callers must then fall back to the
 * block-scaled kernel: reading it as channel-wise would silently drop every
 * scale but the first of each row.
 */
inline const PerChannelWeight &prepareWeight(const void *B, unsigned int N,
                                             unsigned int K) {
  static std::mutex mtx;
  static std::unordered_map<const void *, PerChannelWeight> cache;
  std::lock_guard<std::mutex> lk(mtx);
  auto it = cache.find(B);
  if (it != cache.end())
    return it->second;

  const unsigned int nb = K / QK;
  const char *base = static_cast<const char *>(B);
  PerChannelWeight w;

  if (nb == 0 || K % QK != 0)
    return cache.emplace(B, std::move(w)).first->second;

  w.scale.resize(N);
  w.q.resize((size_t)N * K);
  for (unsigned int n = 0; n < N; ++n) {
    const char *src = base + (size_t)n * nb * BLOCK_BYTES;
    uint16_t d0;
    std::memcpy(&d0, src, 2);
    for (unsigned int b = 1; b < nb; ++b) {
      uint16_t d;
      std::memcpy(&d, src + (size_t)b * BLOCK_BYTES, 2);
      if (d != d0) {
        w.q.clear();
        w.q.shrink_to_fit();
        w.scale.clear();
        return cache.emplace(B, std::move(w)).first->second;
      }
    }
    w.scale[n] = fp16BitsToFp32(d0);
    int8_t *dst = w.q.data() + (size_t)n * K;
    for (unsigned int b = 0; b < nb; ++b)
      std::memcpy(dst + (size_t)b * QK, src + (size_t)b * BLOCK_BYTES + 2, QK);
  }
  w.usable = true;
  return cache.emplace(B, std::move(w)).first->second;
}

/**
 * @brief int8 dot product of `len` elements accumulating in int32.
 */
inline int32_t dotI8(const int8_t *a, const int8_t *b, unsigned int len) {
#if defined(__ARM_FEATURE_DOTPROD)
  int32x4_t acc0 = vdupq_n_s32(0);
  int32x4_t acc1 = vdupq_n_s32(0);
  unsigned int i = 0;
  for (; i + 32 <= len; i += 32) {
    acc0 = vdotq_s32(acc0, vld1q_s8(a + i), vld1q_s8(b + i));
    acc1 = vdotq_s32(acc1, vld1q_s8(a + i + 16), vld1q_s8(b + i + 16));
  }
  for (; i + 16 <= len; i += 16)
    acc0 = vdotq_s32(acc0, vld1q_s8(a + i), vld1q_s8(b + i));
  int32_t sum = vaddvq_s32(vaddq_s32(acc0, acc1));
  for (; i < len; ++i)
    sum += (int32_t)a[i] * (int32_t)b[i];
  return sum;
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
  int32x4_t acc = vdupq_n_s32(0);
  unsigned int i = 0;
  for (; i + 16 <= len; i += 16) {
    const int8x16_t va = vld1q_s8(a + i);
    const int8x16_t vb = vld1q_s8(b + i);
    acc = vpadalq_s16(acc, vmull_s8(vget_low_s8(va), vget_low_s8(vb)));
    acc = vpadalq_s16(acc, vmull_s8(vget_high_s8(va), vget_high_s8(vb)));
  }
  int32_t sum = vaddvq_s32(acc);
  for (; i < len; ++i)
    sum += (int32_t)a[i] * (int32_t)b[i];
  return sum;
#else
  int32_t sum = 0;
  for (unsigned int i = 0; i < len; ++i)
    sum += (int32_t)a[i] * (int32_t)b[i];
  return sum;
#endif
}

/**
 * @brief Quantize a whole activation block with one shared scale.
 *
 * @return the scale that maps the int8 values back to FP32
 */
inline float quantizeActivation(const float *A, size_t count,
                                std::vector<int8_t> &out) {
  float amax = 0.0f;
  for (size_t i = 0; i < count; ++i)
    amax = std::max(amax, std::fabs(A[i]));
  const float scale = amax > 0.0f ? amax / 127.0f : 1.0f;
  const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;

  out.resize(count);
  int8_t *q = out.data();
  for (size_t i = 0; i < count; ++i) {
    const float r = std::round(A[i] * inv);
    q[i] = (int8_t)std::max(-127.0f, std::min(127.0f, r));
  }
  return scale;
}

/**
 * @brief W8A8 GEMM: FP32 activation [M, K] x channel-wise Q8_0 weight [N, K]
 * -> FP32 output [M, N].
 *
 * @param M activation rows
 * @param N output channels
 * @param K reduction length, must be a multiple of 32
 * @param A activation, row-major [M, K]
 * @param B weight, Q8_0 blocks laid out per output channel
 * @param C output, row-major [M, N]
 * @retval false the weight is not channel-wise; nothing was written and the
 *         caller must use the block-scaled kernel
 */
inline bool gemmPerChannelA8(unsigned int M, unsigned int N, unsigned int K,
                             const float *A, const void *B, float *C) {
  const PerChannelWeight &w = prepareWeight(B, N, K);
  if (!w.usable)
    return false;

  // One scratch buffer per thread, reused across calls: the quantized
  // activation is the only per-call allocation this kernel would otherwise do.
  static thread_local std::vector<int8_t> scratch;
  const float a_scale = quantizeActivation(A, (size_t)M * K, scratch);
  const int8_t *Aq = scratch.data();

  auto &tm = ThreadManager::Global();
  tm.parallel_for(0, N, [&](size_t n) {
    const int8_t *wq = w.q.data() + (size_t)n * K;
    const float s = a_scale * w.scale[n];
    for (unsigned int m = 0; m < M; ++m)
      C[(size_t)m * N + n] = (float)dotI8(Aq + (size_t)m * K, wq, K) * s;
  });
  return true;
}

} // namespace int8_gemm
} // namespace nntrainer

#endif /* __NNTR_INT8_GEMM_H__ */
