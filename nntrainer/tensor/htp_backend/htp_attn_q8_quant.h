// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd.
 * @file htp_attn_q8_quant.h
 * @brief Producer quantization for the A8/KV8 transport.
 */
#ifndef __NNTRAINER_HTP_ATTN_Q8_QUANT_H__
#define __NNTRAINER_HTP_ATTN_Q8_QUANT_H__

#include "hmx/hexkl_kv_q.h"
#include <cmath>
#include <cstdint>
#include <cstring>
#if defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace nntrainer {
/** @brief Per-head UINT8 query, matching the decode kernel's RNE rule. */
inline bool htp_quant_query_u8(const float *x, unsigned int n, uint8_t *out,
                               float &scale, int32_t &zp) {
  float lo = 0.0f, hi = 0.0f;
  for (unsigned int i = 0; i < n; ++i) {
    if (!std::isfinite(x[i]))
      return false;
    lo = x[i] < lo ? x[i] : lo;
    hi = x[i] > hi ? x[i] : hi;
  }
  scale = lo == hi ? 1.0f : (hi - lo) / 255.0f;
  if (!(scale > 0.0f) || !std::isfinite(scale))
    return false;
  zp = static_cast<int32_t>(std::rint(-lo / scale));
  zp = zp < 0 ? 0 : zp > 255 ? 255 : zp;
  const float inv = 1.0f / scale;
  unsigned int i = 0;
#if defined(__aarch64__) && defined(__ARM_NEON)
  for (; i + 8 <= n; i += 8) {
    const int32x4_t a = vaddq_s32(
      vcvtnq_s32_f32(vmulq_n_f32(vld1q_f32(x + i), inv)), vdupq_n_s32(zp));
    const int32x4_t b = vaddq_s32(
      vcvtnq_s32_f32(vmulq_n_f32(vld1q_f32(x + i + 4), inv)), vdupq_n_s32(zp));
    vst1_u8(out + i, vqmovn_u16(vcombine_u16(vqmovun_s32(a), vqmovun_s32(b))));
  }
#endif
  for (; i < n; ++i) {
    int32_t q = static_cast<int32_t>(std::rint(x[i] * inv)) + zp;
    out[i] = static_cast<uint8_t>(q < 0 ? 0 : q > 255 ? 255 : q);
  }
  return true;
}

/** @brief Per-token/head symmetric INT8 K or V, from existing FP16 bits. */
inline bool htp_quant_kv_i8(const uint16_t *bits, unsigned int n, uint8_t *out,
                            float &scale) {
  if (n > HEXKL_KV_Q_MAX_HEAD_DIM)
    return false;
  float x[HEXKL_KV_Q_MAX_HEAD_DIM];
  unsigned int i = 0;
#if defined(__aarch64__) && defined(__ARM_NEON)
  for (; i + 4 <= n; i += 4)
    vst1q_f32(x + i, vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(bits + i))));
#endif
  for (; i < n; ++i)
    x[i] = hexkl_kv_q_hf_to_f32(bits[i]);
  float peak = 0.0f;
  for (i = 0; i < n; ++i) {
    if (!std::isfinite(x[i]))
      return false;
    const float a = std::fabs(x[i]);
    peak = a > peak ? a : peak;
  }
  if (peak == 0.0f) {
    scale = 1.0f;
    std::memset(out, 0, n);
    return true;
  }
  scale = peak / 127.0f;
  const float inv = 127.0f / peak;
  i = 0;
#if defined(__aarch64__) && defined(__ARM_NEON)
  for (; i + 8 <= n; i += 8) {
    const int32x4_t a = vcvtnq_s32_f32(vmulq_n_f32(vld1q_f32(x + i), inv));
    const int32x4_t b = vcvtnq_s32_f32(vmulq_n_f32(vld1q_f32(x + i + 4), inv));
    const int8x8_t q = vqmovn_s16(vcombine_s16(vqmovn_s32(a), vqmovn_s32(b)));
    vst1_u8(out + i, vreinterpret_u8_s8(vmax_s8(q, vdup_n_s8(-127))));
  }
#endif
  for (; i < n; ++i) {
    int32_t q = static_cast<int32_t>(std::rint(x[i] * inv));
    q = q < -127 ? -127 : q > 127 ? 127 : q;
    out[i] = static_cast<uint8_t>(static_cast<int8_t>(q));
  }
  return true;
}
} // namespace nntrainer
#endif
