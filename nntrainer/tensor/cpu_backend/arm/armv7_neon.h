// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file   cpu_backend.h
 * @date   16 August 2024
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Conditional header file to support unsupported intrinsics on armv7l
 *
 */

#include <arm_neon.h>
#include <cmath>

/**
 * @brief macro for vfmaq_n_f32
 *
 */
#define vfmaq_n_f32(a, b, n) vaddq_f32(a, vmulq_f32(b, vmovq_n_f32(n)))

/**
 * @brief vdivq_f32 macro
 *
 * @param a a for a / b
 * @param b b for a / b
 * @return float32x4_t
 */
static inline float32x4_t vdivq_f32(float32x4_t a, float32x4_t b) {
  float32x4_t ret;
  for (unsigned int i = 0; i < 4; ++i) {
    ret[i] = a[i] / b[i];
  }
  return ret;
}

/**
 * @brief vsqrtq_f32 macro
 *
 * @param a input vector
 * @return float32x4_t
 */
static inline float32x4_t vsqrtq_f32(float32x4_t a) {
  float32x4_t ret;
  for (unsigned int i = 0; i < 4; ++i) {
    ret[i] = std::sqrt(a[i]);
  }
  return ret;
}

/**
 * @brief vmaxvq_f32 macro
 *
 * @param a input vector
 * @return float
 */
static inline float vmaxvq_f32(float32x4_t a) {
  float ret = a[0];
  for (unsigned int i = 1; i < 4; ++i) {
    if (ret > a[i])
      ret = a[i];
  }
  return ret;
}

/**
 * @brief vaddvq_f32
 *
 * @param a input vector
 * @return float32_t
 */
static inline float32_t vaddvq_f32(float32x4_t a) {
  float32_t ret = a[0];
  for (unsigned int i = 1; i < 4; ++i) {
    ret += a[i];
  }
  return ret;
}

/**
 * @brief vaddvq_f32
 *
 * @param a input vector
 * @return uint32_t
 */
static inline uint32_t vaddvq_u32(uint32x4_t a) {
  uint32_t ret = a[0];
  for (unsigned int i = 1; i < 4; ++i) {
    ret += a[i];
  }
  return ret;
}

/**
 * @brief vcvtnq_s32_f32
 *
 * @param a input vector
 * @return int32x4_t
 */
static inline int32x4_t vcvtnq_s32_f32(float32x4_t a) {
  int32x4_t ret;
  for (unsigned int i = 0; i < 4; ++i) {
    ret[i] = std::lround(a[i]);
  }
  return ret;
}

/**
 * @brief vuzp1q_u16 macro
 *
 * @param a input vector
 * @param b input vector
 * @return uint16x8_t
 */
static inline uint16x8_t vuzp1q_u16(uint16x8_t a, uint16x8_t b) {
  uint16x8x2_t t = vuzpq_u16(a, b);
  return t.val[0];
}

/**
 * @brief vuzp2q_u16 macro
 *
 * @param a input vector
 * @param b input vector
 * @return uint16x8_t
 */
static inline uint16x8_t vuzp2q_u16(uint16x8_t a, uint16x8_t b) {
  uint16x8x2_t t = vuzpq_u16(a, b);
  return t.val[1];
}

/**
 * @brief vzip1q_u64 macro
 *
 * @param a input vector
 * @param b input vector
 * @return uint64x2_t
 */
static inline uint64x2_t vzip1q_u64(uint64x2_t a, uint64x2_t b) {
  uint64x1_t al = vget_low_u64(a);
  uint64x1_t bl = vget_low_u64(b);
  return vcombine_u64(al, bl);
}

/**
 * @brief vzip2q_u64 macro
 *
 * @param a input vector
 * @param b input vector
 * @return uint64x2_t
 */
static inline uint64x2_t vzip2q_u64(uint64x2_t a, uint64x2_t b) {
  uint64x1_t ah = vget_high_u64(a);
  uint64x1_t bh = vget_high_u64(b);
  return vcombine_u64(ah, bh);
}

#ifdef ENABLE_FP16
/**
 * @brief macro for vfmaq_n_f16
 *
 */
#define vfmaq_n_f16(a, b, c, n) vaddq_f16(a, vmulq_f16(b, vmovq_n_f16(c[n])))

/**
 * @brief vmaxvq_f16 macro
 *
 * @param a input vector
 * @return float16_t
 */
static inline float16_t vmaxvq_f16(float16x8_t a) {
  float16_t ret = a[0];
  for (unsigned int i = 1; i < 8; ++i) {
    if (ret > a[i])
      ret = a[i];
  }
  return ret;
}
#endif
