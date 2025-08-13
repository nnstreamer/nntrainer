// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file   nntr_ggml_impl.cpp
 * @date   13 August 2025
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Custom-implemented functions to support ggml functions for internal
 * uses in nntrainer
 */

#include <algorithm>
#include <assert.h>
#include <cstring>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <tensor_dim.h>

#include <nntr_ggml_impl.h>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#elif defined (__ARM_ARCH_7A__) || defined (__arm__) || ARMV7
#include <armv7_neon.h>
#elif defined(__AVX2__) || defined(__AVX__)
#include <immintrin.h>
#endif // !defined(__aarch64__)

#ifndef MAX
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#endif
#ifndef MIN
#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#endif

struct block_q6_K {
  uint8_t ql[128];   // quants, lower 4 bits
  uint8_t qh[64];    // quants, upper 2 bits
  int8_t scales[16]; // scales, quantized with 8 bits
  uint16_t d;        // super-block scale
};

typedef struct {
  float d;                 // delta
  int8_t qs[256];          // quants
  int16_t bsums[256 / 16]; // sum of quants in groups of 16
} block_q8_K;

#define Q4_0 32
#define Q8_0 32

#if defined(__aarch64__)
static inline float nntr_compute_fp16_to_fp32(uint16_t h) {
  __fp16 tmp;
  memcpy(&tmp, &h, sizeof(uint16_t));
  return (float)tmp;
}

static inline uint16_t nntr_compute_fp32_to_fp16(float f) {
  uint16_t res;
  __fp16 tmp = f;
  memcpy(&res, &tmp, sizeof(uint16_t));
  return res;
}
#else
static inline float fp32_from_bits(uint32_t w) {
  union {
    uint32_t as_bits;
    float as_value;
  } fp32;
  fp32.as_bits = w;
  return fp32.as_value;
}

static inline uint32_t fp32_to_bits(float f) {
  union {
    float as_value;
    uint32_t as_bits;
  } fp32;
  fp32.as_value = f;
  return fp32.as_bits;
}

static inline float nntr_compute_fp16_to_fp32(uint16_t h) {
  const uint32_t w = (uint32_t)h << 16;
  const uint32_t sign = w & UINT32_C(0x80000000);
  const uint32_t two_w = w + w;

  const uint32_t exp_offset = UINT32_C(0xE0) << 23;
#if (defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) ||             \
     defined(__GNUC__) && !defined(__STRICT_ANSI__)) &&                        \
  (!defined(__cplusplus) || __cplusplus >= 201703L)
  const float exp_scale = 0x1.0p-112f;
#else
  const float exp_scale = fp32_from_bits(UINT32_C(0x7800000));
#endif
  const float normalized_value =
    fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;

  const uint32_t magic_mask = UINT32_C(126) << 23;
  const float magic_bias = 0.5f;
  const float denormalized_value =
    fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;

  const uint32_t denormalized_cutoff = UINT32_C(1) << 27;
  const uint32_t result =
    sign | (two_w < denormalized_cutoff ? fp32_to_bits(denormalized_value)
                                        : fp32_to_bits(normalized_value));
  return fp32_from_bits(result);
}

static inline uint16_t nntr_compute_fp32_to_fp16(float f) {
#if (defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) ||             \
     defined(__GNUC__) && !defined(__STRICT_ANSI__)) &&                        \
  (!defined(__cplusplus) || __cplusplus >= 201703L)
  const float scale_to_inf = 0x1.0p+112f;
  const float scale_to_zero = 0x1.0p-110f;
#else
  const float scale_to_inf = fp32_from_bits(UINT32_C(0x77800000));
  const float scale_to_zero = fp32_from_bits(UINT32_C(0x08800000));
#endif
  float base = (fabsf(f) * scale_to_inf) * scale_to_zero;

  const uint32_t w = fp32_to_bits(f);
  const uint32_t shl1_w = w + w;
  const uint32_t sign = w & UINT32_C(0x80000000);
  uint32_t bias = shl1_w & UINT32_C(0xFF000000);
  if (bias < UINT32_C(0x71000000)) {
    bias = UINT32_C(0x71000000);
  }

  base = fp32_from_bits((bias >> 1) + UINT32_C(0x07800000)) + base;
  const uint32_t bits = fp32_to_bits(base);
  const uint32_t exp_bits = (bits >> 13) & UINT32_C(0x00007C00);
  const uint32_t mantissa_bits = bits & UINT32_C(0x00000FFF);
  const uint32_t nonsign = exp_bits + mantissa_bits;
  return (sign >> 16) |
         (shl1_w > UINT32_C(0xFF000000) ? UINT16_C(0x7E00) : nonsign);
}
#endif

/**
 * @brief struct template for q4_0 and q8_0
 *
 * @tparam K 4 or 8
 * @return constexpr int number of elements in the quantized block
 */
template <int K> constexpr int QK_0() {
  if constexpr (K == 4) {
    return Q4_0;
  }
  if constexpr (K == 8) {
    return Q8_0;
  }
  return -1;
}

/**
 * @brief block of q4_0 or q8_0 block
 *
 * @tparam K 4 or 8
 * @tparam N number of blocks to be packed
 */
template <int K, int N> struct block {
  uint16_t d[N];                      // deltas for N qK_0 blocks
  int8_t qs[(QK_0<K>() * N * K) / 8]; // quants for N qK_0 blocks
};

using block_q4_0x4 = block<4, 4>;
using block_q8_0x4 = block<8, 4>;

typedef struct {
  uint16_t d;      // delta
  int8_t qs[Q8_0]; // quants
} block_q8_0;

void nntr_gemv_q4_0_4x8_q8_0(int n, float *__restrict s, size_t bs,
                             const void *__restrict vx,
                             const void *__restrict vy, int nr, int nc) {
  const int qk = Q8_0;
  const int nb = n / qk;
  const int ncols_interleaved = 4;
  const int blocklen = 8;

  assert(n % qk == 0);
  assert(nc % ncols_interleaved == 0);
#if !((defined(_MSC_VER)) && !defined(__clang__)) && defined(__aarch64__) &&   \
  defined(__aarch64__)
#if defined(__ARM_FEATURE_DOTPROD)
  const block_q4_0x4 *b_ptr = (const block_q4_0x4 *)vx;
  for (int c = 0; c < nc; c += ncols_interleaved) {
    const block_q8_0 *a_ptr = (const block_q8_0 *)vy;
    float32x4_t acc = vdupq_n_f32(0);
    for (int b = 0; b < nb; b++) {
      int8x16_t b0 = vld1q_s8((const int8_t *)b_ptr->qs);
      int8x16_t b1 = vld1q_s8((const int8_t *)b_ptr->qs + 16);
      int8x16_t b2 = vld1q_s8((const int8_t *)b_ptr->qs + 32);
      int8x16_t b3 = vld1q_s8((const int8_t *)b_ptr->qs + 48);
      float16x4_t bd = vld1_f16((const __fp16 *)b_ptr->d);

      int8x16_t a0 = (int8x16_t)vld1q_dup_s64((const int64_t *)a_ptr->qs);
      int8x16_t a1 = (int8x16_t)vld1q_dup_s64((const int64_t *)a_ptr->qs + 1);
      int8x16_t a2 = (int8x16_t)vld1q_dup_s64((const int64_t *)a_ptr->qs + 2);
      int8x16_t a3 = (int8x16_t)vld1q_dup_s64((const int64_t *)a_ptr->qs + 3);
      float16x4_t ad = vld1_dup_f16((const __fp16 *)&a_ptr->d);

      int32x4_t ret0 = vdupq_n_s32(0);
      int32x4_t ret1 = vdupq_n_s32(0);

      ret0 = vdotq_s32(ret0, b0 << 4, a0);
      ret1 = vdotq_s32(ret1, b1 << 4, a0);
      ret0 = vdotq_s32(ret0, b2 << 4, a1);
      ret1 = vdotq_s32(ret1, b3 << 4, a1);

      ret0 = vdotq_s32(ret0, b0 & 0xf0U, a2);
      ret1 = vdotq_s32(ret1, b1 & 0xf0U, a2);
      ret0 = vdotq_s32(ret0, b2 & 0xf0U, a3);
      ret1 = vdotq_s32(ret1, b3 & 0xf0U, a3);

      int32x4_t ret = vpaddq_s32(ret0, ret1);

      acc = vfmaq_f32(acc, vcvtq_n_f32_s32(ret, 4),
                      vmulq_f32(vcvt_f32_f16(ad), vcvt_f32_f16(bd)));
      a_ptr++;
      b_ptr++;
    }
    vst1q_f32(s, acc);
    s += ncols_interleaved;
  }
  return;

#else
  const void *b_ptr = vx;
  const void *a_ptr = vy;
  float *res_ptr = s;

  __asm__ __volatile__(
    "movi v2.16b, #0x4\n"
    "movi v1.16b, #0xf0\n"
    "add %x[b_ptr], %x[b_ptr], #0x8\n"
    "1:" // Column loop
    "add x23, %x[a_ptr], #0x2\n"
    "movi v0.16b, #0x0\n"
    "mov x22, %x[nb]\n"
    "2:" // Block loop
    "ldr q31, [%x[b_ptr], #0x0]\n"
    "ldr q30, [%x[b_ptr], #0x10]\n"
    "mov x21, x23\n"
    "movi v29.4s, #0x0\n"
    "ldr q28, [%x[b_ptr], #0x20]\n"
    "ldr q27, [%x[b_ptr], #0x30]\n"
    "movi v26.4s, #0x0\n"
    "sub x20, x23, #0x2\n"
    "ld1r { v25.8h }, [x20]\n"
    "ldr q24, [%x[b_ptr], #-0x8]\n"
    "sub x22, x22, #0x1\n"
    "add x23, x23, #0x22\n"
    "ld1r { v23.2d }, [x21], #0x8\n"
    "sshl v22.16b, v31.16b, v2.16b\n"
    "sshl v16.16b, v30.16b, v2.16b\n"
    "add %x[b_ptr], %x[b_ptr], #0x48\n"
    "ld1r { v21.2d }, [x21], #0x8\n"
    "sshl v20.16b, v28.16b, v2.16b\n"
    "sshl v19.16b, v27.16b, v2.16b\n"
    "ld1r { v18.2d }, [x21], #0x8\n"
    "ld1r { v17.2d }, [x21], #0x8\n"
    "and v31.16b, v31.16b, v1.16b\n"
    "and v30.16b, v30.16b, v1.16b\n"
    ".inst 0x4e9796dd  // sdot v29.4s, v22.16b, v23.16b\n"
    ".inst 0x4e97961a  // sdot v26.4s, v16.16b, v23.16b\n"
    "and v28.16b, v28.16b, v1.16b\n"
    "and v27.16b, v27.16b, v1.16b\n"
    "fcvtl v25.4s, v25.4h\n"
    "fcvtl v16.4s, v24.4h\n"
    ".inst 0x4e95969d  // sdot v29.4s, v20.16b, v21.16b\n"
    ".inst 0x4e95967a  // sdot v26.4s, v19.16b, v21.16b\n"
    "fmul v16.4s, v16.4s, v25.4s\n"
    ".inst 0x4e9297fd  // sdot v29.4s, v31.16b, v18.16b\n"
    ".inst 0x4e9297da  // sdot v26.4s, v30.16b, v18.16b\n"
    ".inst 0x4e91979d  // sdot v29.4s, v28.16b, v17.16b\n"
    ".inst 0x4e91977a  // sdot v26.4s, v27.16b, v17.16b\n"
    "addp v29.4s, v29.4s, v26.4s\n"
    "scvtf v29.4s, v29.4s, #0x4\n"
    "fmla v0.4s, v29.4s, v16.4s\n"
    "cbnz x22, 2b\n"
    "sub %x[nc], %x[nc], #0x4\n"
    "str q0, [%x[res_ptr], #0x0]\n"
    "add %x[res_ptr], %x[res_ptr], #0x10\n"
    "cbnz %x[nc], 1b\n"
    : [b_ptr] "+&r"(b_ptr), [res_ptr] "+&r"(res_ptr), [nc] "+&r"(nc)
    : [a_ptr] "r"(a_ptr), [nb] "r"(nb)
    : "memory", "v0", "v1", "v2", "v16", "v17", "v18", "v19", "v20", "v21",
      "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31",
      "x20", "x21", "x22", "x23");
  return;
#endif
#else
  float sumf[4];
  int sumi;

  const block_q8_0 *a_ptr = (const block_q8_0 *)vy;
  for (int x = 0; x < nc / ncols_interleaved; x++) {
    const block_q4_0x4 *b_ptr = (const block_q4_0x4 *)vx + (x * nb);

    for (int j = 0; j < ncols_interleaved; j++)
      sumf[j] = 0.0;
    for (int l = 0; l < nb; l++) {
      for (int k = 0; k < (qk / (2 * blocklen)); k++) {
        for (int j = 0; j < ncols_interleaved; j++) {
          sumi = 0;
          for (int i = 0; i < blocklen; ++i) {
            const int v0 =
              (int8_t)(b_ptr[l].qs[k * ncols_interleaved * blocklen +
                                   j * blocklen + i]
                       << 4);
            const int v1 =
              (int8_t)(b_ptr[l].qs[k * ncols_interleaved * blocklen +
                                   j * blocklen + i] &
                       0xF0);
            sumi += ((v0 * a_ptr[l].qs[k * blocklen + i]) +
                     (v1 * a_ptr[l].qs[k * blocklen + i + qk / 2])) >>
                    4;
          }
          sumf[j] += sumi * nntr_compute_fp16_to_fp32(b_ptr[l].d[j]) *
                     nntr_compute_fp16_to_fp32(a_ptr[l].d);
        }
      }
    }
    for (int j = 0; j < ncols_interleaved; j++)
      s[x * ncols_interleaved + j] = sumf[j];
  }
#endif
}

void nntr_gemm_q4_0_4x8_q8_0(int n, float *__restrict s, size_t bs,
                             const void *__restrict vx,
                             const void *__restrict vy, int nr, int nc) {
  const int qk = Q8_0;
  const int nb = n / qk;
  const int ncols_interleaved = 4;
  const int blocklen = 8;

  assert(n % qk == 0);
  assert(nr % 4 == 0);
  assert(nc % ncols_interleaved == 0);

#if !((defined(_MSC_VER)) && !defined(__clang__)) && defined(__aarch64__) &&   \
  defined(__aarch64__)
  const void *b_ptr = vx;
  const void *a_ptr = vy;
  float *res_ptr = s;
  size_t res_stride = bs * sizeof(float);

  __asm__ __volatile__("mov x10, %x[nr]\n"
                       "mov x9, #0x88\n"
                       "cmp x10, #0x10\n"
                       "mul x9, %x[nb], x9\n"
                       "blt 4f\n"
                       "1:" // Row loop
                       "add x28, %x[b_ptr], #0x8\n"
                       "mov x27, %x[nc]\n"
                       "add x26, %x[res_ptr], %x[res_stride], LSL #4\n"
                       "2:" // Column loop
                       "add x25, %x[a_ptr], #0x8\n"
                       "movi v2.16b, #0x0\n"
                       "movi v10.16b, #0x0\n"
                       "mov x24, %x[nb]\n"
                       "add x23, x25, x9\n"
                       "movi v12.16b, #0x0\n"
                       "movi v28.16b, #0x0\n"
                       "add x22, x23, x9\n"
                       "movi v11.16b, #0x0\n"
                       "movi v13.16b, #0x0\n"
                       "add x21, x22, x9\n"
                       "movi v22.16b, #0x0\n"
                       "movi v23.16b, #0x0\n"
                       "movi v25.16b, #0x0\n"
                       "movi v5.16b, #0x0\n"
                       "movi v7.16b, #0x0\n"
                       "movi v4.16b, #0x0\n"
                       "movi v6.16b, #0x0\n"
                       "movi v30.16b, #0x0\n"
                       "movi v24.16b, #0x0\n"
                       "movi v14.16b, #0x0\n"
                       "3:" // Block loop
                       "ldr q21, [x28, #0x0]\n"
                       "ldr q16, [x28, #0x10]\n"
                       "movi v1.16b, #0x4\n"
                       "movi v19.4s, #0x0\n"
                       "ldr q27, [x25, #0x0]\n"
                       "ldr q15, [x25, #0x10]\n"
                       "movi v26.4s, #0x0\n"
                       "movi v18.4s, #0x0\n"
                       "ldr q29, [x28, #0x20]\n"
                       "ldr q3, [x28, #0x30]\n"
                       "movi v17.4s, #0x0\n"
                       "movi v0.16b, #0xf0\n"
                       "ldr d20, [x25, #-0x8]\n"
                       "ldr d9, [x23, #-0x8]\n"
                       "sshl v8.16b, v21.16b, v1.16b\n"
                       "sshl v31.16b, v16.16b, v1.16b\n"
                       "and v21.16b, v21.16b, v0.16b\n"
                       "and v16.16b, v16.16b, v0.16b\n"
                       "sub x20, x28, #0x8\n"
                       "subs x24, x24, #0x1\n"
                       "add x28, x28, #0x48\n"
                       ".inst 0x4e88a773  // smmla v19.4s, v27.16b, v8.16b\n"
                       ".inst 0x4e9fa77a  // smmla v26.4s, v27.16b, v31.16b\n"
                       "ldr q27, [x25, #0x20]\n"
                       ".inst 0x4e88a5f2  // smmla v18.4s, v15.16b, v8.16b\n"
                       ".inst 0x4e9fa5f1  // smmla v17.4s, v15.16b, v31.16b\n"
                       "sshl v15.16b, v29.16b, v1.16b\n"
                       "sshl v1.16b, v3.16b, v1.16b\n"
                       "and v29.16b, v29.16b, v0.16b\n"
                       "and v3.16b, v3.16b, v0.16b\n"
                       "ldr q0, [x25, #0x30]\n"
                       "fcvtl v20.4s, v20.4h\n"
                       ".inst 0x4e8fa773  // smmla v19.4s, v27.16b, v15.16b\n"
                       "fcvtl v9.4s, v9.4h\n"
                       ".inst 0x4e81a77a  // smmla v26.4s, v27.16b, v1.16b\n"
                       "ldr q27, [x25, #0x40]\n"
                       ".inst 0x4e8fa412  // smmla v18.4s, v0.16b, v15.16b\n"
                       ".inst 0x4e81a411  // smmla v17.4s, v0.16b, v1.16b\n"
                       "ldr q0, [x25, #0x50]\n"
                       ".inst 0x4e95a773  // smmla v19.4s, v27.16b, v21.16b\n"
                       ".inst 0x4e90a77a  // smmla v26.4s, v27.16b, v16.16b\n"
                       "ldr q27, [x25, #0x60]\n"
                       ".inst 0x4e95a412  // smmla v18.4s, v0.16b, v21.16b\n"
                       ".inst 0x4e90a411  // smmla v17.4s, v0.16b, v16.16b\n"
                       "ldr q0, [x25, #0x70]\n"
                       "add x25, x25, #0x88\n"
                       ".inst 0x4e9da773  // smmla v19.4s, v27.16b, v29.16b\n"
                       ".inst 0x4e83a77a  // smmla v26.4s, v27.16b, v3.16b\n"
                       "ldr d27, [x20, #0x0]\n"
                       ".inst 0x4e9da412  // smmla v18.4s, v0.16b, v29.16b\n"
                       ".inst 0x4e83a411  // smmla v17.4s, v0.16b, v3.16b\n"
                       "fcvtl v27.4s, v27.4h\n"
                       "uzp1 v0.2d, v19.2d, v26.2d\n"
                       "uzp2 v26.2d, v19.2d, v26.2d\n"
                       "fmul v19.4s, v27.4s, v20.s[0]\n"
                       "scvtf v0.4s, v0.4s, #0x4\n"
                       "scvtf v26.4s, v26.4s, #0x4\n"
                       "fmla v2.4s, v0.4s, v19.4s\n"
                       "ldr q19, [x23, #0x0]\n"
                       "uzp1 v0.2d, v18.2d, v17.2d\n"
                       "uzp2 v18.2d, v18.2d, v17.2d\n"
                       "fmul v17.4s, v27.4s, v20.s[1]\n"
                       "scvtf v0.4s, v0.4s, #0x4\n"
                       "scvtf v18.4s, v18.4s, #0x4\n"
                       "fmla v10.4s, v26.4s, v17.4s\n"
                       "ldr q17, [x23, #0x10]\n"
                       "fmul v26.4s, v27.4s, v20.s[2]\n"
                       "fmul v20.4s, v27.4s, v20.s[3]\n"
                       "fmla v12.4s, v0.4s, v26.4s\n"
                       "ldr d0, [x22, #-0x8]\n"
                       "ldr d26, [x21, #-0x8]\n"
                       "fcvtl v0.4s, v0.4h\n"
                       "fmla v28.4s, v18.4s, v20.4s\n"
                       "movi v20.4s, #0x0\n"
                       "movi v18.4s, #0x0\n"
                       ".inst 0x4e88a674  // smmla v20.4s, v19.16b, v8.16b\n"
                       ".inst 0x4e9fa672  // smmla v18.4s, v19.16b, v31.16b\n"
                       "ldr q19, [x23, #0x20]\n"
                       "fcvtl v26.4s, v26.4h\n"
                       ".inst 0x4e8fa674  // smmla v20.4s, v19.16b, v15.16b\n"
                       ".inst 0x4e81a672  // smmla v18.4s, v19.16b, v1.16b\n"
                       "ldr q19, [x23, #0x40]\n"
                       ".inst 0x4e95a674  // smmla v20.4s, v19.16b, v21.16b\n"
                       ".inst 0x4e90a672  // smmla v18.4s, v19.16b, v16.16b\n"
                       "ldr q19, [x23, #0x60]\n"
                       ".inst 0x4e9da674  // smmla v20.4s, v19.16b, v29.16b\n"
                       ".inst 0x4e83a672  // smmla v18.4s, v19.16b, v3.16b\n"
                       "uzp1 v19.2d, v20.2d, v18.2d\n"
                       "scvtf v19.4s, v19.4s, #0x4\n"
                       "uzp2 v20.2d, v20.2d, v18.2d\n"
                       "fmul v18.4s, v27.4s, v9.s[0]\n"
                       "scvtf v20.4s, v20.4s, #0x4\n"
                       "fmla v11.4s, v19.4s, v18.4s\n"
                       "ldr q18, [x22, #0x0]\n"
                       "fmul v19.4s, v27.4s, v9.s[1]\n"
                       "fmla v13.4s, v20.4s, v19.4s\n"
                       "movi v19.4s, #0x0\n"
                       "movi v20.4s, #0x0\n"
                       ".inst 0x4e88a633  // smmla v19.4s, v17.16b, v8.16b\n"
                       ".inst 0x4e9fa634  // smmla v20.4s, v17.16b, v31.16b\n"
                       "ldr q17, [x23, #0x30]\n"
                       ".inst 0x4e8fa633  // smmla v19.4s, v17.16b, v15.16b\n"
                       ".inst 0x4e81a634  // smmla v20.4s, v17.16b, v1.16b\n"
                       "ldr q17, [x23, #0x50]\n"
                       ".inst 0x4e95a633  // smmla v19.4s, v17.16b, v21.16b\n"
                       ".inst 0x4e90a634  // smmla v20.4s, v17.16b, v16.16b\n"
                       "ldr q17, [x23, #0x70]\n"
                       "add x23, x23, #0x88\n"
                       ".inst 0x4e9da633  // smmla v19.4s, v17.16b, v29.16b\n"
                       ".inst 0x4e83a634  // smmla v20.4s, v17.16b, v3.16b\n"
                       "uzp1 v17.2d, v19.2d, v20.2d\n"
                       "scvtf v17.4s, v17.4s, #0x4\n"
                       "uzp2 v20.2d, v19.2d, v20.2d\n"
                       "fmul v19.4s, v27.4s, v9.s[2]\n"
                       "fmul v9.4s, v27.4s, v9.s[3]\n"
                       "scvtf v20.4s, v20.4s, #0x4\n"
                       "fmla v22.4s, v17.4s, v19.4s\n"
                       "ldr q17, [x22, #0x10]\n"
                       "movi v19.4s, #0x0\n"
                       ".inst 0x4e88a653  // smmla v19.4s, v18.16b, v8.16b\n"
                       "fmla v23.4s, v20.4s, v9.4s\n"
                       "movi v20.4s, #0x0\n"
                       "movi v9.4s, #0x0\n"
                       ".inst 0x4e9fa654  // smmla v20.4s, v18.16b, v31.16b\n"
                       "ldr q18, [x22, #0x20]\n"
                       ".inst 0x4e88a629  // smmla v9.4s, v17.16b, v8.16b\n"
                       ".inst 0x4e8fa653  // smmla v19.4s, v18.16b, v15.16b\n"
                       ".inst 0x4e81a654  // smmla v20.4s, v18.16b, v1.16b\n"
                       "ldr q18, [x22, #0x40]\n"
                       ".inst 0x4e95a653  // smmla v19.4s, v18.16b, v21.16b\n"
                       ".inst 0x4e90a654  // smmla v20.4s, v18.16b, v16.16b\n"
                       "ldr q18, [x22, #0x60]\n"
                       ".inst 0x4e9da653  // smmla v19.4s, v18.16b, v29.16b\n"
                       ".inst 0x4e83a654  // smmla v20.4s, v18.16b, v3.16b\n"
                       "movi v18.4s, #0x0\n"
                       ".inst 0x4e9fa632  // smmla v18.4s, v17.16b, v31.16b\n"
                       "ldr q17, [x22, #0x30]\n"
                       ".inst 0x4e8fa629  // smmla v9.4s, v17.16b, v15.16b\n"
                       ".inst 0x4e81a632  // smmla v18.4s, v17.16b, v1.16b\n"
                       "ldr q17, [x22, #0x50]\n"
                       ".inst 0x4e95a629  // smmla v9.4s, v17.16b, v21.16b\n"
                       ".inst 0x4e90a632  // smmla v18.4s, v17.16b, v16.16b\n"
                       "ldr q17, [x22, #0x70]\n"
                       "add x22, x22, #0x88\n"
                       ".inst 0x4e9da629  // smmla v9.4s, v17.16b, v29.16b\n"
                       ".inst 0x4e83a632  // smmla v18.4s, v17.16b, v3.16b\n"
                       "uzp1 v17.2d, v19.2d, v20.2d\n"
                       "uzp2 v20.2d, v19.2d, v20.2d\n"
                       "fmul v19.4s, v27.4s, v0.s[0]\n"
                       "scvtf v17.4s, v17.4s, #0x4\n"
                       "scvtf v20.4s, v20.4s, #0x4\n"
                       "fmla v25.4s, v17.4s, v19.4s\n"
                       "ldr q19, [x21, #0x0]\n"
                       "fmul v17.4s, v27.4s, v0.s[1]\n"
                       "fmla v5.4s, v20.4s, v17.4s\n"
                       "ldr q17, [x21, #0x10]\n"
                       "uzp1 v20.2d, v9.2d, v18.2d\n"
                       "uzp2 v9.2d, v9.2d, v18.2d\n"
                       "fmul v18.4s, v27.4s, v0.s[2]\n"
                       "fmul v0.4s, v27.4s, v0.s[3]\n"
                       "scvtf v20.4s, v20.4s, #0x4\n"
                       "scvtf v9.4s, v9.4s, #0x4\n"
                       "fmla v7.4s, v20.4s, v18.4s\n"
                       "movi v20.4s, #0x0\n"
                       "movi v18.4s, #0x0\n"
                       ".inst 0x4e88a674  // smmla v20.4s, v19.16b, v8.16b\n"
                       ".inst 0x4e9fa672  // smmla v18.4s, v19.16b, v31.16b\n"
                       "ldr q19, [x21, #0x20]\n"
                       "fmla v4.4s, v9.4s, v0.4s\n"
                       "movi v9.4s, #0x0\n"
                       "movi v0.4s, #0x0\n"
                       ".inst 0x4e88a629  // smmla v9.4s, v17.16b, v8.16b\n"
                       "fmul v8.4s, v27.4s, v26.s[0]\n"
                       ".inst 0x4e9fa620  // smmla v0.4s, v17.16b, v31.16b\n"
                       "ldr q17, [x21, #0x30]\n"
                       ".inst 0x4e8fa674  // smmla v20.4s, v19.16b, v15.16b\n"
                       "fmul v31.4s, v27.4s, v26.s[1]\n"
                       ".inst 0x4e81a672  // smmla v18.4s, v19.16b, v1.16b\n"
                       "ldr q19, [x21, #0x40]\n"
                       ".inst 0x4e8fa629  // smmla v9.4s, v17.16b, v15.16b\n"
                       "fmul v15.4s, v27.4s, v26.s[2]\n"
                       "fmul v27.4s, v27.4s, v26.s[3]\n"
                       ".inst 0x4e81a620  // smmla v0.4s, v17.16b, v1.16b\n"
                       "ldr q1, [x21, #0x50]\n"
                       ".inst 0x4e95a674  // smmla v20.4s, v19.16b, v21.16b\n"
                       ".inst 0x4e90a672  // smmla v18.4s, v19.16b, v16.16b\n"
                       "ldr q26, [x21, #0x60]\n"
                       ".inst 0x4e95a429  // smmla v9.4s, v1.16b, v21.16b\n"
                       ".inst 0x4e90a420  // smmla v0.4s, v1.16b, v16.16b\n"
                       "ldr q21, [x21, #0x70]\n"
                       "add x21, x21, #0x88\n"
                       ".inst 0x4e9da754  // smmla v20.4s, v26.16b, v29.16b\n"
                       ".inst 0x4e83a752  // smmla v18.4s, v26.16b, v3.16b\n"
                       ".inst 0x4e9da6a9  // smmla v9.4s, v21.16b, v29.16b\n"
                       ".inst 0x4e83a6a0  // smmla v0.4s, v21.16b, v3.16b\n"
                       "uzp1 v29.2d, v20.2d, v18.2d\n"
                       "uzp2 v21.2d, v20.2d, v18.2d\n"
                       "scvtf v29.4s, v29.4s, #0x4\n"
                       "uzp1 v18.2d, v9.2d, v0.2d\n"
                       "uzp2 v16.2d, v9.2d, v0.2d\n"
                       "scvtf v21.4s, v21.4s, #0x4\n"
                       "fmla v6.4s, v29.4s, v8.4s\n"
                       "scvtf v18.4s, v18.4s, #0x4\n"
                       "scvtf v16.4s, v16.4s, #0x4\n"
                       "fmla v30.4s, v21.4s, v31.4s\n"
                       "fmla v24.4s, v18.4s, v15.4s\n"
                       "fmla v14.4s, v16.4s, v27.4s\n"
                       "bgt 3b\n"
                       "mov x20, %x[res_ptr]\n"
                       "subs x27, x27, #0x4\n"
                       "add %x[res_ptr], %x[res_ptr], #0x10\n"
                       "str q2, [x20, #0x0]\n"
                       "add x20, x20, %x[res_stride]\n"
                       "str q10, [x20, #0x0]\n"
                       "add x20, x20, %x[res_stride]\n"
                       "str q12, [x20, #0x0]\n"
                       "add x20, x20, %x[res_stride]\n"
                       "str q28, [x20, #0x0]\n"
                       "add x20, x20, %x[res_stride]\n"
                       "str q11, [x20, #0x0]\n"
                       "add x20, x20, %x[res_stride]\n"
                       "str q13, [x20, #0x0]\n"
                       "add x20, x20, %x[res_stride]\n"
                       "str q22, [x20, #0x0]\n"
                       "add x20, x20, %x[res_stride]\n"
                       "str q23, [x20, #0x0]\n"
                       "add x20, x20, %x[res_stride]\n"
                       "str q25, [x20, #0x0]\n"
                       "add x20, x20, %x[res_stride]\n"
                       "str q5, [x20, #0x0]\n"
                       "add x20, x20, %x[res_stride]\n"
                       "str q7, [x20, #0x0]\n"
                       "add x20, x20, %x[res_stride]\n"
                       "str q4, [x20, #0x0]\n"
                       "add x20, x20, %x[res_stride]\n"
                       "str q6, [x20, #0x0]\n"
                       "add x20, x20, %x[res_stride]\n"
                       "str q30, [x20, #0x0]\n"
                       "add x20, x20, %x[res_stride]\n"
                       "str q24, [x20, #0x0]\n"
                       "add x20, x20, %x[res_stride]\n"
                       "str q14, [x20, #0x0]\n"
                       "bne 2b\n"
                       "mov x20, #0x4\n"
                       "sub x10, x10, #0x10\n"
                       "cmp x10, #0x10\n"
                       "mov %x[res_ptr], x26\n"
                       "madd %x[a_ptr], x20, x9, %x[a_ptr]\n"
                       "bge 1b\n"
                       "4:" // Row loop skip
                       "cbz x10, 9f\n"
                       "5:" // Row tail: Row loop
                       "add x24, %x[b_ptr], #0x8\n"
                       "mov x23, %x[nc]\n"
                       "add x22, %x[res_ptr], %x[res_stride], LSL #2\n"
                       "6:" // Row tail: Column loop
                       "movi v2.16b, #0x0\n"
                       "movi v10.16b, #0x0\n"
                       "add x25, %x[a_ptr], #0x8\n"
                       "mov x21, %x[nb]\n"
                       "movi v12.16b, #0x0\n"
                       "movi v28.16b, #0x0\n"
                       "7:" // Row tail: Block loop
                       "ldr q6, [x24, #0x0]\n"
                       "ldr q5, [x24, #0x10]\n"
                       "movi v17.16b, #0x4\n"
                       "movi v8.4s, #0x0\n"
                       "ldr q4, [x25, #0x0]\n"
                       "ldr q13, [x25, #0x10]\n"
                       "movi v27.4s, #0x0\n"
                       "movi v0.4s, #0x0\n"
                       "ldr q31, [x24, #0x20]\n"
                       "ldr q14, [x24, #0x30]\n"
                       "movi v29.4s, #0x0\n"
                       "movi v22.16b, #0xf0\n"
                       "ldr q11, [x25, #0x20]\n"
                       "ldr q23, [x25, #0x30]\n"
                       "sshl v21.16b, v6.16b, v17.16b\n"
                       "sshl v16.16b, v5.16b, v17.16b\n"
                       "ldr q20, [x25, #0x40]\n"
                       "ldr q26, [x25, #0x50]\n"
                       "and v6.16b, v6.16b, v22.16b\n"
                       "and v5.16b, v5.16b, v22.16b\n"
                       "ldr q25, [x25, #0x60]\n"
                       "ldr q3, [x25, #0x70]\n"
                       "sshl v19.16b, v31.16b, v17.16b\n"
                       "sshl v18.16b, v14.16b, v17.16b\n"
                       "ldr d17, [x25, #-0x8]\n"
                       ".inst 0x4e95a488  // smmla v8.4s, v4.16b, v21.16b\n"
                       ".inst 0x4e90a49b  // smmla v27.4s, v4.16b, v16.16b\n"
                       "and v31.16b, v31.16b, v22.16b\n"
                       ".inst 0x4e95a5a0  // smmla v0.4s, v13.16b, v21.16b\n"
                       ".inst 0x4e90a5bd  // smmla v29.4s, v13.16b, v16.16b\n"
                       "and v14.16b, v14.16b, v22.16b\n"
                       "sub x20, x24, #0x8\n"
                       "ldr d16, [x20, #0x0]\n"
                       "subs x21, x21, #0x1\n"
                       "add x25, x25, #0x88\n"
                       "fcvtl v17.4s, v17.4h\n"
                       "add x24, x24, #0x48\n"
                       ".inst 0x4e93a568  // smmla v8.4s, v11.16b, v19.16b\n"
                       ".inst 0x4e92a57b  // smmla v27.4s, v11.16b, v18.16b\n"
                       ".inst 0x4e93a6e0  // smmla v0.4s, v23.16b, v19.16b\n"
                       ".inst 0x4e92a6fd  // smmla v29.4s, v23.16b, v18.16b\n"
                       "fcvtl v16.4s, v16.4h\n"
                       ".inst 0x4e86a688  // smmla v8.4s, v20.16b, v6.16b\n"
                       ".inst 0x4e85a69b  // smmla v27.4s, v20.16b, v5.16b\n"
                       "fmul v23.4s, v16.4s, v17.s[0]\n"
                       "fmul v21.4s, v16.4s, v17.s[1]\n"
                       "fmul v1.4s, v16.4s, v17.s[2]\n"
                       "fmul v20.4s, v16.4s, v17.s[3]\n"
                       ".inst 0x4e86a740  // smmla v0.4s, v26.16b, v6.16b\n"
                       ".inst 0x4e85a75d  // smmla v29.4s, v26.16b, v5.16b\n"
                       ".inst 0x4e9fa728  // smmla v8.4s, v25.16b, v31.16b\n"
                       ".inst 0x4e8ea73b  // smmla v27.4s, v25.16b, v14.16b\n"
                       ".inst 0x4e9fa460  // smmla v0.4s, v3.16b, v31.16b\n"
                       ".inst 0x4e8ea47d  // smmla v29.4s, v3.16b, v14.16b\n"
                       "uzp1 v19.2d, v8.2d, v27.2d\n"
                       "uzp2 v18.2d, v8.2d, v27.2d\n"
                       "scvtf v19.4s, v19.4s, #0x4\n"
                       "uzp1 v17.2d, v0.2d, v29.2d\n"
                       "uzp2 v16.2d, v0.2d, v29.2d\n"
                       "scvtf v18.4s, v18.4s, #0x4\n"
                       "fmla v2.4s, v19.4s, v23.4s\n"
                       "scvtf v17.4s, v17.4s, #0x4\n"
                       "scvtf v16.4s, v16.4s, #0x4\n"
                       "fmla v10.4s, v18.4s, v21.4s\n"
                       "fmla v12.4s, v17.4s, v1.4s\n"
                       "fmla v28.4s, v16.4s, v20.4s\n"
                       "bgt 7b\n"
                       "mov x20, %x[res_ptr]\n"
                       "cmp x10, #0x1\n"
                       "str q2, [x20, #0x0]\n"
                       "add x20, x20, %x[res_stride]\n"
                       "ble 8f\n"
                       "cmp x10, #0x2\n"
                       "str q10, [x20, #0x0]\n"
                       "add x20, x20, %x[res_stride]\n"
                       "ble 8f\n"
                       "cmp x10, #0x3\n"
                       "str q12, [x20, #0x0]\n"
                       "add x20, x20, %x[res_stride]\n"
                       "ble 8f\n"
                       "str q28, [x20, #0x0]\n"
                       "8:" // Row tail: Accumulator store skip
                       "subs x23, x23, #0x4\n"
                       "add %x[res_ptr], %x[res_ptr], #0x10\n"
                       "bne 6b\n"
                       "subs x10, x10, #0x4\n"
                       "add %x[a_ptr], %x[a_ptr], x9\n"
                       "mov %x[res_ptr], x22\n"
                       "bgt 5b\n"
                       "9:" // Row tail: Row loop skip
                       : [a_ptr] "+&r"(a_ptr), [res_ptr] "+&r"(res_ptr)
                       : [b_ptr] "r"(b_ptr), [nr] "r"(nr), [nb] "r"(nb),
                         [res_stride] "r"(res_stride), [nc] "r"(nc)
                       : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5",
                         "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13",
                         "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21",
                         "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29",
                         "v30", "v31", "x9", "x10", "x20", "x21", "x22", "x23",
                         "x24", "x25", "x26", "x27", "x28");
  return;
#else
  float sumf[4][4];
  int sumi;

  for (int y = 0; y < nr / 4; y++) {
    const block_q8_0x4 *a_ptr = (const block_q8_0x4 *)vy + (y * nb);
    for (int x = 0; x < nc / ncols_interleaved; x++) {
      const block_q4_0x4 *b_ptr = (const block_q4_0x4 *)vx + (x * nb);
      for (int m = 0; m < 4; m++) {
        for (int j = 0; j < ncols_interleaved; j++)
          sumf[m][j] = 0.0;
      }
      for (int l = 0; l < nb; l++) {
        for (int k = 0; k < (qk / (2 * blocklen)); k++) {
          for (int m = 0; m < 4; m++) {
            for (int j = 0; j < ncols_interleaved; j++) {
              sumi = 0;
              for (int i = 0; i < blocklen; ++i) {
                const int v0 =
                  (int8_t)(b_ptr[l].qs[k * ncols_interleaved * blocklen +
                                       j * blocklen + i]
                           << 4);
                const int v1 =
                  (int8_t)(b_ptr[l].qs[k * ncols_interleaved * blocklen +
                                       j * blocklen + i] &
                           0xF0);
                sumi +=
                  ((v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i]) +
                   (v1 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i +
                                     qk / 2 * 4])) >>
                  4;
              }
              sumf[m][j] += sumi * nntr_compute_fp16_to_fp32(b_ptr[l].d[j]) *
                            nntr_compute_fp16_to_fp32(a_ptr[l].d[m]);
            }
          }
        }
      }
      for (int m = 0; m < 4; m++) {
        for (int j = 0; j < ncols_interleaved; j++)
          s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
      }
    }
  }
#endif
}

void nntr_quantize_mat_q8_0_4x8(const float *__restrict x, void *__restrict vy,
                                int64_t k) {
  assert(Q8_0 == 32);
  assert(k % Q8_0 == 0);
  const int nb = k / Q8_0;

  block_q8_0x4 *__restrict y = (block_q8_0x4 *)vy;

#if defined(__ARM_NEON)
  float32x4_t srcv[4][8];
  float id[4];

  for (int i = 0; i < nb; i++) {
    float32x4_t asrcv[8];
    float32x4_t amaxv[8];

    for (int row_iter = 0; row_iter < 4; row_iter++) {
      for (int j = 0; j < 8; j++)
        srcv[row_iter][j] = vld1q_f32(x + row_iter * k + i * 32 + 4 * j);
      for (int j = 0; j < 8; j++)
        asrcv[j] = vabsq_f32(srcv[row_iter][j]);

      for (int j = 0; j < 4; j++)
        amaxv[2 * j] = vmaxq_f32(asrcv[2 * j], asrcv[2 * j + 1]);
      for (int j = 0; j < 2; j++)
        amaxv[4 * j] = vmaxq_f32(amaxv[4 * j], amaxv[4 * j + 2]);
      for (int j = 0; j < 1; j++)
        amaxv[8 * j] = vmaxq_f32(amaxv[8 * j], amaxv[8 * j + 4]);

      const float amax = vmaxvq_f32(amaxv[0]);

      const float d = amax / ((1 << 7) - 1);
      id[row_iter] = d ? 1.0f / d : 0.0f;

      y[i].d[row_iter] = nntr_compute_fp32_to_fp16(d);
    }

    for (int j = 0; j < 4; j++) {
      float32x4_t v = vmulq_n_f32(srcv[0][2 * j], id[0]);
      int32x4_t vi = vcvtnq_s32_f32(v);
      y[i].qs[32 * j + 0] = vgetq_lane_s32(vi, 0);
      y[i].qs[32 * j + 1] = vgetq_lane_s32(vi, 1);
      y[i].qs[32 * j + 2] = vgetq_lane_s32(vi, 2);
      y[i].qs[32 * j + 3] = vgetq_lane_s32(vi, 3);
      v = vmulq_n_f32(srcv[0][2 * j + 1], id[0]);
      vi = vcvtnq_s32_f32(v);
      y[i].qs[32 * j + 4] = vgetq_lane_s32(vi, 0);
      y[i].qs[32 * j + 5] = vgetq_lane_s32(vi, 1);
      y[i].qs[32 * j + 6] = vgetq_lane_s32(vi, 2);
      y[i].qs[32 * j + 7] = vgetq_lane_s32(vi, 3);

      v = vmulq_n_f32(srcv[1][2 * j], id[1]);
      vi = vcvtnq_s32_f32(v);
      y[i].qs[32 * j + 8] = vgetq_lane_s32(vi, 0);
      y[i].qs[32 * j + 9] = vgetq_lane_s32(vi, 1);
      y[i].qs[32 * j + 10] = vgetq_lane_s32(vi, 2);
      y[i].qs[32 * j + 11] = vgetq_lane_s32(vi, 3);
      v = vmulq_n_f32(srcv[1][2 * j + 1], id[1]);
      vi = vcvtnq_s32_f32(v);
      y[i].qs[32 * j + 12] = vgetq_lane_s32(vi, 0);
      y[i].qs[32 * j + 13] = vgetq_lane_s32(vi, 1);
      y[i].qs[32 * j + 14] = vgetq_lane_s32(vi, 2);
      y[i].qs[32 * j + 15] = vgetq_lane_s32(vi, 3);

      v = vmulq_n_f32(srcv[2][2 * j], id[2]);
      vi = vcvtnq_s32_f32(v);
      y[i].qs[32 * j + 16] = vgetq_lane_s32(vi, 0);
      y[i].qs[32 * j + 17] = vgetq_lane_s32(vi, 1);
      y[i].qs[32 * j + 18] = vgetq_lane_s32(vi, 2);
      y[i].qs[32 * j + 19] = vgetq_lane_s32(vi, 3);
      v = vmulq_n_f32(srcv[2][2 * j + 1], id[2]);
      vi = vcvtnq_s32_f32(v);
      y[i].qs[32 * j + 20] = vgetq_lane_s32(vi, 0);
      y[i].qs[32 * j + 21] = vgetq_lane_s32(vi, 1);
      y[i].qs[32 * j + 22] = vgetq_lane_s32(vi, 2);
      y[i].qs[32 * j + 23] = vgetq_lane_s32(vi, 3);

      v = vmulq_n_f32(srcv[3][2 * j], id[3]);
      vi = vcvtnq_s32_f32(v);
      y[i].qs[32 * j + 24] = vgetq_lane_s32(vi, 0);
      y[i].qs[32 * j + 25] = vgetq_lane_s32(vi, 1);
      y[i].qs[32 * j + 26] = vgetq_lane_s32(vi, 2);
      y[i].qs[32 * j + 27] = vgetq_lane_s32(vi, 3);
      v = vmulq_n_f32(srcv[3][2 * j + 1], id[3]);
      vi = vcvtnq_s32_f32(v);
      y[i].qs[32 * j + 28] = vgetq_lane_s32(vi, 0);
      y[i].qs[32 * j + 29] = vgetq_lane_s32(vi, 1);
      y[i].qs[32 * j + 30] = vgetq_lane_s32(vi, 2);
      y[i].qs[32 * j + 31] = vgetq_lane_s32(vi, 3);
    }
  }
#elif defined(__AVX2__) || defined(__AVX__)
  float id[4];
  __m256 srcv[4][4];
  __m256 idvec[4];

  for (int i = 0; i < nb; i++) {
    for (int row_iter = 0; row_iter < 4; row_iter++) {
      // Load elements into 4 AVX vectors
      __m256 v0 = _mm256_loadu_ps(x + row_iter * k + i * 32);
      __m256 v1 = _mm256_loadu_ps(x + row_iter * k + i * 32 + 8);
      __m256 v2 = _mm256_loadu_ps(x + row_iter * k + i * 32 + 16);
      __m256 v3 = _mm256_loadu_ps(x + row_iter * k + i * 32 + 24);

      // Compute max(abs(e)) for the block
      const __m256 signBit = _mm256_set1_ps(-0.0f);
      __m256 maxAbs = _mm256_andnot_ps(signBit, v0);
      maxAbs = _mm256_max_ps(maxAbs, _mm256_andnot_ps(signBit, v1));
      maxAbs = _mm256_max_ps(maxAbs, _mm256_andnot_ps(signBit, v2));
      maxAbs = _mm256_max_ps(maxAbs, _mm256_andnot_ps(signBit, v3));

      __m128 max4 = _mm_max_ps(_mm256_extractf128_ps(maxAbs, 1),
                               _mm256_castps256_ps128(maxAbs));
      max4 = _mm_max_ps(max4, _mm_movehl_ps(max4, max4));
      max4 = _mm_max_ss(max4, _mm_movehdup_ps(max4));
      const float maxScalar = _mm_cvtss_f32(max4);

      // Divided by 127.f to mirror results in quantize_row_q8_0
      const float d = maxScalar / 127.f;
      id[row_iter] =
        (maxScalar != 0.0f) ? 127.f / maxScalar : 0.0f; // d ? 1.0f / d : 0.0f;

      // Store the scale for the individual block
      y[i].d[row_iter] = nntr_compute_fp32_to_fp16(d);

      // Store the values in blocks of eight values - Aim is to use these later
      // for block interleaving
      srcv[row_iter][0] = v0;
      srcv[row_iter][1] = v1;
      srcv[row_iter][2] = v2;
      srcv[row_iter][3] = v3;
      idvec[row_iter] = _mm256_set1_ps(id[row_iter]);
    }

    // The loop iterates four times - The aim is to get 4 corresponding chunks
    // of eight bytes from the original weight blocks that are interleaved
    for (int j = 0; j < 4; j++) {
      // Apply the multiplier
      __m256 v0 = _mm256_mul_ps(srcv[0][j], idvec[0]);
      __m256 v1 = _mm256_mul_ps(srcv[1][j], idvec[1]);
      __m256 v2 = _mm256_mul_ps(srcv[2][j], idvec[2]);
      __m256 v3 = _mm256_mul_ps(srcv[3][j], idvec[3]);

      // Round to nearest integer
      v0 = _mm256_round_ps(v0, _MM_ROUND_NEAREST);
      v1 = _mm256_round_ps(v1, _MM_ROUND_NEAREST);
      v2 = _mm256_round_ps(v2, _MM_ROUND_NEAREST);
      v3 = _mm256_round_ps(v3, _MM_ROUND_NEAREST);

      // Convert floats to integers
      __m256i i0 = _mm256_cvtps_epi32(v0);
      __m256i i1 = _mm256_cvtps_epi32(v1);
      __m256i i2 = _mm256_cvtps_epi32(v2);
      __m256i i3 = _mm256_cvtps_epi32(v3);

#if defined(__AVX2__)
      // Convert int32 to int16
      i0 = _mm256_packs_epi32(i0, i1);
      i2 = _mm256_packs_epi32(i2, i3);
      // Convert int16 to int8
      i0 = _mm256_packs_epi16(i0, i2);

      //  Permute and store the quantized weights in the required order after
      //  the pack instruction
      const __m256i perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
      i0 = _mm256_permutevar8x32_epi32(i0, perm);

      _mm256_storeu_si256((__m256i *)(y[i].qs + 32 * j), i0);
#else
      // Since we don't have in AVX some necessary functions,
      // we split the registers in half and call AVX2 analogs from SSE
      __m128i ni0 = _mm256_castsi256_si128(i0);
      __m128i ni1 = _mm256_extractf128_si256(i0, 1);
      __m128i ni2 = _mm256_castsi256_si128(i1);
      __m128i ni3 = _mm256_extractf128_si256(i1, 1);
      __m128i ni4 = _mm256_castsi256_si128(i2);
      __m128i ni5 = _mm256_extractf128_si256(i2, 1);
      __m128i ni6 = _mm256_castsi256_si128(i3);
      __m128i ni7 = _mm256_extractf128_si256(i3, 1);

      // Convert int32 to int16
      ni0 = _mm_packs_epi32(ni0, ni1);
      ni2 = _mm_packs_epi32(ni2, ni3);
      ni4 = _mm_packs_epi32(ni4, ni5);
      ni6 = _mm_packs_epi32(ni6, ni7);
      // Convert int16 to int8
      ni0 = _mm_packs_epi16(ni0, ni2);
      ni4 = _mm_packs_epi16(ni4, ni6);
      _mm_storeu_si128((__m128i *)(y[i].qs + 32 * j), ni0);
      _mm_storeu_si128((__m128i *)(y[i].qs + 32 * j + 16), ni4);
#endif
    }
  }
#else
  // scalar
  const int blck_size_interleave = 8;
  float srcv[4][Q8_0];
  float id[4];

  for (int i = 0; i < nb; i++) {
    for (int row_iter = 0; row_iter < 4; row_iter++) {
      float amax = 0.0f; // absolute max

      for (int j = 0; j < Q8_0; j++) {
        srcv[row_iter][j] = x[row_iter * k + i * Q8_0 + j];
        amax = MAX(amax, fabsf(srcv[row_iter][j]));
      }

      const float d = amax / ((1 << 7) - 1);
      id[row_iter] = d ? 1.0f / d : 0.0f;

      y[i].d[row_iter] = nntr_compute_fp32_to_fp16(d);
    }

    for (int j = 0; j < Q8_0 * 4; j++) {
      int src_offset = (j / (4 * blck_size_interleave)) * blck_size_interleave;
      int src_id = (j % (4 * blck_size_interleave)) / blck_size_interleave;
      src_offset += (j % blck_size_interleave);

      float x0 = srcv[src_id][src_offset] * id[src_id];
      y[i].qs[j] = roundf(x0);
    }
  }
#endif
}
