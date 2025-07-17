// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file   hgemm_util.cpp
 * @date   01 August 2024
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is for util functions for half-precision GEMM
 */

#include <arm_neon.h>
#include <cmath>
#include <hgemm_util.h>

/**
 * @brief aligned dynamic allocation function
 *
 * @param sz amount of data to allocate
 * @return __fp16* addr of allocated memory
 */
__fp16 *alignedMalloc(unsigned int sz) {
  void *addr = 0;
  int iRet = posix_memalign(&addr, 64, sz * sizeof(__fp16));
  assert(0 == iRet);
  return (__fp16 *)addr;
}

unsigned int get_next_mltpl_of_n(unsigned int x, unsigned int n) {
  assert(x > 0);
  return ((x - 1) / n + 1) * n;
}

unsigned int get_prev_mltpl_of_2p_n(unsigned int x, unsigned int n) {
  assert(x > 0);
  return (x >> n) << n;
}

void copy_C_to_C32(__fp16 *C, float *C32, unsigned int M, unsigned int N,
                   float beta) {
  float32x4_t ZEROS = vmovq_n_f32(0.F);
  unsigned int size = M * N;
  unsigned int size4 = (size >> 2) << 2;
  const unsigned int N8_low = get_prev_mltpl_of_2p_n(N, 3);

  if (std::fpclassify(beta) != FP_ZERO) {
    for (unsigned int m = 0; m < M; ++m) {
      for (unsigned int n = 0; n < N8_low; n += 8) {
        float16x8_t c = vmulq_n_f16(vld1q_f16(&C[m * N + n]), beta);
        vst1q_f32(&C32[m * N + n], vcvt_f32_f16(vget_low_f16(c)));
        vst1q_f32(&C32[m * N + n + 4], vcvt_f32_f16(vget_high_f16(c)));
      }
      for (unsigned int n = N8_low; n < N; ++n) {
        C32[m * N + n] = beta * C[m * N + n];
      }
    }
  } else {
    for (unsigned int idx = 0; idx < size4; idx += 4) {
      vst1q_f32(&C32[idx], ZEROS);
    }
    for (unsigned int idx = size4; idx < size; idx++) {
      C32[idx] = 0.F;
    }
  }
}

void copy_C32_to_C(float *C32, __fp16 *C, unsigned int M, unsigned int N,
                   float beta) {
  const unsigned int N8_low = get_prev_mltpl_of_2p_n(N, 3);
  for (unsigned int m = 0; m < M; ++m) {
    for (unsigned int n = 0; n < N8_low; n += 8) {
      float32x4_t x1 = vld1q_f32(&C32[m * N + n]);
      float32x4_t x2 = vld1q_f32(&C32[m * N + n + 4]);
      vst1q_f16(&C[m * N + n],
                vcombine_f16(vcvt_f16_f32(x1), vcvt_f16_f32(x2)));
    }
    for (unsigned int n = N8_low; n < N; ++n) {
      C[m * N + n] = C32[m * N + n];
    }
  }
}
