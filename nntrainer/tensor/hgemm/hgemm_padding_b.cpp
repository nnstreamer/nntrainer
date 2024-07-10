// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file   hgemm_padding_b.cpp
 * @date   05 July 2024
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is a source file for padding function used in hgemm
 *
 */

#include <arm_neon.h>
#include <hgemm_padding_b.h>
#include <hgemm_util.h>
#include <iostream>

void hgemm_padding_B(const __fp16 *B, __fp16 *Bp, unsigned int K,
                     unsigned int N, unsigned int K8, unsigned int N16,
                     bool transB) {
  if (transB) {
    hgemm_padding_B_Trans(B, Bp, K, N, K8, N16);
  } else {
    hgemm_padding_B_noTrans(B, Bp, K, N, K8, N16);
  }
}

void hgemm_padding_B_noTrans(const __fp16 *B, __fp16 *Bp, unsigned int K,
                             unsigned int N, unsigned int K8,
                             unsigned int N16) {
  if (K != K8 && N != N16) {
    hgemm_padding_B_noTrans_wrt_KN(B, Bp, K, N, K8, N16);
  } else if (K != K8) {
    hgemm_padding_B_noTrans_wrt_K(B, Bp, K, N, K8, N16);
  } else if (N != N16) {
    hgemm_padding_B_noTrans_wrt_N(B, Bp, K, N, K8, N16);
  } else {
    std::cerr << "Error : No room for matrix B padding\n";
  }
}

void hgemm_padding_B_Trans(const __fp16 *B, __fp16 *Bp, unsigned int K,
                           unsigned int N, unsigned int K8, unsigned int N16) {
  if (K != K8 && N != N16) {
    hgemm_padding_B_Trans_wrt_KN(B, Bp, K, N, K8, N16);
  } else if (K != K8) {
    hgemm_padding_B_Trans_wrt_K(B, Bp, K, N, K8, N16);
  } else if (N != N16) {
    hgemm_padding_B_Trans_wrt_N(B, Bp, K, N, K8, N16);
  } else {
    std::cerr << "Error : No room for matrix B padding\n";
  }
}

void hgemm_padding_B_noTrans_wrt_N(const __fp16 *B, __fp16 *Bp, unsigned int K,
                                   unsigned int N, unsigned int K8,
                                   unsigned int N16) {
  std::cerr << "Error : hgemm_padding_B_noTrans_wrt_N NYI!\n";
}

void hgemm_padding_B_noTrans_wrt_K(const __fp16 *B, __fp16 *Bp, unsigned int K,
                                   unsigned int N, unsigned int K8,
                                   unsigned int N16) {
  float16x8_t ZEROS = vmovq_n_f16(0.F);

  for (unsigned int k = 0; k < K; ++k) {
    for (unsigned int n = 0; n < N; n += 8) {
      vst1q_f16(&Bp[k * N + n], vld1q_f16(&B[k * N + n]));
    }
  }
  for (unsigned int k = K; k < K8; ++k) {
    for (unsigned int n = 0; n < N; n += 8) {
      vst1q_f16(&Bp[k * N + n], ZEROS);
    }
  }
}

void hgemm_padding_B_noTrans_wrt_KN(const __fp16 *B, __fp16 *Bp, unsigned int K,
                                    unsigned int N, unsigned int K8,
                                    unsigned int N16) {
  std::cerr << "Error : hgemm_padding_B_noTrans_wrt_KN NYI!\n";
}


void hgemm_padding_B_Trans_wrt_N(const __fp16 *B, __fp16 *Bp, unsigned int K,
                                 unsigned int N, unsigned int K8,
                                 unsigned int N16) {
  std::cerr << "Error : hgemm_padding_B_Trans_wrt_N NYI!\n";
}

void hgemm_padding_B_Trans_wrt_K(const __fp16 *B, __fp16 *Bp, unsigned int K,
                                 unsigned int N, unsigned int K8,
                                 unsigned int N16) {
  const unsigned int K8_low = (K >> 3) << 3;
  float16x8_t ZEROS = vmovq_n_f16(0.F);

  for (unsigned int n = 0; n < N; ++n) {
    for (unsigned int k = 0; k < K8_low; k += 8) {
      vst1q_f16(&Bp[n * K8 + k], vld1q_f16(&B[n * K + k]));
    }
    for (unsigned int k = K8_low; k < K; ++k) {
      Bp[n * K8 + k] = B[n * K + k];
    }
    for (unsigned int k = K; k < K8; ++k) {
      Bp[n * K8 + k] = 0.F;
    }
  }
}

void hgemm_padding_B_Trans_wrt_KN(const __fp16 *B, __fp16 *Bp, unsigned int K,
                                  unsigned int N, unsigned int K8,
                                  unsigned int N16) {
  std::cerr << "Error : hgemm_padding_B_Trans_wrt_KN NYI!\n";
}
