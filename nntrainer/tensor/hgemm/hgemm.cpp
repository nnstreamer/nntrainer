// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file   hgemm.cpp
 * @date   03 April 2024
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @author Debadri Samaddar <s.debadri@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is half-precision GEMM interface
 *
 */

#include <arm_neon.h>
#include <cmath>
#include <hgemm.h>
#include <hgemm_common.h>
#include <hgemm_noTrans.h>
#include <hgemm_padding.h>
#include <hgemm_transA.h>
#include <hgemm_transAB.h>
#include <hgemm_transB.h>
#include <hgemm_util.h>

void hgemm(const __fp16 *A, const __fp16 *B, __fp16 *C, unsigned int M,
           unsigned int N, unsigned int K, float alpha, float beta, bool TransA,
           bool TransB) {
  if (K == 1) {
    return hgemm_K1(A, B, C, M, N, K, alpha, beta, TransA, TransB);
  }
  // dynamic creation to avoid reaching stack limit(causes segmentation fault)
  float *C32 = (float *)malloc(M * N * sizeof(float));

  // performing beta*C
  unsigned int idx = 0;
  unsigned int size = M * N;
  unsigned int size8 = (size >> 3) << 3;
  unsigned int size4 = (size >> 2) << 2;

  if (std::fpclassify(beta) != FP_ZERO) {
    for (; idx < size8; idx += 8) {
      float16x8_t c =
        vmulq_n_f16(vld1q_f16(&C[idx]), static_cast<__fp16>(beta));

      vst1q_f32(&C32[idx], vcvt_f32_f16(vget_low_f16(c)));
      vst1q_f32(&C32[idx + 4], vcvt_f32_f16(vget_high_f16(c)));
    }
    // remaining 4
    for (; idx < size4; idx += 4) {
      float16x4_t c = vmul_n_f16(vld1_f16(&C[idx]), static_cast<__fp16>(beta));

      vst1q_f32(&C32[idx], vcvt_f32_f16(c));
    }

    // remaining values if dimensions not a multiple of 8
    for (; idx < size; idx++) {
      C32[idx] = C[idx] * beta;
    }
  } else {
    float32x4_t zeros = vmovq_n_f32(0.F);
    for (; idx < size4; idx += 4) {
      vst1q_f32(&C32[idx], zeros);
    }
    for (; idx < size; idx++) {
      C32[idx] = 0.F;
    }
  }

  hgemm_ensure_divisibility(A, B, C32, M, N, K, alpha, beta, TransA, TransB);

  unsigned int L = M * N;
  unsigned int L8 = (L >> 3) << 3;

  for (unsigned int idx = 0; idx < L8; idx += 8) {
    float32x4_t x1 = vld1q_f32(&C32[idx]);
    float32x4_t x2 = vld1q_f32(&C32[idx + 4]);

    float16x8_t y1 = vcombine_f16(vcvt_f16_f32(x1), vcvt_f16_f32(x2));

    vst1q_f16(&C[idx], y1);
  }
  for (unsigned int idx = L8; idx < L; ++idx) {
    C[idx] = static_cast<__fp16>(C32[idx]);
  }

  free(C32);
}

void hgemm_ensure_divisibility(const __fp16 *A, const __fp16 *B, float *C32,
                               unsigned int M, unsigned int N, unsigned int K,
                               float alpha, float beta, bool TransA,
                               bool TransB) {
  /// @note Padding standard : 8x16 is the only KERNEL that outperforms single
  /// precision GEMM 'so far'. Padding will forcibly make every GEMM cases to
  /// use it. Note that padding is not the optimal way here, but just an option
  /// that is easier to implement. Fine-grained packing should be supported on
  /// the future for optimal performance.

  __fp16 *A_ = (__fp16 *)A, *B_ = (__fp16 *)B;
  unsigned int M_ = M, N_ = N, K_ = K;
  bool pad_A = false, pad_B = false;

  // Case 2 : smaller than 8, 16 | padding would be redundant?
  if (M < 8 && K < 16 && N < 16)
    return hgemm_classify(A_, B_, C32, M_, N_, K_, alpha, beta, TransA, TransB);

  __fp16 *Ap;
  __fp16 *Bp;

  const unsigned int M8_high = ((M - 1) / 8 + 1) * 8;
  const unsigned int K8_high = ((K - 1) / 8 + 1) * 8;
  const unsigned int N16_high = ((N - 1) / 16 + 1) * 16;

  if ((M8_high != M) || (K8_high != K)) {
    pad_A = true;
    Ap = alignedMalloc(M8_high * K8_high);
    hgemm_padding_A(A, Ap, M, K, M8_high, K8_high, TransA);
    A_ = Ap;
    M_ = M8_high;
    K_ = K8_high;
  }
  if ((K8_high != K) || (N16_high != N)) {
    pad_B = true;
    Bp = alignedMalloc(K8_high * N16_high);
    hgemm_padding_B(B, Bp, K, N, K8_high, N16_high, TransB);
    B_ = Bp;
    K_ = K8_high;
    N_ = N16_high;
  }

  hgemm_classify(A_, B_, C32, M_, N_, K_, alpha, beta, TransA, TransB);

  if (pad_A)
    free(Ap);
  if (pad_B)
    free(Bp);
}

void hgemm_classify(const __fp16 *A, const __fp16 *B, float *C32,
                    unsigned int M, unsigned int N, unsigned int K, float alpha,
                    float beta, bool TransA, bool TransB) {
  if (!TransA && !TransB) {
    hgemm_noTrans(A, B, C32, M, N, K, alpha, beta);
  } else if (TransA && !TransB) {
    hgemm_transA(A, B, C32, M, N, K, alpha, beta);
  } else if (!TransA && TransB) {
    hgemm_transB(A, B, C32, M, N, K, alpha, beta);
  } else { // TransA && TransB
    hgemm_transAB(A, B, C32, M, N, K, alpha, beta);
  }
}

void hgemm_K1(const __fp16 *A, const __fp16 *B, __fp16 *C, unsigned int M,
              unsigned int N, unsigned int K, float alpha, float beta,
              bool TransA, bool TransB) {
  unsigned int lda = (TransA) ? M : K;
  unsigned int ldb = (TransB) ? K : N;

  return hgemm_K1_noTrans(M, N, K, A, lda, B, ldb, C, N, alpha, beta);

  if (!TransA && TransB) {
    hgemm_K1_transB(M, N, K, A, lda, B, ldb, C, N, alpha, beta);
  } else if (TransA && !TransB) {
    hgemm_K1_transA(M, N, K, A, lda, B, ldb, C, N, alpha, beta);
  } else if (!TransA && !TransB) {
    hgemm_K1_noTrans(M, N, K, A, lda, B, ldb, C, N, alpha, beta);
  } else { // TransA && TransB
    hgemm_K1_transAB(M, N, K, A, lda, B, ldb, C, N, alpha, beta);
  }
}
