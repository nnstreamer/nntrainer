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
#include <limits>

void hgemm(const __fp16 *A, const __fp16 *B, __fp16 *C, unsigned int M,
           unsigned int N, unsigned int K, float alpha, float beta, bool TransA,
           bool TransB) {
  if (K == 1) {
    return hgemm_K1(A, B, C, M, N, K, alpha, beta, TransA, TransB);
  } else if (M < 8 && K < 16 && N < 16) {
    return hgemm_small(A, B, C, M, N, K, alpha, beta, TransA, TransB);
  }

  const unsigned int M8_high = get_next_mltpl_of_n(M, 8);
  const unsigned int K8_high = get_next_mltpl_of_n(K, 8);
  const unsigned int N16_high = get_next_mltpl_of_n(N, 16);
  const unsigned int N8_low = get_prev_mltpl_of_2p_n(N, 3);

  float32x4_t ZEROS = vmovq_n_f32(0.F);

  float *C32 = (float *)malloc(M8_high * N16_high * sizeof(float));

  unsigned int size = M8_high * N16_high;
  unsigned int size8 = get_prev_mltpl_of_2p_n(size, 3);
  unsigned int size4 = get_prev_mltpl_of_2p_n(size, 2);

  if (std::fpclassify(beta) != FP_ZERO) {
    for (unsigned int m = 0; m < M; ++m) {
      for (unsigned int n = 0; n < N8_low; n += 8) {
        float16x8_t c =
          vmulq_n_f16(vld1q_f16(&C[m * N + n]), static_cast<__fp16>(beta));
        vst1q_f32(&C32[m * N16_high + n], vcvt_f32_f16(vget_low_f16(c)));
        vst1q_f32(&C32[m * N16_high + n + 4], vcvt_f32_f16(vget_high_f16(c)));
      }
      for (unsigned int n = N8_low; n < N; ++n) {
        C32[m * N16_high + n] = beta * C[m * N + n];
      }
      for (unsigned int n = N; n < N16_high; ++n) {
        C32[m * N16_high + n] = 0.F;
      }
    }
    for (unsigned m = M; m < M8_high; ++m) {
      for (unsigned int n = 0; n < N16_high; n += 4) {
        vst1q_f32(&C32[m * N16_high + n], ZEROS);
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

  hgemm_ensure_divisibility(A, B, C32, M, N, K, alpha, beta, TransA, TransB);

  for (unsigned int m = 0; m < M; ++m) {
    for (unsigned int n = 0; n < N8_low; n += 8) {
      float32x4_t x1 = vld1q_f32(&C32[m * N16_high + n]);
      float32x4_t x2 = vld1q_f32(&C32[m * N16_high + n + 4]);
      vst1q_f16(&C[m * N + n],
                vcombine_f16(vcvt_f16_f32(x1), vcvt_f16_f32(x2)));
    }
    for (unsigned int n = N8_low; n < N; ++n) {
      C[m * N + n] = C32[m * N16_high + n];
    }
  }

  free(C32);
}

void hgemm_small(const __fp16 *A, const __fp16 *B, __fp16 *C, unsigned int M,
                 unsigned int N, unsigned int K, float alpha, float beta,
                 bool TransA, bool TransB) {
  float *C32 = (float *)malloc(M * N * sizeof(float));

  copy_C_to_C32(C, C32, M, N, beta);

  hgemm_classify(A, B, C32, M, N, K, alpha, beta, TransA, TransB);

  copy_C32_to_C(C32, C, M, N, beta);

  free(C32);
}

void hgemm_ensure_divisibility(const __fp16 *A, const __fp16 *B, float *C32,
                               unsigned int M, unsigned int N, unsigned int K,
                               float alpha, float beta, bool TransA,
                               bool TransB) {
  /// @note Padding standard : 8x16 is the only KERNEL that outperforms single
  /// precision GEMM 'so far'. Padding will forcibly make every GEMM cases to
  /// use it. Note that padding is not an optimal way here, but just an option
  /// that is easier to implement. Fine-grained packing, blocking, and
  /// corresponding kernels should be supported in the future for optimal
  /// performance in terms of both latency and memory.

  __fp16 *A_ = (__fp16 *)A, *B_ = (__fp16 *)B;
  unsigned int M_ = M, N_ = N, K_ = K;
  bool pad_A = false, pad_B = false;

  __fp16 *Ap;
  __fp16 *Bp;

  const unsigned int M8_high = ((M - 1) / 8 + 1) * 8;
  const unsigned int K8_high = ((K - 1) / 16 + 1) * 16;
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
  unsigned int ldc = N;

  const float eps = std::numeric_limits<float>::epsilon();
  float16x8_t a_vec;
  unsigned int N8 = (N >> 3) << 3;
  for (unsigned int m = 0; m < M; ++m) {
    a_vec = vmovq_n_f16(alpha * A[m]);
    if (std::fpclassify(beta) != FP_ZERO) {
      for (unsigned int n = 0; n < N8; n += 8) {
        vst1q_f16(&C[m * ldc + n],
                  vaddq_f16(vmulq_f16(a_vec, vld1q_f16(&B[n])),
                            vmulq_n_f16(vld1q_f16(&C[m * ldc + n]), beta)));
      }
    } else {
      for (unsigned int n = 0; n < N8; n += 8) {
        vst1q_f16(&C[m * ldc + n], vmulq_f16(a_vec, vld1q_f16(&B[n])));
      }
    }
    for (unsigned int n = N8; n < N; ++n) {
      C[m * ldc + n] = alpha * A[m] * B[n] + beta * C[m * ldc + n];
    }
  }
}
