// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file   hgemm_transAB.cpp
 * @date   10 July 2024
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is half-precision GEMM interface of transposed  AB case
 *
 */

#include <hgemm_noTrans.h>
#include <hgemm_transAB.h>
#include <hgemm_util.h>
#include <matrix_transpose_neon.h>

void hgemm_transAB(const __fp16 *A, const __fp16 *B, float *C, unsigned int M,
                   unsigned int N, unsigned int K, float alpha, float beta) {
  __fp16 *A_T = alignedMalloc(M * K);
  __fp16 *B_T = alignedMalloc(K * N);

  transpose_neon<__fp16>(K, M, A, M, A_T, K);
  transpose_neon<__fp16>(N, K, B, K, B_T, N);

  hgemm_noTrans(A_T, B_T, C, M, N, K, alpha, beta);

  free(A_T);
  free(B_T);
}

void hgemm_K1_transAB(unsigned int M, unsigned int N, unsigned int K,
                      const __fp16 *A, unsigned int lda, const __fp16 *B,
                      unsigned int ldb, __fp16 *C, unsigned int ldc,
                      float alpha, float beta) {
  __fp16 *A_T = alignedMalloc(M * K);
  __fp16 *B_T = alignedMalloc(K * N);

  transpose_neon<__fp16>(K, M, A, M, A_T, K);
  transpose_neon<__fp16>(N, K, B, K, B_T, N);

  hgemm_K1_noTrans(M, N, K, A_T, lda, B_T, ldb, C, ldc, alpha, beta);

  free(A_T);
  free(B_T);
}
