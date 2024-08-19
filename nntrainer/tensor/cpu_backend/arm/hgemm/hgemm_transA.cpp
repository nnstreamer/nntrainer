// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file   hgemm_transA.cpp
 * @date   10 July 2024
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is half-precision GEMM interface of transposed A case
 *
 */

#include <hgemm_noTrans.h>
#include <hgemm_transA.h>
#include <hgemm_util.h>
#include <matrix_transpose_neon.h>

void hgemm_transA(const __fp16 *A, const __fp16 *B, float *C, unsigned int M,
                  unsigned int N, unsigned int K, float alpha, float beta) {
  __fp16 *A_T = alignedMalloc(M * K);

  transpose_neon<__fp16>(K, M, A, M, A_T, K);

  hgemm_noTrans(A_T, B, C, M, N, K, alpha, beta);

  free(A_T);
}
