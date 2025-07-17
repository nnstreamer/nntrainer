// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file   hgemm_transA.h
 * @date   10 July 2024
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is half-precision GEMM interface of transposed A case
 *
 */

/**
 * @brief     hgemm computation with neon : Y = alpha*A_T*B + beta*C,
 * @param[in] A __fp16 * for Matrix A
 * @param[in] B __fp16 * for Matrix B
 * @param[in] C float * for Matrix C
 * @param[in] M number of op(A)'s and C's row
 * @param[in] N number of op(B)'s and C's columns
 * @param[in] K number of op(A)'s and columns and op(B)'s rows
 * @param[in] alpha float number
 * @param[in] beta float number
 */
void hgemm_transA(const __fp16 *A, const __fp16 *B, float *C, unsigned int M,
                  unsigned int N, unsigned int K, float alpha, float beta);
