// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file   hgemm_transB.h
 * @date   10 July 2024
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is half-precision GEMM interface of transposed B case
 *
 */

/**
 * @brief     hgemm computation with neon : Y = alpha*op(A)*op(B) + beta*C,
 * where op(X) is one of X or X**T
 * @param[in] A __fp16 * for Matrix A
 * @param[in] B __fp16 * for Matrix B
 * @param[in] C __fp16 * for Matrix C
 * @param[in] M number of op(A)'s and C's row
 * @param[in] N number of op(B)'s and C's columns
 * @param[in] K number of op(A)'s and columns and op(B)'s rows
 * @param[in] alpha float number
 * @param[in] beta float number
 */
void hgemm_transB(const __fp16 *A, const __fp16 *B, float *C, unsigned int M,
                  unsigned int N, unsigned int K, float alpha, float beta);

void hgemm_transB_fallback(const __fp16 *A, const __fp16 *B, float *C,
                           unsigned int M, unsigned int N, unsigned int K,
                           float alpha, float beta);

/**
 * @brief     hgemm computation with neon : Y = alpha*op(A)*op(B) + beta*C,
 * where op(X) is one of X or X**T
 * @param[in] A __fp16 * for Matrix A
 * @param[in] B __fp16 * for Matrix B
 * @param[in] C __fp16 * for Matrix C
 * @param[in] M number of op(A)'s and C's row
 * @param[in] N number of op(B)'s and C's columns
 * @param[in] K number of op(A)'s and columns and op(B)'s rows
 * @param[in] alpha float number
 * @param[in] beta float number
 */
void hgemm_transB_8x16(unsigned int M, unsigned int N, unsigned int K,
                       const __fp16 *A, unsigned int lda, const __fp16 *B,
                       unsigned int ldb, float *C, unsigned int ldc,
                       float alpha = 1.F, float beta = 0.F);
