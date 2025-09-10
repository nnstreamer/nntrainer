// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file   hgemm.h
 * @date   01 April 2024
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @author Debadri Samaddar <s.debadri@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is half-precision GEMM interface
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
 * @param[in] TransA bool transpose info of lhs matrix
 * @param[in] TransB bool transpose info of rhs matrix
 */
void hgemm(const __fp16 *A, const __fp16 *B, __fp16 *C, unsigned int M,
           unsigned int N, unsigned int K, float alpha, float beta, bool TransA,
           bool TransB);

/**
 * @brief     hgemm computation with neon but with small dim without padding : Y
 * = alpha*op(A)*op(B) + beta*C, where op(X) is one of X or X**T
 * @param[in] A __fp16 * for Matrix A
 * @param[in] B __fp16 * for Matrix B
 * @param[in] C __fp16 * for Matrix C
 * @param[in] M number of op(A)'s and C's row
 * @param[in] N number of op(B)'s and C's columns
 * @param[in] K number of op(A)'s and columns and op(B)'s rows
 * @param[in] alpha float number
 * @param[in] beta float number
 * @param[in] TransA bool transpose info of lhs matrix
 * @param[in] TransB bool transpose info of rhs matrix
 */
void hgemm_small(const __fp16 *A, const __fp16 *B, __fp16 *C, unsigned int M,
                 unsigned int N, unsigned int K, float alpha, float beta,
                 bool TransA, bool TransB);

/**
 * @brief     Checking function for whether matrix A or B needs padding for
 * optimal performance of fixed blocking-kernel sequence
 * @param[in] A __fp16 * for Matrix A
 * @param[in] B __fp16 * for Matrix B
 * @param[in] C float * for Matrix C
 * @param[in] M number of op(A)'s and C's row
 * @param[in] N number of op(B)'s and C's columns
 * @param[in] K number of op(A)'s and columns and op(B)'s rows
 * @param[in] alpha float number
 * @param[in] beta float number
 * @param[in] TransA bool transpose info of lhs matrix
 * @param[in] TransB bool transpose info of rhs matrix
 */
void hgemm_ensure_divisibility(const __fp16 *A, const __fp16 *B, float *C32,
                               unsigned int M, unsigned int N, unsigned int K,
                               float alpha = 1.F, float beta = 0.F,
                               bool TransA = false, bool TransB = false);

/**
 * @brief     Classifying function for GEMM computation case for noTrans,
 * transA, transB, transAB
 * @param[in] A __fp16 * for Matrix A
 * @param[in] B __fp16 * for Matrix B
 * @param[in] C __fp16 * for Matrix C
 * @param[in] M number of op(A)'s and C's row
 * @param[in] N number of op(B)'s and C's columns
 * @param[in] K number of op(A)'s and columns and op(B)'s rows
 * @param[in] alpha float number
 * @param[in] beta float number
 * @param[in] TransA bool transpose info of lhs matrix
 * @param[in] TransB bool transpose info of rhs matrix
 */
void hgemm_classify(const __fp16 *A, const __fp16 *B, float *C32,
                    unsigned int M, unsigned int N, unsigned int K,
                    float alpha = 1.F, float beta = 0.F, bool TransA = false,
                    bool TransB = false);
/**
 * @brief     hgemm computation when K = 1. Transpose is mathematically no use
 * for here, and partial accumulation is also not needed.
 * @param[in] A __fp16 * for Matrix A
 * @param[in] B __fp16 * for Matrix B
 * @param[in] C __fp16 * for Matrix C
 * @param[in] M number of op(A)'s and C's row
 * @param[in] N number of op(B)'s and C's columns
 * @param[in] K number of op(A)'s and columns and op(B)'s rows
 * @param[in] alpha float number
 * @param[in] beta float number
 * @param[in] TransA bool transpose info of lhs matrix
 * @param[in] TransB bool transpose info of rhs matrix
 */
void hgemm_K1(const __fp16 *A, const __fp16 *B, __fp16 *C, unsigned int M,
              unsigned int N, unsigned int K, float alpha, float beta,
              bool TransA, bool TransB);
