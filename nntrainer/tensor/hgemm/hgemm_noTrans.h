// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file   hgemm_noTrans.h
 * @date   10 July 2024
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is half-precision GEMM interface of non-transposed case
 *
 */

/**
 * @brief hgemm noTrans computation with 1x4 kernel : C = A*B,
 *
 * @param M length of the row of matrix A
 * @param N length of the col of matrix B
 * @param K length of the col of matrix A
 * @param A input matrix A
 * @param lda length of the col of matrix A
 * @param B input matrix B
 * @param ldb length of the col of matrix B
 * @param C output matrix C
 * @param ldc length of the col of matrix C
 * @param[in] alpha float number
 * @param[in] beta float number
 */
void hgemm_noTrans_1x4(unsigned int M, unsigned int N, unsigned int K,
                       const __fp16 *A, unsigned int lda, const __fp16 *B,
                       unsigned int ldb, __fp16 *C, unsigned int ldc,
                       float alpha = 1.F, float beta = 0.F);

/**
 * @brief hgemm noTrans computation with 1x4 kernel : C = A*B,
 *
 * @param M length of the row of matrix A
 * @param N length of the col of matrix B
 * @param K length of the col of matrix A
 * @param A input matrix A
 * @param lda length of the col of matrix A
 * @param B input matrix B
 * @param ldb length of the col of matrix B
 * @param C output matrix C
 * @param ldc length of the col of matrix C
 * @param[in] alpha float number
 * @param[in] beta float number
 */
void hgemm_noTrans_1x4(unsigned int M, unsigned int N, unsigned int K,
                       const __fp16 *A, unsigned int lda, const __fp16 *B,
                       unsigned int ldb, float *C, unsigned int ldc,
                       float alpha = 1.F, float beta = 0.F);

/**
 * @brief hgemm noTrans computation with 4x4 kernel : C = A*B,
 *
 * @param M length of the row of matrix A
 * @param N length of the col of matrix B
 * @param K length of the col of matrix A
 * @param A input matrix A
 * @param lda length of the col of matrix A
 * @param B input matrix B
 * @param ldb length of the col of matrix B
 * @param C output matrix C
 * @param ldc length of the col of matrix C
 * @param[in] alpha float number
 * @param[in] beta float number
 */
void hgemm_noTrans_4x4(unsigned int M, unsigned int N, unsigned int K,
                       const __fp16 *A, unsigned int lda, const __fp16 *B,
                       unsigned int ldb, __fp16 *C, unsigned int ldc,
                       float alpha = 1.F, float beta = 0.F);

/**
 * @brief hgemm noTrans computation with 1x8 kernel : C = A*B,
 *
 * @param M length of the row of matrix A
 * @param N length of the col of matrix B
 * @param K length of the col of matrix A
 * @param A input matrix A
 * @param lda length of the col of matrix A
 * @param B input matrix B
 * @param ldb length of the col of matrix B
 * @param C output matrix C
 * @param ldc length of the col of matrix C
 * @param[in] alpha float number
 * @param[in] beta float number
 */
void hgemm_noTrans_1x8(unsigned int M, unsigned int N, unsigned int K,
                       const __fp16 *A, unsigned int lda, const __fp16 *B,
                       unsigned int ldb, __fp16 *C, unsigned int ldc,
                       float alpha = 1.F, float beta = 0.F);

/**
 * @brief hgemm noTrans computation with 1x8 kernel : C = A*B,
 *
 * @param M length of the row of matrix A
 * @param N length of the col of matrix B
 * @param K length of the col of matrix A
 * @param A input matrix A
 * @param lda length of the col of matrix A
 * @param B input matrix B
 * @param ldb length of the col of matrix B
 * @param C output matrix C
 * @param ldc length of the col of matrix C
 * @param[in] alpha float number
 * @param[in] beta float number
 */
void hgemm_noTrans_1x8(unsigned int M, unsigned int N, unsigned int K,
                       const __fp16 *A, unsigned int lda, const __fp16 *B,
                       unsigned int ldb, float *C, unsigned int ldc,
                       float alpha = 1.F, float beta = 0.F);

/**
 * @brief hgemm noTrans computation with 8x8 kernel : C = A*B,
 *
 * @param M length of the row of matrix A
 * @param N length of the col of matrix B
 * @param K length of the col of matrix A
 * @param A input matrix A
 * @param lda length of the col of matrix A
 * @param B input matrix B
 * @param ldb length of the col of matrix B
 * @param C output matrix C
 * @param ldc length of the col of matrix C
 * @param[in] alpha float number
 * @param[in] beta float number
 */
void hgemm_noTrans_8x8(unsigned int M, unsigned int N, unsigned int K,
                       const __fp16 *A, unsigned int lda, const __fp16 *B,
                       unsigned int ldb, __fp16 *C, unsigned int ldc,
                       float alpha = 1.F, float beta = 0.F);

/**
 * @brief hgemm noTrans computation with 4x4 kernel : C = A*B,
 *
 * @param M length of the row of matrix A
 * @param N length of the col of matrix B
 * @param K length of the col of matrix A
 * @param A input matrix A
 * @param lda length of the col of matrix A
 * @param B input matrix B
 * @param ldb length of the col of matrix B
 * @param C output matrix C
 * @param ldc length of the col of matrix C
 * @param[in] alpha float number
 * @param[in] beta float number
 */
void hgemm_noTrans_4x4(unsigned int M, unsigned int N, unsigned int K,
                       const __fp16 *A, unsigned int lda, const __fp16 *B,
                       unsigned int ldb, float *C, unsigned int ldc,
                       float alpha = 1.F, float beta = 0.F);

/**
 * @brief hgemm noTrans computation with 8x8 kernel : C = A*B,
 *
 * @param M length of the row of matrix A
 * @param N length of the col of matrix B
 * @param K length of the col of matrix A
 * @param A input matrix A
 * @param lda length of the col of matrix A
 * @param B input matrix B
 * @param ldb length of the col of matrix B
 * @param C output matrix C
 * @param ldc length of the col of matrix C
 * @param[in] alpha float number
 * @param[in] beta float number
 */
void hgemm_noTrans_8x8(unsigned int M, unsigned int N, unsigned int K,
                       const __fp16 *A, unsigned int lda, const __fp16 *B,
                       unsigned int ldb, float *C, unsigned int ldc,
                       float alpha = 1.F, float beta = 0.F);

/**
 * @brief hgemm noTrans computation with 4x8 kernel : C = A*B,
 *
 * @param M length of the row of matrix A
 * @param N length of the col of matrix B
 * @param K length of the col of matrix A
 * @param A input matrix A
 * @param lda length of the col of matrix A
 * @param B input matrix B
 * @param ldb length of the col of matrix B
 * @param C output matrix C
 * @param ldc length of the col of matrix C
 * @param[in] alpha float number
 * @param[in] beta float number
 */
void hgemm_noTrans_4x8(unsigned int M, unsigned int N, unsigned int K,
                       const __fp16 *A, unsigned int lda, const __fp16 *B,
                       unsigned int ldb, __fp16 *C, unsigned int ldc,
                       float alpha = 1.F, float beta = 0.F);

/**
 * @brief hgemm noTrans computation with 4x8 kernel : C = A*B,
 *
 * @param M length of the row of matrix A
 * @param N length of the col of matrix B
 * @param K length of the col of matrix A
 * @param A input matrix A
 * @param lda length of the col of matrix A
 * @param B input matrix B
 * @param ldb length of the col of matrix B
 * @param C output matrix C
 * @param ldc length of the col of matrix C
 * @param[in] alpha float number
 * @param[in] beta float number
 */
void hgemm_noTrans_4x8(unsigned int M, unsigned int N, unsigned int K,
                       const __fp16 *A, unsigned int lda, const __fp16 *B,
                       unsigned int ldb, float *C, unsigned int ldc,
                       float alpha = 1.F, float beta = 0.F);

/**
 * @brief hgemm noTrans computation with 8x16 kernel : C = A*B,
 *
 * @param M length of the row of matrix A
 * @param N length of the col of matrix B
 * @param K length of the col of matrix A
 * @param A input matrix A
 * @param lda length of the col of matrix A
 * @param B input matrix B
 * @param ldb length of the col of matrix B
 * @param C output matrix C
 * @param ldc length of the col of matrix C
 * @param[in] alpha float number
 * @param[in] beta float number
 */
void hgemm_noTrans_8x16(unsigned int M, unsigned int N, unsigned int K,
                        const __fp16 *A, unsigned int lda, const __fp16 *B,
                        unsigned int ldb, __fp16 *C, unsigned int ldc,
                        float alpha = 1.F, float beta = 0.F);

/**
 * @brief hgemm noTrans computation with 8x16 kernel : C = A*B,
 *
 * @param M length of the row of matrix A
 * @param N length of the col of matrix B
 * @param K length of the col of matrix A
 * @param A input matrix A
 * @param lda length of the col of matrix A
 * @param B input matrix B
 * @param ldb length of the col of matrix B
 * @param C output matrix C
 * @param ldc length of the col of matrix C
 * @param[in] alpha float number
 * @param[in] beta float number
 */
void hgemm_noTrans_8x16(unsigned int M, unsigned int N, unsigned int K,
                        const __fp16 *A, unsigned int lda, const __fp16 *B,
                        unsigned int ldb, float *C, unsigned int ldc,
                        float alpha = 1.F, float beta = 0.F);

/**
 * @brief hgemm noTrans computation with 8x16 kernel : C = A*B,
 *
 * @param M length of the row of matrix A
 * @param N length of the col of matrix B
 * @param K length of the col of matrix A
 * @param A input matrix A
 * @param lda length of the col of matrix A
 * @param B input matrix B
 * @param ldb length of the col of matrix B
 * @param C output matrix C
 * @param ldc length of the col of matrix C
 * @param[in] alpha float number
 * @param[in] beta float number
 */
void hgemm_noTrans_8x16_experimental(unsigned int M, unsigned int N,
                                     unsigned int K, const __fp16 *A,
                                     unsigned int lda, const __fp16 *B,
                                     unsigned int ldb, float *C,
                                     unsigned int ldc, float alpha = 1.F,
                                     float beta = 0.F);

/**
 * @brief     hgemm fallback with NEON : Y = alpha*op(A)*op(B) + beta*C,
 * @param M length of the row of matrix A
 * @param N length of the col of matrix B
 * @param K length of the col of matrix A
 * @param A input matrix A
 * @param lda length of the col of matrix A
 * @param B input matrix B
 * @param ldb length of the col of matrix B
 * @param C output matrix C
 * @param ldc length of the col of matrix C
 * @param[in] alpha float number
 * @param[in] beta float number
 */
void hgemm_noTrans_fallback(unsigned int M, unsigned int N, unsigned int K,
                            const __fp16 *A, unsigned int lda, const __fp16 *B,
                            unsigned int ldb, float *C, unsigned int ldc,
                            float alpha = 1.F, float beta = 0.F);

/**
 * @brief     hgemm computation with neon : Y = alpha*op(A)*op(B) + beta*C,
 * @param[in] A __fp16 * for Matrix A
 * @param[in] B __fp16 * for Matrix B
 * @param[in] C float * for Matrix C
 * @param[in] M number of op(A)'s and C's row
 * @param[in] N number of op(B)'s and C's columns
 * @param[in] K number of op(A)'s and columns and op(B)'s rows
 * @param[in] alpha float number
 * @param[in] beta float number
 */
void hgemm_noTrans(const __fp16 *A, const __fp16 *B, float *C, unsigned int M,
                   unsigned int N, unsigned int K, float alpha = 1.F,
                   float beta = 0.F);

/**
 * @brief     hgemm computation with neon : Y = alpha*op(A)*op(B) + beta*C,
 * where M, N, K are divisible by at least 4
 * @param[in] A __fp16 * for Matrix A
 * @param[in] B __fp16 * for Matrix B
 * @param[in] C __fp16 * for Matrix C
 * @param[in] M number of op(A)'s and C's row
 * @param[in] N number of op(B)'s and C's columns
 * @param[in] K number of op(A)'s and columns and op(B)'s rows
 * @param[in] alpha float number
 * @param[in] beta float number
 */
void hgemm_noTrans_strict(const __fp16 *A, const __fp16 *B, __fp16 *C,
                          unsigned int M, unsigned int N, unsigned int K,
                          float alpha = 1.F, float beta = 0.F);

/**
 * @brief     hgemm computation with neon : Y = alpha*op(A)*op(B) + beta*C,
 * where M, N, K are divisible by at least 4
 * @param[in] A __fp16 * for Matrix A
 * @param[in] B __fp16 * for Matrix B
 * @param[in] C float * for Matrix C
 * @param[in] M number of op(A)'s and C's row
 * @param[in] N number of op(B)'s and C's columns
 * @param[in] K number of op(A)'s and columns and op(B)'s rows
 * @param[in] alpha float number
 * @param[in] beta float number
 */
void hgemm_noTrans_strict(const __fp16 *A, const __fp16 *B, float *C,
                          unsigned int M, unsigned int N, unsigned int K,
                          float alpha = 1.F, float beta = 0.F);
