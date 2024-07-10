// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file   hgemm_pack.h
 * @date   01 April 2024
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @author Debadri Samaddar <s.debadri@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is for half-precision packing for kernel-based GEMM
 */

/**
 * @brief packing function of input matrix A
 *
 * @param M length of the row of the matrix
 * @param K length of the col of the matrix
 * @param src input of original source of the matrix
 * @param lda leading dimension of the matrix
 * @param dst output of packed data of the matrix
 */
void packing_A1(unsigned int m, unsigned int k, const __fp16 *from,
                unsigned int lda, const __fp16 *to);
/**
 * @brief packing function of input matrix A
 *
 * @param M length of the row of the matrix
 * @param K length of the col of the matrix
 * @param src input of original source of the matrix
 * @param lda leading dimension of the matrix
 * @param dst output of packed data of the matrix
 */
void packing_A4(unsigned int M, unsigned int K, const __fp16 *src,
                unsigned int lda, const __fp16 *dst);
/**
 * @brief packing function of input matrix A
 *
 * @param M length of the row of the matrix
 * @param K length of the col of the matrix
 * @param src input of original source of the matrix
 * @param lda leading dimension of the matrix
 * @param dst output of packed data of the matrix
 */
void packing_A8(unsigned int M, unsigned int K, const __fp16 *src,
                unsigned int lda, const __fp16 *dst);
/**
 * @brief packing function of input matrix B
 *
 * @param M length of the row of the matrix
 * @param K length of the col of the matrix
 * @param src input of original source of the matrix
 * @param ldb leading dimension of the matrix
 * @param dst output of packed data of the matrix
 */
void packing_B1(unsigned int K, unsigned int N, const __fp16 *src,
                unsigned int ldb, const __fp16 *dst);
/**
 * @brief packing function of input matrix B
 *
 * @param M length of the row of the matrix
 * @param K length of the col of the matrix
 * @param src input of original source of the matrix
 * @param ldb leading dimension of the matrix
 * @param dst output of packed data of the matrix
 */
void packing_B4(unsigned int K, unsigned int N, const __fp16 *src,
                unsigned int ldb, const __fp16 *dst);
/**
 * @brief packing function of input matrix B
 *
 * @param M length of the row of the matrix
 * @param K length of the col of the matrix
 * @param src input of original source of the matrix
 * @param ldb leading dimension of the matrix
 * @param dst output of packed data of the matrix
 */
void packing_B8(unsigned int K, unsigned int N, const __fp16 *src,
                unsigned int ldb, const __fp16 *dst);
/**
 * @brief packing function of input matrix B
 *
 * @param M length of the row of the matrix
 * @param K length of the col of the matrix
 * @param src input of original source of the matrix
 * @param ldb leading dimension of the matrix
 * @param dst output of packed data of the matrix
 */
void packing_B16(unsigned int K, unsigned int N, const __fp16 *src,
                 unsigned int ldb, const __fp16 *dst);
/**
 * @brief packing function of input matrix B_T
 *
 * @param M length of the row of the matrix
 * @param K length of the col of the matrix
 * @param src input of original source of the matrix
 * @param ldb leading dimension of the matrix
 * @param dst output of packed data of the matrix
 */
void packing_transB16(unsigned int K, unsigned int N, const __fp16 *src,
                      unsigned int ldb, const __fp16 *dst);
