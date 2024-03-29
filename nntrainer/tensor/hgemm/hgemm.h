// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file   hgemm.h
 * @date   01 April 2024
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is half-precision GEMM interface
 *
 */

#include <hgemm_kernel_4x4.h>
#include <hgemm_kernel_8x8.h>
#include <hgemm_kernel_pack.h>
#include <hgemm_util.h>

#define KERNEL_4x4 hgemm_kernel_4x4
#define KERNEL_8x8 hgemm_kernel_8x8

/**
 * @brief hgemm noTrans computation with 4x4 kernel : C = A*B,
 * 
 * @param m length of the row of matrix A
 * @param n length of the col of matrix B 
 * @param k length of the col of matrix A
 * @param a input matrix A
 * @param lda length of the col of matrix C
 * @param b input matrix B
 * @param ldb length of the col of matrix C
 * @param c output matrix C
 * @param ldc length of the col of matrix C
 */
void hgemm_noTrans_4x4(unsigned int m, unsigned int n, unsigned int k,
                       const __fp16 *a, unsigned int lda, const __fp16 *b,
                       unsigned int ldb, __fp16 *c, unsigned int ldc);

/**
 * @brief hgemm noTrans computation with 8x8 kernel : C = A*B,
 * 
 * @param m length of the row of matrix A
 * @param n length of the col of matrix B 
 * @param k length of the col of matrix A
 * @param a input matrix A
 * @param lda length of the col of matrix C
 * @param b input matrix B
 * @param ldb length of the col of matrix C
 * @param c output matrix C
 * @param ldc length of the col of matrix C
 */
void hgemm_noTrans_8x8(unsigned int m, unsigned int n, unsigned int k,
                       const __fp16 *a, unsigned int lda, const __fp16 *b,
                       unsigned int ldb, __fp16 *c, unsigned int ldc);
