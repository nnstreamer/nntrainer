// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file   hgemm_kernel_8x8.h
 * @date   01 April 2024
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is half-precision GEMM 8x8 kernel
 *
 */

#include <assert.h>
#include <cmath>
#include <hgemm_common.h>
#include <math.h>
#include <stdlib.h>

/**
 * @brief hgemm 4x4 kernel sc = sa * sb
 * 
 * @param m length of the row of matrix A
 * @param n length of the col of matrix B 
 * @param k length of the col of matrix A
 * @param sa sub-matrix of input matrix A
 * @param sb sub-matrix of input matrix B
 * @param sc sub-matrix of output matrix C
 * @param ldc leading-dimension of matrix C
 */
void hgemm_kernel_8x8(unsigned int m, unsigned int n, unsigned int k,
                      __fp16 *sa, __fp16 *sb, __fp16 *sc, unsigned int ldc);
