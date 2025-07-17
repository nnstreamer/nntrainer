// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file   hgemm_util.h
 * @date   01 April 2024
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is for util functions for half-precision GEMM
 */

#include <assert.h>
#include <stdlib.h>

/**
 * @brief aligned dynamic allocation function
 *
 * @param sz amount of data to allocate
 * @return __fp16* addr of allocated memory
 */
__fp16 *alignedMalloc(unsigned int sz);

/**
 * @brief Get the mltpl of n that is bigger than or equal to x
 *
 * @param x unsigned int x
 * @param n unsigned int n
 * @return unsigned int
 */
unsigned int get_next_mltpl_of_n(unsigned int x, unsigned int n);

/**
 * @brief Get the mltpl of 2 power of n that is smaller than or equal to x
 *
 * @param x unsigned int x
 * @param n unsigned int n
 * @return unsigned int
 */
unsigned int get_prev_mltpl_of_2p_n(unsigned int x, unsigned int n);

/**
 * @brief from given output matrix address C, formulate fp32 version of it with
 * scale factor beta
 *
 * @param C __fp16* matrix to be converted
 * @param C32 float* converted and scaled matrix
 * @param M row size of matrix
 * @param N col size of matrix
 * @param beta scale factor beta
 */
void copy_C_to_C32(__fp16 *C, float *C32, unsigned int M, unsigned int N,
                   float beta = 0.F);

/**
 * @brief from matrix C32, copy to matrix C
 *
 * @param C32 float* matrix to be converted
 * @param C __fp16* converted matrix
 * @param M row size of matrix
 * @param N col size of matrix
 * @param beta scale factor beta
 */
void copy_C32_to_C(float *C32, __fp16 *C, unsigned int M, unsigned int N,
                   float beta = 0.F);
