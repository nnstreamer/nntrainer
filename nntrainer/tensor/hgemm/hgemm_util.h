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

#include <arm_neon.h>
#include <assert.h>
#include <stdlib.h>

/**
 * @brief aligned dynamic allocation function
 *
 * @param sz amount of data to allocate
 * @return __fp16* addr of allocated memory
 */
__fp16 *alignedMalloc(unsigned int sz);

unsigned int get_next_mltpl_of_n(unsigned int x, unsigned int n);

unsigned int get_prev_mltpl_of_2p_n(unsigned int x, unsigned int n);

void copy_C_to_C32(__fp16 *C, float *C32, unsigned int M, unsigned int N,
                   float beta = 0.F);

void copy_C32_to_C(float *C32, __fp16 *C, unsigned int M, unsigned int N,
                   float beta = 0.F);
