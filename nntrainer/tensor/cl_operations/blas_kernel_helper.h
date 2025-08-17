/**
 * Copyright (C) 2025 Daekyoung Jung <daekyoung.jung@gmail.com>
 *
 * @file	blas_kernel_helper.h
 * @date	07 August 2025
 * @brief	functions that are used for preprocessing of tensor
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Daekyoung Jung <daekyoung.jung@gmail.com>
 * @bug		No known bugs except for NYI items
 *
 */

#ifndef __BLAS_KERNELS_HELPER_H__
#define __BLAS_KERNELS_HELPER_H__

namespace nntrainer {

/**
 * @brief Convert array of block_q4_0x8 into two arrays of scale, qs (st)
 * @param[in] x Array of struct block_q4_0x8
 * @param[out] d Array of unsigned short
 * @param[out] qs Array of unsigned char
 * @param N size of @param x
 * @param K number of columns of the original weight matrix
 */
void convert_q4_0x8_st(const void *x, unsigned short *d, unsigned char *qs,
                       int N, int K);

/**
 * @brief Convert array of block_q4_0x8 into two arrays of scale, qs (multi th)
 * @param[in] x Array of struct block_q4_0x8
 * @param[out] d Array of unsigned short
 * @param[out] qs Array of unsigned char
 * @param N size of @param x
 * @param K number of columns of the original weight matrix
 */
void convert_q4_0x8_omp(const void *x, unsigned short *d, unsigned char *qs,
                        int N, int K);

} // namespace nntrainer

#endif /* __BLAS_KERNELS_HELPER_H__ */
