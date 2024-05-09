// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file   matrix_transpose_neon.h
 * @date   09 May 2024
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is header file for matrix transpose using NEON
 *
 */

#include <cstdint>

/**
 * @brief Matrix Transpose fallback. Note that this transposes a matrix, which
 * is 2D Tensor
 *
 * @tparam T dataType of the incoming matrix. Implement more Kernels and connect
 * to this function in order to support more datatypes.
 * @param M row length of input matrix
 * @param N col length of input matrix
 * @param src source data of input matrix
 * @param ld_src data offset of input matrix
 * @param dst destination data of this function
 * @param ld_dst data offset of output matrix 
 */
template <typename T>
void transpose_fallback(unsigned int M, unsigned int N, const T *src,
                   unsigned int ld_src, T *dst, unsigned int ld_dst);

/**
 * @brief Matrix Transpose using NEON. Note that this transposes a matrix, which
 * is 2D Tensor
 *
 * @tparam T dataType of the incoming matrix. Implement more Kernels and connect
 * to this function in order to support more datatypes.
 * @param M row length of input matrix
 * @param N col length of input matrix
 * @param src source data of input matrix
 * @param ld_src data offset of input matrix
 * @param dst destination data of this function
 * @param ld_dst data offset of output matrix 
 */
template <typename T>
void transpose_neon(unsigned int M, unsigned int N, const T *src,
                    unsigned int ld_src, T *dst, unsigned int ld_dst);
