// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Debadri Samaddar <s.debadri@samsung.com>
 *
 * @file	blas_kernels.h
 * @date	14 May 2024
 * @brief	Common blas OpenCL kernels
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Debadri Samaddar <s.debadri@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#ifndef __BLAS_KERNELS_H__
#define __BLAS_KERNELS_H__

#include <layer_context.h>
#include <opencl_buffer.h>
#include <opencl_kernel.h>
#include <string>

namespace nntrainer {

/**
 * @brief declaring global kernel objects
 */
extern opencl::Kernel kernel_sgemv;
extern opencl::Kernel kernel_sgemm;
extern opencl::Kernel kernel_dot;

/**
 * @brief     sgemv computation : Y = A*X + Y
 * @param[in] matAdata float * for Matrix A
 * @param[in] vecXdata float * for Vector X
 * @param[in] vecYdata float * for Vector Y
 * @param[in] dim1 number of A's row
 * @param[in] dim2 number of X's columns
 * @param[in] lda number of X's columns
 * @param[in] context RunLayerContext reference
 */
void sgemv_cl(const float *matAdata, const float *vecXdata, float *vecYdata,
              unsigned int dim1, unsigned int dim2, unsigned int lda,
              RunLayerContext &context);

/**
 * @brief     dot computation : sum of all X * Y
 * @param[in] matAdata float * for Vector A
 * @param[in] vecXdata float * for Vector X
 * @param[in] dim1 number of elements in both input vectors
 * @param[in] context RunLayerContext reference
 */
float dot_cl(const float *matAdata, const float *vecXdata, unsigned int dim1,
             RunLayerContext &context);

/**
 * @brief     sgemm computation : Y = op(A)*op(B) + C,
 * where op(X) is one of X or X**T
 * @param[in] A float * for Matrix A
 * @param[in] B float * for Matrix B
 * @param[in] C float * for Matrix C
 * @param[in] M number of op(A)'s and C's row
 * @param[in] N number of op(B)'s and C's columns
 * @param[in] K number of op(A)'s and columns and op(B)'s rows
 * @param[in] lda number of A's columns
 * @param[in] ldb number of B's columns
 * @param[in] ldc number of C's columns
 * @param[in] context RunLayerContext reference
 */
void sgemm_cl(const float *A, const float *B, float *C, unsigned int M,
              unsigned int N, unsigned int K, unsigned int lda,
              unsigned int ldb, unsigned int ldc, RunLayerContext &context);

} // namespace nntrainer
#endif /* __BLAS_KERNELS_H__ */
