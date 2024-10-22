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

#include <cl_context.h>
#include <opencl_buffer.h>
#include <opencl_kernel.h>
#include <string>

namespace nntrainer {

// get global cl_context to use in kernels
static ClContext cl_context_ref;

/**
 * @brief     sgemv computation : Y = A*X + Y
 * @param[in] matAdata float * for Matrix A
 * @param[in] vecXdata float * for Vector X
 * @param[in] vecYdata float * for Vector Y
 * @param[in] transA bool transpose
 * @param[in] dim1 number of A's columns
 * @param[in] dim2 number of A's rows
 * @param[in] lda number of X's columns
 * @param[in] context RunLayerContext reference
 */
void sgemv_cl(const float *matAdata, const float *vecXdata, float *vecYdata,
              bool TransA, unsigned int dim1, unsigned int dim2,
              unsigned int lda);

/**
 * @brief     dot computation : sum of all X * Y
 * @param[in] vecAdata float * for Vector A
 * @param[in] vecXdata float * for Vector X
 * @param[in] dim1 number of elements in both input vectors
 * @param[in] context RunLayerContext reference
 * @return    float dot product result
 */
float dot_cl(const float *vecAdata, const float *vecXdata, unsigned int dim1);

/**
 * @brief     sgemm computation : Y = op(A)*op(B) + C,
 * where op(X) is one of X or X**T
 * @param[in] transA bool transpose
 * @param[in] transB bool transpose
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
void sgemm_cl(bool TransA, bool TransB, const float *A, const float *B,
              float *C, unsigned int M, unsigned int N, unsigned int K,
              unsigned int lda, unsigned int ldb, unsigned int ldc);

/**
 * @brief     addition : sum of all input vectors
 * @param[in] input float * for input
 * @param[in] res float * for result/output
 * @param[in] size number of elements in input vector
 * @param[in] context RunLayerContext reference
 */
void addition_cl(const float *input, float *res, unsigned int size);

/**
 * @brief     sscal value element by element immediately
 * @param[in] X float * input
 * @param[in] N unsigned int number of elements
 * @param[in] alpha float multiplier
 * @param[in] context RunLayerContext reference
 */
void sscal_cl(float *X, const unsigned int N, const float alpha);

/**
 * @brief     transpose computation
 * @param[in] input float * for Input Tensor
 * @param[in] res float * for Output Tensor
 * @param[in] input_batch_size  represents the number of samples in the input
 * tensor
 * @param[in] input_channels   represents the channels of the input tensor
 * @param[in] input_height   represents the height of the input tensor
 * @param[in] input_width   represents the width of the input tensor
 * @param[in] axis   transpose about axis, 0-> channels & height, 1-> height &
 * width, 2-> channels & width
 */
void transpose_cl_axis(const float *in, float *res,
                       unsigned int input_batch_size,
                       unsigned int input_channels, unsigned int input_height,
                       unsigned int input_width, unsigned int axis);

#ifdef ENABLE_FP16

/**
 * @brief     fp16 sgemv computation : Y = A*X + Y
 * @param[in] matAdata fp16 * for Matrix A
 * @param[in] vecXdata fp16 * for Vector X
 * @param[in] vecYdata fp16 * for Vector Y
 * @param[in] transA bool transpose
 * @param[in] dim1 number of A's columns
 * @param[in] dim2 number of A's rows
 * @param[in] lda number of X's columns
 * @param[in] context RunLayerContext reference
 */
void sgemv_cl(const __fp16 *matAdata, const __fp16 *vecXdata, __fp16 *vecYdata,
              bool TransA, unsigned int dim1, unsigned int dim2,
              unsigned int lda);

/**
 * @brief     fp16 dot computation : sum of all X * Y
 * @param[in] vecAdata fp16 * for Vector A
 * @param[in] vecXdata fp16 * for Vector X
 * @param[in] dim1 number of elements in both input vectors
 * @param[in] context RunLayerContext reference
 * @return    fp16 dot product result
 */
__fp16 dot_cl(const __fp16 *vecAdata, const __fp16 *vecXdata,
              unsigned int dim1);

/**
 * @brief     fp16 sgemm computation : Y = op(A)*op(B) + C,
 * where op(X) is one of X or X**T
 * @param[in] transA bool transpose
 * @param[in] transB bool transpose
 * @param[in] A fp16 * for Matrix A
 * @param[in] B fp16 * for Matrix B
 * @param[in] C fp16 * for Matrix C
 * @param[in] M number of op(A)'s and C's row
 * @param[in] N number of op(B)'s and C's columns
 * @param[in] K number of op(A)'s and columns and op(B)'s rows
 * @param[in] lda number of A's columns
 * @param[in] ldb number of B's columns
 * @param[in] ldc number of C's columns
 * @param[in] context RunLayerContext reference
 */
void sgemm_cl(bool TransA, bool TransB, const __fp16 *A, const __fp16 *B,
              __fp16 *C, unsigned int M, unsigned int N, unsigned int K,
              unsigned int lda, unsigned int ldb, unsigned int ldc);

/**
 * @brief     fp16 addition : sum of all input vectors
 * @param[in] input fp16 * for input
 * @param[in] res fp16 * for result/output
 * @param[in] size number of elements in input vector
 * @param[in] context RunLayerContext reference
 */
void addition_cl(const __fp16 *input, __fp16 *res, unsigned int size);

/**
 * @brief     fp16 sscal value element by element immediately
 * @param[in] X __fp16 * input
 * @param[in] N unsigned int number of elements
 * @param[in] alpha float multiplier
 * @param[in] context RunLayerContext reference
 */
void sscal_cl(__fp16 *X, const unsigned int N, const float alpha);

/**
 * @brief     transpose computation
 * @param[in] input fp16 * for Input Tensor
 * @param[in] res fp16 * for Output Tensor
 * @param[in] input_batch_size  represents the number of samples in the input
 * tensor
 * @param[in] input_channels   represents the channels of the input tensor
 * @param[in] input_height   represents the height of the input tensor
 * @param[in] input_width   represents the width of the input tensor
 * @param[in] axis   transpose about axis, 0-> channels & height, 1-> height &
 * width, 2-> channels and width
 */
void transpose_cl_axis(const __fp16 *in, __fp16 *res,
                       unsigned int input_batch_size,
                       unsigned int input_channels, unsigned int input_height,
                       unsigned int input_width, unsigned int axis);
#endif

} // namespace nntrainer
#endif /* __BLAS_KERNELS_H__ */
