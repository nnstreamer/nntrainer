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
extern opencl::Kernel kernel_sgemm_noTrans;
extern opencl::Kernel kernel_sgemm_transAB;
extern opencl::Kernel kernel_sgemm_transA;
extern opencl::Kernel kernel_sgemm_transB;
extern opencl::Kernel kernel_dot;
extern opencl::Kernel kernel_addition;
extern opencl::Kernel kernel_sscal;

/**
 * @brief     sgemv computation : Y = A*X + Y
 * @param[in] matAdata float * for Matrix A
 * @param[in] vecXdata float * for Vector X
 * @param[in] vecYdata float * for Vector Y
 * @param[in] dim1 number of A's columns
 * @param[in] dim2 number of A's rows
 * @param[in] lda number of X's columns
 * @param[in] context RunLayerContext reference
 */
void sgemv_cl(const float *matAdata, const float *vecXdata, float *vecYdata,
              unsigned int dim1, unsigned int dim2, unsigned int lda,
              RunLayerContext &context);

/**
 * @brief     dot computation : sum of all X * Y
 * @param[in] vecAdata float * for Vector A
 * @param[in] vecXdata float * for Vector X
 * @param[in] dim1 number of elements in both input vectors
 * @param[in] context RunLayerContext reference
 * @return    float dot product result
 */
float dot_cl(const float *vecAdata, const float *vecXdata, unsigned int dim1,
             RunLayerContext &context);

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
              unsigned int lda, unsigned int ldb, unsigned int ldc,
              RunLayerContext &context);

/**
 * @brief     addition : sum of all input vectors
 * @param[in] input float * for input
 * @param[in] res float * for result/output
 * @param[in] size number of elements in input vector
 * @param[in] context RunLayerContext reference
 */
void addition_cl(const float *input, float *res, unsigned int size,
                 RunLayerContext &context);

/**
 * @brief     sscal value element by element immediately
 * @param[in] X float * input
 * @param[in] N unsigned int number of elements
 * @param[in] alpha float multiplier
 * @param[in] context RunLayerContext reference
 */
void sscal_cl(float *X, const unsigned int N, const float alpha,
              RunLayerContext &context);

#ifdef ENABLE_FP16
/**
 * @brief declaring global fp16 kernel objects
 */
extern opencl::Kernel kernel_sgemv_fp16;
extern opencl::Kernel kernel_sgemm_noTrans_fp16;
extern opencl::Kernel kernel_sgemm_transAB_fp16;
extern opencl::Kernel kernel_sgemm_transA_fp16;
extern opencl::Kernel kernel_sgemm_transB_fp16;
extern opencl::Kernel kernel_dot_fp16;
extern opencl::Kernel kernel_addition_fp16;
extern opencl::Kernel kernel_sscal_fp16;

/**
 * @brief     fp16 sgemv computation : Y = A*X + Y
 * @param[in] matAdata fp16 * for Matrix A
 * @param[in] vecXdata fp16 * for Vector X
 * @param[in] vecYdata fp16 * for Vector Y
 * @param[in] dim1 number of A's columns
 * @param[in] dim2 number of A's rows
 * @param[in] lda number of X's columns
 * @param[in] context RunLayerContext reference
 */
void sgemv_cl(const _FP16 *matAdata, const _FP16 *vecXdata, _FP16 *vecYdata,
              unsigned int dim1, unsigned int dim2, unsigned int lda,
              RunLayerContext &context);

/**
 * @brief     fp16 dot computation : sum of all X * Y
 * @param[in] vecAdata fp16 * for Vector A
 * @param[in] vecXdata fp16 * for Vector X
 * @param[in] dim1 number of elements in both input vectors
 * @param[in] context RunLayerContext reference
 * @return    fp16 dot product result
 */
_FP16 dot_cl(const _FP16 *vecAdata, const _FP16 *vecXdata, unsigned int dim1,
             RunLayerContext &context);

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
void sgemm_cl(bool TransA, bool TransB, const _FP16 *A, const _FP16 *B,
              _FP16 *C, unsigned int M, unsigned int N, unsigned int K,
              unsigned int lda, unsigned int ldb, unsigned int ldc,
              RunLayerContext &context);

/**
 * @brief     fp16 addition : sum of all input vectors
 * @param[in] input fp16 * for input
 * @param[in] res fp16 * for result/output
 * @param[in] size number of elements in input vector
 * @param[in] context RunLayerContext reference
 */
void addition_cl(const _FP16 *input, _FP16 *res, unsigned int size,
                 RunLayerContext &context);

/**
 * @brief     fp16 sscal value element by element immediately
 * @param[in] X _FP16 * input
 * @param[in] N unsigned int number of elements
 * @param[in] alpha float multiplier
 * @param[in] context RunLayerContext reference
 */
void sscal_cl(_FP16 *X, const unsigned int N, const float alpha,
              RunLayerContext &context);
#endif

} // namespace nntrainer
#endif /* __BLAS_KERNELS_H__ */
