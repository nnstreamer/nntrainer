// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Donghyeon Jeong <dhyeon.jeong@samsung.com>
 *
 * @file	clblast_interface.h
 * @date	12 May 2025
 * @brief	CLBlast library interface
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Donghyeon Jeong <dhyeon.jeong@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#ifndef __CLBLAST_INTERFACE_H__
#define __CLBLAST_INTERFACE_H__

#include <engine.h>

namespace nntrainer {

static ClContext *clblast_cc =
  static_cast<ClContext *>(Engine::Global().maybeGetRegisteredContext("gpu"));
static ClBufferManager &clBuffManagerInst = ClBufferManager::getInstance();

/**
 * @brief Multiplies n elements of vector x by a scalar constant alpha.
 * @param N Number of elements
 * @param alpha Scalar constant
 * @param X Vector X (input)
 * @param incX Increment for input
 */
void scal_cl(const unsigned int N, const float alpha, float *X,
             unsigned int incX = 1);

/**
 * @brief Copies the contents of vector x into vector y.
 *
 * @param N Number of elements
 * @param X Vector X (input)
 * @param Y Vector Y (output)
 * @param incX Increment for input
 * @param incY Increment for output
 * @note incX and incY are used to skip elements in the input and output
 */
void copy_cl(const unsigned int N, const float *X, float *Y,
             unsigned int incX = 1, unsigned int incY = 1);

/**
 * @brief Performs the operation y = alpha * x + y
 * @param N Number of elements
 * @param alpha Scalar constant
 * @param X Vector X (input)
 * @param Y Vector Y (output)
 */
void axpy_cl(const unsigned int N, const float alpha, const float *X, float *Y,
             unsigned int incX = 1, unsigned int incY = 1);

/**
 * @brief Multiplies n elements of the vectors x and y element-wise
 * @param N Number of elements
 * @param X Vector X (input)
 * @param Y Vector Y (input)
 * @param incX Increment for input X
 * @param incY Increment for input Y
 * @note incX and incY are used to skip elements in the X and Y
 */
float dot_cl(const unsigned int N, const float *X, const float *Y,
             unsigned int incX = 1, unsigned int incY = 1);

/**
 * @brief Accumulates the square of n elements in the x vector and takes the
 * square root.
 * @param N Number of elements
 * @param X Vector X (input)
 * @param incX Increment for input
 */
float nrm2_cl(const unsigned int N, const float *X, unsigned int incX = 1);

/**
 * @brief Computes the absolute sum of value in the vector X
 * @param N Number of elements
 * @param X Vector X (input)
 * @param incX Increment for input
 */
float asum_cl(const unsigned int N, const float *X, unsigned int incX = 1);

/**
 * @brief Index of absolute maximum value in a vector X
 * @param N Number of elements
 * @param X Vector X (input)
 * @param incX Increment for input
 */
int amax_cl(const unsigned int N, const float *X, unsigned int incX = 1);

/**
 * @brief Index of absolute minimum value in a vector X
 * @param N Number of elements
 * @param X Vector X (input)
 * @param incX Increment for input
 */
int amin_cl(const unsigned int N, const float *X, unsigned int incX = 1);

/**
 * @brief General matrix-vector multiplication
 * Performs the operation y = alpha * A * x + beta * y
 *
 * @param layout Data-layout of the matrix (Row-major or Column-major)
 * @param TransA Transpose flag for matrix A
 * @param M number of rows in A
 * @param N number of columns in A
 * @param alpha scalar multiplier for A
 * @param A Matrix A (input)
 * @param lda leading dimension of A
 * @param X Vector X (input)
 * @param beta scalar multiplier for Y
 * @param Y Vector Y (output)
 * @param incX increment for input
 * @param incY increment for output
 */
void gemv_cl(const unsigned int layout, bool TransA, const unsigned int M,
             const unsigned int N, const float alpha, const float *A,
             const unsigned int lda, const float *X, const float beta, float *Y,
             unsigned int incX = 1, unsigned int incY = 1);

/**
 * @brief General matrix-matrix multiplication
 * Performs the matrix product C = alpha * A * B + beta * C
 *
 * @param layout Data-layout of the matrix (Row-major or Column-major)
 * @param TransA Transpose flag for matrix A
 * @param TransB Transpose flag for matrix B
 * @param M number of rows in A and C
 * @param N number of columns in B and C
 * @param K number of columns in A and rows in B
 * @param alpha scalar multiplier for A and B
 * @param A Matrix A (input)
 * @param lda leading dimension of A
 * @param B Matrix B (input)
 * @param ldb leading dimension of B
 * @param beta scalar multiplier for C
 * @param C Matrix C (input/output)
 * @param ldc leading dimension of C
 * @note The result is stored in C, which is also the output matrix.
 */
void gemm_cl(const unsigned int layout, bool TransA, bool TransB,
             const unsigned int M, const unsigned int N, const unsigned int K,
             const float alpha, const float *A, const unsigned int lda,
             const float *B, const unsigned int ldb, const float beta, float *C,
             const unsigned int ldc);

/**
 * @brief Batched version of GEMM
 * As GEMM, but multiple operations are batched together for better performance.
 *
 * @param layout Data-layout of the matrix (Row-major or Column-major)
 * @param TransA Transpose flag for matrix A
 * @param TransB Transpose flag for matrix B
 * @param M number of rows in A and C
 * @param N number of columns in B and C
 * @param K number of columns in A and rows in B
 * @param alpha scalar multipliers for A and B
 * @param A Matrix A (input)
 * @param lda leading dimension of A
 * @param B Matrix B (input)
 * @param ldb leading dimension of B
 * @param beta scalar multipliers for C
 * @param C Matrix C (input/output)
 * @param ldc leading dimension of C
 * @param batch_size number of batches
 * @note The result is stored in C, which is also the output matrix.
 */
void gemm_batched_cl(const unsigned int layout, bool TransA, bool TransB,
                     const unsigned int M, const unsigned int N,
                     const unsigned int K, const float *alpha, const float *A,
                     const unsigned int lda, const float *B,
                     const unsigned int ldb, const float *beta, float *C,
                     const unsigned int ldc, const unsigned int batch_size);

/**
 * @brief Performs the im2col algorithm
 *
 * @param C channel size
 * @param H height of the input
 * @param W width of the input
 * @param kernel_h height of the kernel
 * @param kernel_w width of the kernel
 * @param pad_h padding height
 * @param pad_w padding width
 * @param stride_h stride height
 * @param stride_w stride width
 * @param dilation_h dilation height
 * @param dilation_w dilation width
 * @param input input tensor
 * @param output output tensor
 * @note The output tensor is a 2D matrix where each column corresponds to a
 * patch of the input tensor. The number of rows is equal to the number of
 * channels multiplied by the kernel size (kernel_h * kernel_w).
 * The number of columns is equal to the number of patches that can be
 * extracted from the input tensor.
 */
void im2col_cl(const unsigned int C, const unsigned int H, const unsigned int W,
               const unsigned int kernel_h, const unsigned int kernel_w,
               const unsigned int pad_h, const unsigned int pad_w,
               const unsigned int stride_h, const unsigned int stride_w,
               const unsigned int dilation_h, const unsigned int dilation_w,
               const float *input, float *output);

} // namespace nntrainer

#endif /* __CLBLAST_INTERFACE_H__ */
