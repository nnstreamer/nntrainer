// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Debadri Samaddar <s.debadri@samsung.com>
 *
 * @file	blas_kernel_strings.h
 * @date	18 Sep 2024
 * @brief	All blas OpenCL kernel strings
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Debadri Samaddar <s.debadri@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#ifndef __BLAS_KERNEL_STRINGS_H__
#define __BLAS_KERNEL_STRINGS_H__

#include <string>

namespace nntrainer {
static const std::string sgemv_cl_kernel_ =
  R"(__kernel void sgemv_cl(const __global float* A, const __global float* X,
                      __global float* Y, unsigned int N, unsigned int lda) {                                            
        unsigned int i;
        i = get_global_id(0);                         
        float y0 = 0.0f;
        for (unsigned int j = 0; j < N; j++)                         
            y0 += A[i + j * lda] * X[j]; 
        Y[i] = y0;                            
          
    })";

static const std::string dot_cl_kernel_ =
  R"(__kernel void dot_cl(const __global float* A, const __global float* X, unsigned int K, __global float* res) {
        *res = 0;
        for (unsigned int i = 0; i < K; i++){
            *res += A[i] * X[i];
        }
    })";

static const std::string sgemm_cl_noTrans_kernel_ =
  R"(__kernel void sgemm_cl_noTrans(const __global float* A, const __global float* B,
                      __global float* C, unsigned int K, unsigned int lda, unsigned int ldb, unsigned int ldc) {
        
        unsigned int m = get_global_id(0);
        unsigned int n = get_global_id(1);
        float c = 0.0f;
        for (unsigned int k = 0; k < K; ++k) {
          float a, b;
          a = A[m * lda + k];
          b = B[k * ldb + n];
          c += a * b;
        }
        C[m * ldc + n] = c;
    })";

static const std::string sgemm_cl_transA_kernel_ =
  R"(__kernel void sgemm_cl_transA(const __global float* A, const __global float* B,
                      __global float* C, unsigned int K, unsigned int lda, unsigned int ldb, unsigned int ldc) {
        
        unsigned int m = get_global_id(0);
        unsigned int n = get_global_id(1);
        float c = 0.0f;
        for (unsigned int k = 0; k < K; ++k) {
          float a, b;
          a = A[k * lda + m];
          b = B[k * ldb + n];
          c += a * b;
        }
        C[m * ldc + n] = c;
    })";

static const std::string sgemm_cl_transB_kernel_ =
  R"(__kernel void sgemm_cl_transB(const __global float *A, const __global float *B,
                              __global float *C, unsigned int K,
                              unsigned int lda, unsigned int ldb,
                              unsigned int ldc) {

        unsigned int m = get_global_id(0);
        unsigned int n = get_global_id(1);
        float c = 0.0f;
        for (unsigned int k = 0; k < K; ++k) {
          float a, b;
          a = A[m * lda + k];
          b = B[n * ldb + k];
          c += a * b;
        }
        C[m * ldc + n] = c;
    })";

static const std::string sgemm_cl_transAB_kernel_ =
  R"(__kernel void sgemm_cl_transAB(const __global float *A, const __global float *B,
                               __global float *C, unsigned int K,
                               unsigned int lda, unsigned int ldb,
                               unsigned int ldc) {

        unsigned int m = get_global_id(0);
        unsigned int n = get_global_id(1);
        float c = 0.0f;
        for (unsigned int k = 0; k < K; ++k) {
          float a, b;
          a = A[k * lda + m];
          b = B[n * ldb + k];
          c += a * b;
        }
        C[m * ldc + n] = c;
    })";

static const std::string addition_cl_kernel_ =
  R"(__kernel void addition_cl(__global const float* input, __global float* output, const unsigned int size) {
    #pragma printf_support
    size_t idx = get_global_id(0);
    if (idx < size) {
        output[idx] = output[idx] + input[idx];
    }
  })";

static const std::string sscal_cl_kernel_ =
  R"(__kernel void sscal_cl(__global float* X, const float alpha) {
        
        unsigned int i = get_global_id(0);
        X[i] *= alpha;
    })";

#ifdef ENABLE_FP16
static const std::string sgemv_cl_kernel_fp16_ =
  R"(
    #pragma OPENCL EXTENSION cl_khr_fp16 : enable
    
    __kernel void sgemv_cl_fp16(const __global half* A, const __global half* X,
                      __global half* Y, unsigned int N, unsigned int lda) {                                            
        unsigned int i;
        i = get_global_id(0);                         
        half y0 = 0.0f;
        for (unsigned int j = 0; j < N; j++)                         
            y0 += A[i + j * lda] * X[j]; 
        Y[i] = y0;                            
          
    })";

static const std::string dot_cl_kernel_fp16_ =
  R"(
    #pragma OPENCL EXTENSION cl_khr_fp16 : enable

    __kernel void dot_cl_fp16(const __global half* A, const __global half* X, unsigned int K, __global half* res) {
        *res = 0;
        for (unsigned int i = 0; i < K; i++){
            *res += A[i] * X[i];
        }
    })";

static const std::string sgemm_cl_noTrans_kernel_fp16_ =
  R"(
    #pragma OPENCL EXTENSION cl_khr_fp16 : enable

    __kernel void sgemm_cl_noTrans_fp16(const __global half* A, const __global half* B,
                      __global half* C, unsigned int K, unsigned int lda, unsigned int ldb, unsigned int ldc) {
        
        unsigned int m = get_global_id(0);
        unsigned int n = get_global_id(1);
        half c = 0.0f;
        for (unsigned int k = 0; k < K; ++k) {
          half a, b;
          a = A[m * lda + k];
          b = B[k * ldb + n];
          c += a * b;
        }
        C[m * ldc + n] = c;
    })";

static const std::string sgemm_cl_transA_kernel_fp16_ =
  R"(
    #pragma OPENCL EXTENSION cl_khr_fp16 : enable

    __kernel void sgemm_cl_transA_fp16(const __global half* A, const __global half* B,
                      __global half* C, unsigned int K, unsigned int lda, unsigned int ldb, unsigned int ldc) {
        
        unsigned int m = get_global_id(0);
        unsigned int n = get_global_id(1);
        half c = 0.0f;
        for (unsigned int k = 0; k < K; ++k) {
          half a, b;
          a = A[k * lda + m];
          b = B[k * ldb + n];
          c += a * b;
        }
        C[m * ldc + n] = c;
    })";

static const std::string sgemm_cl_transB_kernel_fp16_ =
  R"(
    #pragma OPENCL EXTENSION cl_khr_fp16 : enable

    __kernel void sgemm_cl_transB_fp16(const __global half* A, const __global half* B,
                      __global half* C, unsigned int K, unsigned int lda, unsigned int ldb, unsigned int ldc) {
        
        unsigned int m = get_global_id(0);
        unsigned int n = get_global_id(1);
        half c = 0.0f;
        for (unsigned int k = 0; k < K; ++k) {
          half a, b;
          a = A[m * lda + k];
          b = B[n * ldb + k];
          c += a * b;
        }
        C[m * ldc + n] = c;
    })";

static const std::string sgemm_cl_transAB_kernel_fp16_ =
  R"(
    #pragma OPENCL EXTENSION cl_khr_fp16 : enable

    __kernel void sgemm_cl_transAB_fp16(const __global half* A, const __global half* B,
                      __global half* C, unsigned int K, unsigned int lda, unsigned int ldb, unsigned int ldc) {
        
        unsigned int m = get_global_id(0);
        unsigned int n = get_global_id(1);
        half c = 0.0f;
        for (unsigned int k = 0; k < K; ++k) {
          half a, b;
          a = A[k * lda + m];
          b = B[n * ldb + k];
          c += a * b;
        }
        C[m * ldc + n] = c;
    })";

static const std::string addition_cl_kernel_fp16_ =
  R"(
    #pragma OPENCL EXTENSION cl_khr_fp16 : enable

    __kernel void addition_cl_fp16(__global const half* input, __global half* output, const unsigned int size) {
    size_t idx = get_global_id(0);
    if (idx < size) {
        output[idx] = output[idx] + input[idx];
    }
  })";

static const std::string sscal_cl_kernel_fp16_ =
  R"(
    #pragma OPENCL EXTENSION cl_khr_fp16 : enable

    __kernel void sscal_cl_fp16(__global half* X, const float alpha) {
        
        unsigned int i = get_global_id(0);
        X[i] *= alpha;
    })";
#endif
} // namespace nntrainer
#endif /* __BLAS_KERNEL_INTERFACE_H__ */
