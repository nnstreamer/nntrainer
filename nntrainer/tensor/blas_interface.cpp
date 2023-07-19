// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   blas_interface.cpp
 * @date   28 Aug 2020
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is dummy header for blas support
 *
 */

#include <blas_interface.h>
#include <iostream>
#include <nntrainer_error.h>

#include <cmath>

#define sgemv_loop(ci, cj, cM, cN)           \
  do {                                       \
    double y0;                               \
    unsigned int i, j;                       \
    for (ci = 0; ci != cM; ci++) {           \
      y0 = Y[ci * incy] * beta;              \
      for (cj = 0; cj != cN; cj++)           \
        y0 += A[i + j * lda] * X[cj * incx]; \
      Y[ci * incy] = y0;                     \
    }                                        \
  } while (0);

namespace nntrainer {

#ifndef USE_BLAS

static void saxpy_raw(const unsigned int N, const float alpha, const float *X,
                      const int incX, float *Y, const int incY) {
  if (incX < 0 or incY < 0)
    throw std::invalid_argument(
      "Error: negative inc not supported without cblas");
  for (unsigned int i = 0; i < N; ++i)
    Y[i * incY] = Y[i * incY] + X[i * incX] * alpha;
}

static void saxpy_FP16(const unsigned int N, const float alpha, const __fp16 *X,
                       const int incX, __fp16 *Y, const int incY) {
  if (incX < 0 or incY < 0)
    throw std::invalid_argument(
      "Error: negative inc not supported without cblas");
  for (unsigned int i = 0; i < N; ++i)
    Y[i * incY] = Y[i * incY] + X[i * incX] * alpha;
}

static void sgemv_raw(CBLAS_ORDER order, CBLAS_TRANSPOSE TransA,
                      const unsigned int M, const unsigned int N,
                      const float alpha, const float *A, const unsigned int lda,
                      const float *X, const int incX, const float beta,
                      float *Y, const int incY) {

  unsigned int incy = abs(incY);
  unsigned int incx = abs(incX);

  if (TransA == CblasTrans) {
    sgemv_loop(i, j, N, M);
  } else {
    sgemv_loop(j, i, M, N);
  }
}

static void sgemv_FP16(CBLAS_ORDER order, CBLAS_TRANSPOSE TransA,
                       const unsigned int M, const unsigned int N,
                       const float alpha, const __fp16 *A,
                       const unsigned int lda, const __fp16 *X, const int incX,
                       const float beta, __fp16 *Y, const int incY) {

  unsigned int incy = abs(incY);
  unsigned int incx = abs(incX);

  if (TransA == CblasTrans) {
    sgemv_loop(i, j, N, M);
  } else {
    sgemv_loop(j, i, M, N);
  }
}

static float sdot_raw(const unsigned int N, const float *X,
                      const unsigned int incX, const float *Y,
                      const unsigned int incY) {
  float ret = 0;
  for (unsigned int i = 0; i < N; ++i) {
    ret += X[i * incX] * Y[i * incY];
  }
  return ret;
}

static __fp16 sdot_FP16(const unsigned int N, const __fp16 *X,
                        const unsigned int incX, const __fp16 *Y,
                        const unsigned int incY) {
  __fp16 ret = 0;
  for (unsigned int i = 0; i < N; ++i) {
    ret += X[i * incX] * Y[i * incY];
  }
  return ret;
}

static void scopy_raw(const unsigned int N, const float *X, const int incX,
                      float *Y, const int incY) {
  unsigned int incy = abs(incY);
  unsigned int incx = abs(incX);

  for (unsigned int i = 0; i < N; ++i)
    Y[i * incy] = X[i * incx];
}

static void scopy_FP16(const unsigned int N, const __fp16 *X, const int incX,
                       __fp16 *Y, const int incY) {
  unsigned int incy = abs(incY);
  unsigned int incx = abs(incX);

  for (unsigned int i = 0; i < N; ++i)
    Y[i * incy] = X[i * incx];
}

static void sscal_raw(const unsigned int N, const float alpha, float *X,
                      const int incX) {
  unsigned int incx = abs(incX);

  for (unsigned int i = 0; i < N; ++i)
    X[i * incx] = alpha * X[i * incx];
}

void sscal(const unsigned int N, const float alpha, __fp16 *X, const int incX) {
  unsigned int incx = abs(incX);

  for (unsigned int i = 0; i < N; ++i)
    X[i * incx] = alpha * X[i * incx];
}

void sscal(const unsigned int N, const float alpha, void *X, const int incX,
           ml::train::TensorDim::DataType d_type) {
#ifdef USE_BLAS
#ifdef BLAS_NUM_THREADS
  openblas_set_num_threads(BLAS_NUM_THREADS);
#endif
  if (d_type == ml::train::TensorDim::DataType::FP32)
    cblas_sscal(N, alpha, (float *)X, incX);
#else
  if (d_type == ml::train::TensorDim::DataType::FP32) {
    sscal_raw(N, alpha, (float *)X, incX);
  } else if (d_type == ml::train::TensorDim::DataType::FP16) {
    sscal(N, alpha, (__fp16 *)X, incX);
  }
#endif
}

void sscal(const unsigned int N, const float alpha, float *X, const int incX) {
#ifdef USE_BLAS
#ifdef BLAS_NUM_THREADS
  openblas_set_num_threads(BLAS_NUM_THREADS);
#endif
  cblas_sscal(N, alpha, (float *)X, incX);
#else
  sscal_raw(N, alpha, (float *)X, incX);
#endif
}

static float snrm2_raw(const unsigned int N, const float *X, const int incX) {
  unsigned int incx = abs(incX);
  float sum = 0.0f;
  float tmp;
#pragma omp parallel for private(tmp) reduction(+ : sum)
  for (unsigned int i = 0; i < N; i++) {
    tmp = X[i * incx];
    sum += tmp * tmp;
  }
  return sqrt(sum);
}

static float snrm2_FP16(const unsigned int N, const __fp16 *X, const int incX) {
  unsigned int incx = abs(incX);
  __fp16 sum = 0.0f;
  __fp16 tmp;
#pragma omp parallel for private(tmp) reduction(+ : sum)
  for (unsigned int i = 0; i < N; i++) {
    tmp = X[i * incx];
    sum += tmp * tmp;
  }
  return sqrt(sum);
}

static void sgemm_raw(CBLAS_ORDER order, CBLAS_TRANSPOSE TransA,
                      CBLAS_TRANSPOSE TransB, const unsigned int M,
                      const unsigned int N, const unsigned int K,
                      const float alpha, const float *A, const unsigned int lda,
                      const float *B, const unsigned int ldb, const float beta,
                      float *C, const unsigned int ldc) {

  for (unsigned int m = 0; m < M; ++m) {
    for (unsigned int n = 0; n < N; ++n) {
      double c = 0.0;
      float c_old = C[m * ldc + n];
      for (unsigned int k = 0; k < K; ++k) {
        float a, b;
        a = ((TransA == CblasTrans) ? A[k * lda + m] : A[m * lda + k]);
        b = ((TransB == CblasTrans) ? B[n * ldb + k] : B[k * ldb + n]);
        c += a * b;
      }
      C[m * ldc + n] = alpha * c;
      if (beta != 0.0)
        C[m * ldc + n] += beta * c_old;
    }
  }
}

static void sgemm_FP16(CBLAS_ORDER order, CBLAS_TRANSPOSE TransA,
                       CBLAS_TRANSPOSE TransB, const unsigned int M,
                       const unsigned int N, const unsigned int K,
                       const float alpha, const __fp16 *A,
                       const unsigned int lda, const __fp16 *B,
                       const unsigned int ldb, const float beta, __fp16 *C,
                       const unsigned int ldc) {

  for (unsigned int m = 0; m < M; ++m) {
    for (unsigned int n = 0; n < N; ++n) {
      double c = 0.0;
      __fp16 c_old = C[m * ldc + n];
      for (unsigned int k = 0; k < K; ++k) {
        __fp16 a, b;
        a = ((TransA == CblasTrans) ? A[k * lda + m] : A[m * lda + k]);
        b = ((TransB == CblasTrans) ? B[n * ldb + k] : B[k * ldb + n]);
        c += a * b;
      }
      C[m * ldc + n] = alpha * c;
      if (beta != 0.0)
        C[m * ldc + n] += beta * c_old;
    }
  }
}

static unsigned int isamax_raw(const unsigned int N, const float *X,
                               const int incX) {

  unsigned int max_idx = 0;
  float max_val = X[0];
  for (unsigned int n = 1; n < N; n += incX) {
    float cur_val = abs(X[n]);
    if (cur_val > max_val) {
      max_val = cur_val;
      max_idx = n;
    }
  }

  return max_idx;
}

static unsigned int isamax_FP16(const unsigned int N, const __fp16 *X,
                                const int incX) {

  unsigned int max_idx = 0;
  __fp16 max_val = X[0];
  for (unsigned int n = 1; n < N; n += incX) {
    __fp16 cur_val = abs(X[n]);
    if (cur_val > max_val) {
      max_val = cur_val;
      max_idx = n;
    }
  }

  return max_idx;
}

#endif

void saxpy(const unsigned int N, const float alpha, const void *X,
           const int incX, void *Y, const int incY,
           ml::train::TensorDim::DataType d_type) {
#ifdef USE_BLAS
#ifdef BLAS_NUM_THREADS
  openblas_set_num_threads(BLAS_NUM_THREADS);
#endif
  cblas_saxpy(N, alpha, X, incX, Y, incY);
#else
  if (d_type == ml::train::TensorDim::DataType::FP32) {
    saxpy_raw(N, alpha, (float *)X, incX, (float *)Y, incY);
  } else if (d_type == ml::train::TensorDim::DataType::FP16) {
    saxpy_FP16(N, alpha, (__fp16 *)X, incX, (__fp16 *)Y, incY);
  }
#endif
}

void saxpy(const unsigned int N, const float alpha, const float *X,
           const int incX, float *Y, const int incY) {
#ifdef USE_BLAS
#ifdef BLAS_NUM_THREADS
  openblas_set_num_threads(BLAS_NUM_THREADS);
#endif
  cblas_saxpy(N, alpha, X, incX, Y, incY);
#else
  saxpy_raw(N, alpha, X, incX, Y, incY);
#endif
}

void saxpy(const unsigned int N, const float alpha, const __fp16 *X,
           const int incX, __fp16 *Y, const int incY) {
  saxpy_FP16(N, alpha, X, incX, Y, incY);
}

void sgemm(CBLAS_ORDER order, CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB,
           const unsigned int M, const unsigned int N, const unsigned int K,
           const float alpha, const void *A, const unsigned int lda,
           const void *B, const unsigned int ldb, const float beta, void *C,
           const unsigned int ldc, ml::train::TensorDim::DataType d_type) {
#ifdef USE_CUBLAS
  int devID = 0;
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, devID);
  float *d_A, *d_B, *d_C;

  unsigned int size_A = M * K * sizeof(float);
  unsigned int size_B = K * N * sizeof(float);
  unsigned int size_C = M * N * sizeof(float);

  cudaMalloc((void **)&d_A, size_A);
  cudaMalloc((void **)&d_B, size_B);
  cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice);
  cudaMalloc((void **)&d_C, size_C);

  cublasHandle_t handle;
  cublasCreate(&handle);

  cublasOperation_t transA = (TransA == CblasTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t transB = (TransB == CblasTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasSgemm(handle, transA, transB, N, M, K, &alpha, d_B, N, d_A, K, &beta,
              d_C, N);

  cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost);
  cublasDestroy(handle);
#elif defined USE_BLAS
#ifdef BLAS_NUM_THREADS
  openblas_set_num_threads(BLAS_NUM_THREADS);
#endif
  cblas_sgemm(order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C,
              ldc);
#else
  if (d_type == ml::train::TensorDim::DataType::FP32) {
    sgemm_raw(order, TransA, TransB, M, N, K, alpha, (float *)A, lda,
              (float *)B, ldb, beta, (float *)C, ldc);
  } else if (d_type == ml::train::TensorDim::DataType::FP16) {
    sgemm_FP16(order, TransA, TransB, M, N, K, alpha, (__fp16 *)A, lda,
               (__fp16 *)B, ldb, beta, (__fp16 *)C, ldc);
  }
#endif
}

void sgemm(CBLAS_ORDER order, CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB,
           const unsigned int M, const unsigned int N, const unsigned int K,
           const float alpha, const float *A, const unsigned int lda,
           const float *B, const unsigned int ldb, const float beta, float *C,
           const unsigned int ldc) {

#ifdef USE_CUBLAS
  int devID = 0;
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, devID);
  float *d_A, *d_B, *d_C;

  unsigned int size_A = M * K * sizeof(float);
  unsigned int size_B = K * N * sizeof(float);
  unsigned int size_C = M * N * sizeof(float);

  cudaMalloc((void **)&d_A, size_A);
  cudaMalloc((void **)&d_B, size_B);
  cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice);
  cudaMalloc((void **)&d_C, size_C);

  cublasHandle_t handle;
  cublasCreate(&handle);

  cublasOperation_t transA = (TransA == CblasTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t transB = (TransB == CblasTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasSgemm(handle, transA, transB, N, M, K, &alpha, d_B, N, d_A, K, &beta,
              d_C, N);

  cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost);
  cublasDestroy(handle);
#elif defined USE_BLAS
#ifdef BLAS_NUM_THREADS
  openblas_set_num_threads(BLAS_NUM_THREADS);
#endif
  cblas_sgemm(order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C,
              ldc);
#else
  sgemm_raw(order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C,
            ldc);
#endif
}

void sgemm(CBLAS_ORDER order, CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB,
           const unsigned int M, const unsigned int N, const unsigned int K,
           const float alpha, const __fp16 *A, const unsigned int lda,
           const __fp16 *B, const unsigned int ldb, const float beta, __fp16 *C,
           const unsigned int ldc) {
  sgemm_FP16(order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C,
             ldc);
}

void scopy(const unsigned int N, const void *X, const int incX, void *Y,
           const int incY, ml::train::TensorDim::DataType d_type) {
#ifdef USE_BLAS
#ifdef BLAS_NUM_THREADS
  openblas_set_num_threads(BLAS_NUM_THREADS);
#endif
  if (d_type == ml::train::TensorDim::DataType::FP32) {
    cblas_scopy(N, (float *)X, incX, (float *)Y, incY);
  }
#else
  if (d_type == ml::train::TensorDim::DataType::FP32) {
    scopy_raw(N, (float *)X, incX, (float *)Y, incY);
  } else if (d_type == ml::train::TensorDim::DataType::FP16) {
    scopy_FP16(N, (__fp16 *)X, incX, (__fp16 *)Y, incY);
  }
#endif
} // namespace nntrainer

void scopy(const unsigned int N, const float *X, const int incX, float *Y,
           const int incY) {
#ifdef USE_BLAS
#ifdef BLAS_NUM_THREADS
  openblas_set_num_threads(BLAS_NUM_THREADS);
#endif
  cblas_scopy(N, X, incX, Y, incY);
#else
  scopy_raw(N, X, incX, Y, incY);
#endif
} // namespace nntrainer

void scopy(const unsigned int N, const __fp16 *X, const int incX, __fp16 *Y,
           const int incY) {
  scopy_FP16(N, X, incX, Y, incY);

} // namespace nntrainer

float snrm2(const int N, const float *X, const int incX) {
#ifdef USE_BLAS
#ifdef BLAS_NUM_THREADS
  openblas_set_num_threads(BLAS_NUM_THREADS);
#endif
  return cblas_snrm2(N, X, incX);
#else
  return snrm2_raw(N, X, incX);
#endif
}

__fp16 snrm2(const int N, const __fp16 *X, const int incX) {
  return snrm2_FP16(N, X, incX);
}

float sdot(const unsigned int N, const float *X, const unsigned int incX,
           const float *Y, const unsigned int incY) {
#ifdef USE_BLAS
#ifdef BLAS_NUM_THREADS
  openblas_set_num_threads(BLAS_NUM_THREADS);
#endif
  return cblas_sdot(N, X, incX, Y, incY);
#else
  return sdot_raw(N, X, incX, Y, incY);
#endif
}

__fp16 sdot(const unsigned int N, const __fp16 *X, const unsigned int incX,
            const __fp16 *Y, const unsigned int incY) {
  return sdot_FP16(N, X, incX, Y, incY);
}

void sgemv(CBLAS_ORDER order, CBLAS_TRANSPOSE TransA, const unsigned int M,
           const unsigned int N, const float alpha, const void *A,
           const unsigned int lda, const void *X, const int incX,
           const float beta, void *Y, const int incY,
           ml::train::TensorDim::DataType d_type) {
#ifdef USE_BLAS
#ifdef BLAS_NUM_THREADS
  openblas_set_num_threads(BLAS_NUM_THREADS);
#endif
  return cblas_sgemv(order, TransA, M, N, alpha, A, lda, X, incX, beta, Y,
                     incY);
#else
  if (d_type == ml::train::TensorDim::DataType::FP32) {
    return sgemv_raw(order, TransA, M, N, alpha, (float *)A, lda, (float *)X,
                     incX, beta, (float *)Y, incY);
  } else if (d_type == ml::train::TensorDim::DataType::FP16) {
    return sgemv_FP16(order, TransA, M, N, alpha, (__fp16 *)A, lda, (__fp16 *)X,
                      incX, beta, (__fp16 *)Y, incY);
  }
#endif
}

void sgemv(CBLAS_ORDER order, CBLAS_TRANSPOSE TransA, const unsigned int M,
           const unsigned int N, const float alpha, const float *A,
           const unsigned int lda, const float *X, const int incX,
           const float beta, float *Y, const int incY) {
#ifdef USE_BLAS
#ifdef BLAS_NUM_THREADS
  openblas_set_num_threads(BLAS_NUM_THREADS);
#endif
  return cblas_sgemv(order, TransA, M, N, alpha, A, lda, X, incX, beta, Y,
                     incY);
#else
  return sgemv_raw(order, TransA, M, N, alpha, A, lda, X, incX, beta, Y, incY);
#endif
}

void sgemv(CBLAS_ORDER order, CBLAS_TRANSPOSE TransA, const unsigned int M,
           const unsigned int N, const float alpha, const __fp16 *A,
           const unsigned int lda, const __fp16 *X, const int incX,
           const float beta, __fp16 *Y, const int incY) {
  sgemv_FP16(order, TransA, M, N, alpha, A, lda, X, incX, beta, Y, incY);
}

unsigned int isamax(const unsigned int N, const float *X, const int incX) {
#ifdef USE_BLAS
#ifdef BLAS_NUM_THREADS
  openblas_set_num_threads(BLAS_NUM_THREADS);
#endif
  return cblas_isamax(N, X, incX);
#else
  return isamax_raw(N, X, incX);
#endif
}

unsigned int isamax(const unsigned int N, const __fp16 *X, const int incX) {
  /// @todo isamax_FP16 for BLAS_NUM_THREADS
  return isamax_FP16(N, X, incX);
}

} // namespace nntrainer
