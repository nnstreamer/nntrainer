// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Michal Wlasiuk <testmailsmtp12345@gmail.com>
 *
 * @file   ggml_interface.h
 * @date   15 April 2025
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Michal Wlasiuk <testmailsmtp12345@gmail.com>
 * @bug    No known bugs except for NYI items
 * @brief  Function interface to use ggml lib from cpu_backend
 */

#include "ggml-cpu-quants.h"
#include "ggml-quants.h"
#include <ggml.h>
#include <ggml_interface.h>

#include "ggml-common.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include <iostream>
#include <stdint.h>
#include <stdio.h>
#include <string>
#include <vector>

#include <bs_thread_pool.hpp>

#define MULTITHREADING_METHOD_NONE 0
#define MULTITHREADING_METHOD_OMP 1
#define MULTITHREADING_METHOD_BSTP 2

#define MULTITHREADING_METHOD MULTITHREADING_METHOD_BSTP

#define MULTITHREADING_METHOD_THREADS_TOTAL static_cast<int32_t>(8)
#define MULTITHREADING_METHOD_THREADS_TOTAL_MINUS_ONE                          \
  static_cast<int32_t>(MULTITHREADING_METHOD_THREADS_TOTAL - 1)

static BS::thread_pool
  g_thread_pool(MULTITHREADING_METHOD_THREADS_TOTAL_MINUS_ONE);

namespace nntrainer {

template <typename T>
static inline void
print_matrix_partially_n(const std::string &name, const T *src, int M, int N,
                         int partial_m = 5, int partial_n = 5) {
  std::cout << name << ":" << std::endl;
  std::cout << "--------------------------" << std::endl;
  for (int i = 0; i < partial_m; ++i) {
    for (int j = 0; j < partial_n; ++j) {
      std::cout << src[i * N + j] << "  ";
    }
    std::cout << std::endl;
  }
  std::cout << "--------------------------" << std::endl;
}

/**
 * @brief
 */
struct block_q8_Kx4 {
  float d[4];              // delta
  int8_t qs[QK_K * 4];     // quants
  int16_t bsums[QK_K / 4]; // sum of quants in groups of 16
};

template <int K> constexpr int QK_0() {
  if constexpr (K == 4) {
    return QK4_0;
  }
  if constexpr (K == 8) {
    return QK8_0;
  }
  return -1;
}

/**
 * @brief
 *
 * @tparam K
 * @tparam N
 */
template <int K, int N> struct block {
  ggml_half d[N];                     // deltas for N qK_0 blocks
  int8_t qs[(QK_0<K>() * N * K) / 8]; // quants for N qK_0 blocks
};

using block_q4_0x4 = block<4, 4>;
using block_q8_0x4 = block<8, 4>;

void __ggml_init() {
  // needed to initialize f16 tables
  struct ggml_init_params params = {0, NULL, false};
  struct ggml_context *ctx = ggml_init(params);
  ggml_free(ctx);
}

size_t __ggml_quantize_q4_0(const float *src, void *dst, int64_t nrow,
                            int64_t n_per_row, const float *quant_weights) {
  return ::quantize_q4_0(src, dst, nrow, n_per_row, quant_weights);
}

size_t __ggml_quantize_q4_K(const float *src, void *dst, int64_t nrow,
                            int64_t n_per_row, const float *quant_weights) {
  return ::quantize_q4_K(src, dst, nrow, n_per_row, quant_weights);
}

static inline void __ggml_q4_0_8x8_q8_0_GEMM_GEMV(
  const unsigned int M, const unsigned int N, const unsigned int K,
  const float *A, const unsigned int lda, const void *B, const unsigned int ldb,
  float *C, const unsigned int ldc) {
  int blocks_per_row = (K + QK8_0 - 1) / QK8_0;
  int qa_size = sizeof(block_q8_0) * blocks_per_row;
  std::vector<char> QA = std::vector<char>(qa_size);

  auto qa_data = QA.data();

  ::quantize_row_q8_0(A, qa_data, K);
  ::ggml_gemv_q4_0_8x8_q8_0(K, C, ldc, B, qa_data, M, N);
}

static inline void __ggml_q4_0_8x8_q8_0_GEMM_GEMM(
  const unsigned int M, const unsigned int N, const unsigned int K,
  const float *A, const unsigned int lda, const void *B, const unsigned int ldb,
  float *C, const unsigned int ldc) {
  int blocks_per_4_rows = (K + QK8_0 - 1) / QK8_0;
  int qa_4_rows_size = sizeof(block_q8_0x4) * blocks_per_4_rows;
  int M4 = ((M + 3) / 4);

  int delta = 8;
  // int delta = 384 / 4;
  int step_N = N / delta;
  int step_C = delta;
  int step_B = blocks_per_4_rows * sizeof(block_q4_0) * delta;

  int qa_size = qa_4_rows_size * M4;
  std::vector<char> QA = std::vector<char>(qa_size);

  auto qa_data = QA.data();

#if MULTITHREADING_METHOD == MULTITHREADING_METHOD_NONE
  for (int i = 0; i < M4; i++) {
    ::ggml_quantize_mat_q8_0_4x8(A + 4 * i * K, qa_data + i * qa_4_rows_size,
                                 K);
  }
  ::ggml_gemm_q4_0_8x8_q8_0(K, C, ldc, B, qa_data, M, N);
#elif MULTITHREADING_METHOD == MULTITHREADING_METHOD_OMP
#pragma omp parallel for collapse(1) num_threads(16)
  for (int i = 0; i < M4; i++) {
    ::ggml_quantize_mat_q8_0_4x8(A + 4 * i * K, qa_data + i * qa_4_rows_size,
                                 K);
  }

#pragma omp parallel for collapse(1) num_threads(16)
  for (int i = 0; i < step_N; i++) {
    ::ggml_gemm_q4_0_8x8_q8_0(K, C + i * step_C, ldc, (char *)B + i * step_B,
                              qa_data, M, delta);
  }
#elif MULTITHREADING_METHOD == MULTITHREADING_METHOD_BSTP
  auto quantize_task_size = M4 / MULTITHREADING_METHOD_THREADS_TOTAL;
  auto gemm_task_size = step_N / MULTITHREADING_METHOD_THREADS_TOTAL;

  for (auto n = 0; n < MULTITHREADING_METHOD_THREADS_TOTAL_MINUS_ONE; n++) {
    g_thread_pool.detach_task([=] {
      auto start = (n)*quantize_task_size;
      auto end = (n + 1) * quantize_task_size;

      for (auto i = start; i < end; ++i) {
        ::ggml_quantize_mat_q8_0_4x8(A + 4 * i * K,
                                     qa_data + i * qa_4_rows_size, K);
      }
    });
  }

  for (auto n = MULTITHREADING_METHOD_THREADS_TOTAL_MINUS_ONE;
       n < MULTITHREADING_METHOD_THREADS_TOTAL; n++) {
    auto start = (n)*quantize_task_size;
    auto end = (n + 1) * quantize_task_size;

    for (auto i = start; i < end; ++i) {
      ::ggml_quantize_mat_q8_0_4x8(A + 4 * i * K, qa_data + i * qa_4_rows_size,
                                   K);
    }
  }

  g_thread_pool.wait();

  for (auto n = 0; n < MULTITHREADING_METHOD_THREADS_TOTAL_MINUS_ONE; n++) {
    g_thread_pool.detach_task([=] {
      auto start = (n)*gemm_task_size;
      auto end = (n + 1) * gemm_task_size;

      for (auto i = start; i < end; ++i) {
        ::ggml_gemm_q4_0_8x8_q8_0(K, C + i * step_C, ldc,
                                  (char *)B + i * step_B, qa_data, M, delta);
      }
    });
  }

  for (auto n = MULTITHREADING_METHOD_THREADS_TOTAL_MINUS_ONE;
       n < MULTITHREADING_METHOD_THREADS_TOTAL; n++) {
    auto start = (n)*gemm_task_size;
    auto end = (n + 1) * gemm_task_size;

    for (auto i = start; i < end; ++i) {
      ::ggml_gemm_q4_0_8x8_q8_0(K, C + i * step_C, ldc, (char *)B + i * step_B,
                                qa_data, M, delta);
    }
  }

  g_thread_pool.wait();
#else
#error "Multithreading method not defined!"
#endif
}

void __ggml_q4_0_8x8_q8_0_GEMM(const unsigned int M, const unsigned int N,
                               const unsigned int K, const float *A,
                               const unsigned int lda, const void *B,
                               const unsigned int ldb, float *C,
                               const unsigned int ldc) {
  if (M == 1) { // GEMV
    __ggml_q4_0_8x8_q8_0_GEMM_GEMV(M, N, K, A, lda, B, ldb, C, ldc);
  } else { // GEMM
    __ggml_q4_0_8x8_q8_0_GEMM_GEMM(M, N, K, A, lda, B, ldb, C, ldc);
  }
}

static inline void __ggml_q4_K_8x8_q8_K_GEMM_GEMV(
  const unsigned int M, const unsigned int N, const unsigned int K,
  const float *A, const unsigned int lda, const void *B, const unsigned int ldb,
  float *C, const unsigned int ldc) {
  int blocks_per_row = (K + QK_K - 1) / QK_K;
  int qa_size = sizeof(block_q8_K) * blocks_per_row;
  std::vector<char> QA = std::vector<char>(qa_size);

  auto qa_data = QA.data();

  ::quantize_row_q8_K(A, qa_data, K);
  ::ggml_gemv_q4_K_8x8_q8_K(K, C, ldc, B, qa_data, M, N);
}

static inline void __ggml_q4_K_8x8_q8_K_GEMM_GEMM(
  const unsigned int M, const unsigned int N, const unsigned int K,
  const float *A, const unsigned int lda, const void *B, const unsigned int ldb,
  float *C, const unsigned int ldc) {
  int blocks_per_4_rows = (K + QK_K - 1) / QK_K;
  int qa_4_rows_size = sizeof(block_q8_Kx4) * blocks_per_4_rows;
  int M4 = ((M + 3) / 4);

  int delta = 8;
  // int delta = 384 / 4;
  int step_N = N / delta;
  int step_C = delta;
  int step_B = blocks_per_4_rows * sizeof(block_q4_K) * delta;

  int qa_size = qa_4_rows_size * M4;
  std::vector<char> QA = std::vector<char>(qa_size);

  auto qa_data = QA.data();

#if MULTITHREADING_METHOD == MULTITHREADING_METHOD_NONE
  for (int i = 0; i < M4; i++) {
    ::ggml_quantize_mat_q8_K_4x8(A + 4 * i * K, qa_data + i * qa_4_rows_size,
                                 K);
  }
  ggml_gemm_q4_K_8x8_q8_K(K, C, ldc, B, qa_data, M, N);
#elif MULTITHREADING_METHOD == MULTITHREADING_METHOD_OMP
#pragma omp parallel for collapse(1) num_threads(16)
  for (int i = 0; i < M4; i++) {
    ::ggml_quantize_mat_q8_K_4x8(A + 4 * i * K, qa_data + i * qa_4_rows_size,
                                 K);
  }

#pragma omp parallel for collapse(1) num_threads(16)
  for (int i = 0; i < step_N; i++) {
    ::ggml_gemm_q4_K_8x8_q8_K(K, C + i * step_C, ldc, (char *)B + i * step_B,
                              qa_data, M, delta);
  }
#elif MULTITHREADING_METHOD == MULTITHREADING_METHOD_BSTP
  auto quantize_task_size = M4 / MULTITHREADING_METHOD_THREADS_TOTAL;
  auto gemm_task_size = step_N / MULTITHREADING_METHOD_THREADS_TOTAL;

  for (auto n = 0; n < MULTITHREADING_METHOD_THREADS_TOTAL_MINUS_ONE; n++) {
    g_thread_pool.detach_task([=] {
      auto start = (n)*quantize_task_size;
      auto end = (n + 1) * quantize_task_size;

      for (auto i = start; i < end; ++i) {
        ::ggml_quantize_mat_q8_K_4x8(A + 4 * i * K,
                                     qa_data + i * qa_4_rows_size, K);
      }
    });
  }

  for (auto n = MULTITHREADING_METHOD_THREADS_TOTAL_MINUS_ONE;
       n < MULTITHREADING_METHOD_THREADS_TOTAL; n++) {
    auto start = (n)*quantize_task_size;
    auto end = (n + 1) * quantize_task_size;

    for (auto i = start; i < end; ++i) {
      ::ggml_quantize_mat_q8_K_4x8(A + 4 * i * K, qa_data + i * qa_4_rows_size,
                                   K);
    }
  }

  g_thread_pool.wait();

  for (auto n = 0; n < MULTITHREADING_METHOD_THREADS_TOTAL_MINUS_ONE; n++) {
    g_thread_pool.detach_task([=] {
      auto start = (n)*gemm_task_size;
      auto end = (n + 1) * gemm_task_size;

      for (auto i = start; i < end; ++i) {
        ::ggml_gemm_q4_K_8x8_q8_K(K, C + i * step_C, ldc,
                                  (char *)B + i * step_B, qa_data, M, delta);
      }
    });
  }

  for (auto n = MULTITHREADING_METHOD_THREADS_TOTAL_MINUS_ONE;
       n < MULTITHREADING_METHOD_THREADS_TOTAL; n++) {
    auto start = (n)*gemm_task_size;
    auto end = (n + 1) * gemm_task_size;

    for (auto i = start; i < end; ++i) {
      ::ggml_gemm_q4_K_8x8_q8_K(K, C + i * step_C, ldc, (char *)B + i * step_B,
                                qa_data, M, delta);
    }
  }

  g_thread_pool.wait();
#else
#error "Multithreading method not defined!"
#endif
}

void __ggml_q4_K_8x8_q8_K_GEMM(const unsigned int M, const unsigned int N,
                               const unsigned int K, const float *A,
                               const unsigned int lda, const void *B,
                               const unsigned int ldb, float *C,
                               const unsigned int ldc) {
  if (M == 1) { // GEMV
    __ggml_q4_K_8x8_q8_K_GEMM_GEMV(M, N, K, A, lda, B, ldb, C, ldc);
  } else { // GEMM
    __ggml_q4_K_8x8_q8_K_GEMM_GEMM(M, N, K, A, lda, B, ldb, C, ldc);
  }
}

void __ggml_dequantize_row_q4_K(const void *x_raw, float *y, int64_t k) {
  ::dequantize_row_q4_K((const block_q4_K *)x_raw, y, k);
}

void __ggml_dequantize_row_q8_K(const void *x, float *y, int64_t k) {
  ::dequantize_row_q8_K((const block_q8_K *)x, y, k);
}

void __ggml_repack_q4_0_to_q4_0_8(void *W, void *repacked_W, size_t data_size,
                                  const unsigned int M, const unsigned int N) {
  ::ggml_repack_q4_0_to_q4_0_8_bl(W, 8, repacked_W, data_size, M, N);
}

void __ggml_repack_q4_K_to_q4_K_8(void *W, void *repacked_W, size_t data_size,
                                  const unsigned int M, const unsigned int N) {
  ::ggml_repack_q4_K_to_q4_K_8_bl(W, 8, repacked_W, data_size, M, N);
}

} // namespace nntrainer
