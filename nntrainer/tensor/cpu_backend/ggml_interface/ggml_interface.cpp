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
#include <cstdint>
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

#define NUM_THREADS_TOTAL static_cast<int32_t>(std::thread::hardware_concurrency())
#define NUM_THREADS_TOTAL_MINUS_ONE static_cast<int32_t>(NUM_THREADS_TOTAL - 1)

#define DEFAULT 1
#define OMP 2
#define BS_DETATCH_TASK 3

#define METHOD BS_DETATCH_TASK

static BS::thread_pool g_thread_pool(NUM_THREADS_TOTAL_MINUS_ONE);

namespace nntrainer {

#define QK_K 256
struct block_q8_Kx4 {
  float d[4];              // delta
  int8_t qs[QK_K * 4];     // quants
  int16_t bsums[QK_K / 4]; // sum of quants in groups of 16
};

size_t __ggml_quantize_q4_K(const float *src, void *dst, int64_t nrow,
                            int64_t n_per_row, const float *quant_weights) {
  return ::quantize_q4_K(src, dst, nrow, n_per_row, quant_weights);
}

void __ggml_q4_K_8x8_q8_K_GEMM_perform_GEMV(
  const unsigned int M, const unsigned int N, const unsigned int K,
  const float *A, const unsigned int lda, const void *B, const unsigned int ldb,
  float *C, const unsigned int ldc) {
  const int32_t blocks_per_row = (K + QK_K - 1) / QK_K;
  const int32_t qa_size = sizeof(block_q8_K) * blocks_per_row;
  std::vector<char> QA = std::vector<char>(qa_size);

  const auto qa_data = QA.data();

  ::quantize_row_q8_K(A, qa_data, K);
  ::ggml_gemv_q4_K_8x8_q8_K(K, C, ldc, B, qa_data, M, N);
}

void __ggml_q4_K_8x8_q8_K_GEMM_perform_GEMM(
  const unsigned int M, const unsigned int N, const unsigned int K,
  const float *A, const unsigned int lda, const void *B, const unsigned int ldb,
  float *C, const unsigned int ldc) {

  const int32_t blocks_per_4_rows = (K + QK_K - 1) / QK_K;
  const int32_t qa_4_rows_size = sizeof(block_q8_Kx4) * blocks_per_4_rows;
  const int32_t M4 = ((M + 3) / 4);
  const int32_t qa_size = qa_4_rows_size * M4;

  std::vector<char> QA = std::vector<char>(qa_size);

  char *const qa_data = QA.data();

  const int32_t delta = 8;
  const int32_t step_N = N / delta;
  const int32_t step_C = delta;
  const int32_t step_B = blocks_per_4_rows * sizeof(block_q4_K) * delta;

#if METHOD == DEFAULT
#pragma omp parallel for collapse(1) num_threads(NUM_THREADS_TOTAL)
  for (int i = 0; i < M4; i++) {
    ::ggml_quantize_mat_q8_K_4x8(A + 4 * i * K, qa_data + i * qa_4_rows_size,
                                 K);
  }

  ggml_gemm_q4_K_8x8_q8_K(K, C, ldc, B, qa_data, M, N);
#endif

#if METHOD == OMP
#pragma omp parallel for collapse(1) num_threads(NUM_THREADS_TOTAL)
  for (int i = 0; i < M4; i++) {
    ::ggml_quantize_mat_q8_K_4x8(A + 4 * i * K, qa_data + i * qa_4_rows_size,
                                 K);
  }

#pragma omp parallel for collapse(1) num_threads(NUM_THREADS_TOTAL)
  for (int i = 0; i < step_N; i++) {
    ::ggml_gemm_q4_K_8x8_q8_K(K, C + i * step_C, ldc, (char *)B + i * step_B,
                              qa_data, M, delta);
  }
#endif

#if METHOD == BS_DETATCH_TASK
  auto quantize_task_size = M4 / NUM_THREADS_TOTAL;
  auto gemm_task_size = step_N / NUM_THREADS_TOTAL;

  // printf("step_N = %d / %d = %d\n", step_N, NUM_THREADS_TOTAL,
  // gemm_task_size);

  for (auto n = 0; n < NUM_THREADS_TOTAL_MINUS_ONE; n++) {
    g_thread_pool.detach_task([=] {
      auto start = (n)*quantize_task_size;
      auto end = (n + 1) * quantize_task_size;

      for (auto i = start; i < end; ++i) {
        ::ggml_quantize_mat_q8_K_4x8(A + 4 * i * K,
                                     qa_data + i * qa_4_rows_size, K);
      }
    });
  }

  for (auto n = NUM_THREADS_TOTAL_MINUS_ONE; n < NUM_THREADS_TOTAL; n++) {
    auto start = (n)*quantize_task_size;
    auto end = (n + 1) * quantize_task_size;

    for (auto i = start; i < end; ++i) {
      ::ggml_quantize_mat_q8_K_4x8(A + 4 * i * K, qa_data + i * qa_4_rows_size,
                                   K);
    }
  }

  g_thread_pool.wait();

  for (auto n = 0; n < NUM_THREADS_TOTAL_MINUS_ONE; n++) {
    g_thread_pool.detach_task([=] {
      auto start = (n)*gemm_task_size;
      auto end = (n + 1) * gemm_task_size;

      for (auto i = start; i < end; ++i) {
        ::ggml_gemm_q4_K_8x8_q8_K(K, C + i * step_C, ldc,
                                  (char *)B + i * step_B, qa_data, M, delta);
      }
    });
  }

  for (auto n = NUM_THREADS_TOTAL_MINUS_ONE; n < NUM_THREADS_TOTAL; n++) {
    auto start = (n)*gemm_task_size;
    auto end = (n + 1) * gemm_task_size;

    for (auto i = start; i < end; ++i) {
      ::ggml_gemm_q4_K_8x8_q8_K(K, C + i * step_C, ldc, (char *)B + i * step_B,
                                qa_data, M, delta);
    }
  }

  g_thread_pool.wait();
#endif
}

void __ggml_q4_K_8x8_q8_K_GEMM(const unsigned int M, const unsigned int N,
                               const unsigned int K, const float *A,
                               const unsigned int lda, const void *B,
                               const unsigned int ldb, float *C,
                               const unsigned int ldc) {
  if (M == 1) {
    __ggml_q4_K_8x8_q8_K_GEMM_perform_GEMV(M, N, K, A, lda, B, ldb, C, ldc);
  } else {
    __ggml_q4_K_8x8_q8_K_GEMM_perform_GEMM(M, N, K, A, lda, B, ldb, C, ldc);
  }
}

void __ggml_dequantize_row_q4_K(const void *x_raw, float *y, int64_t k) {
  ::dequantize_row_q4_K((const block_q4_K *)x_raw, y, k);
}

void __ggml_dequantize_row_q8_K(const void *x, float *y, int64_t k) {
  ::dequantize_row_q8_K((const block_q8_K *)x, y, k);
}

void __ggml_repack_q4_K_to_q8_K(void *W, void *repacked_W, size_t data_size,
                                const unsigned int M, const unsigned int N) {
  ::ggml_repack_q4_K_to_q4_K_8_bl(W, 8, repacked_W, data_size, M, N);
}

} // namespace nntrainer
