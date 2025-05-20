// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Michal Wlasiuk <testmailsmtp12345@gmail.com>
 * Copyright (C) 2025 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file   ggml_interface.cpp
 * @date   15 April 2025
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Michal Wlasiuk <testmailsmtp12345@gmail.com>
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Function interface to use ggml lib from cpu_backend
 */

#include "ggml-cpu-quants.h"
#include "ggml-quants.h"
#include <ggml.h>
#include "ggml-common.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <ggml_interface.h>
#include <string>
#include <vector>
#include <thread>

namespace nntrainer {

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

void __ggml_q4_0_8x8_q8_0_GEMM(const unsigned int M, const unsigned int N,
                               const unsigned int K, const float *A,
                               const unsigned int lda, const void *B,
                               const unsigned int ldb, float *C,
                               const unsigned int ldc) {
  if (M == 1) { // GEMV
    int blocks_per_row = (K + QK8_0 - 1) / QK8_0;
    int qa_size = sizeof(block_q8_0) * blocks_per_row;
    std::vector<char> QA = std::vector<char>(qa_size);
    ::quantize_row_q8_0(A, QA.data(), K);

    ::ggml_gemv_q4_0_8x8_q8_0(K, C, ldc, B, QA.data(), M, N);
  } else { // GEMM
    int blocks_per_4_rows = (K + QK8_0 - 1) / QK8_0;
    int qa_4_rows_size = sizeof(block_q8_0x4) * blocks_per_4_rows;
    int M4 = ((M + 3) / 4);

    int qa_size = qa_4_rows_size * M4;
    std::vector<char> QA = std::vector<char>(qa_size);

    // Quantization of activations
// #pragma omp parallel for collapse(1) num_threads(16) // This makes model latency even slower.
    for (int i = 0; i < M4; i++) {
      ::ggml_quantize_mat_q8_0_4x8(A + 4 * i * K,
                                   QA.data() + i * qa_4_rows_size, K);
    }

#if 0
    // single thread
    ::ggml_gemm_q4_0_8x8_q8_0(K, C, ldc, B, QA.data(), M, N);
#else
    int delta = 8;
    int step_N = N / delta;
    int step_C = delta;
    int step_B = blocks_per_4_rows * sizeof(block_q4_0) * delta;
#pragma omp parallel for collapse(1) num_threads(16)
    for (int i = 0; i < step_N; i++) {
      ::ggml_gemm_q4_0_8x8_q8_0(K, C + i * step_C, ldc, (char *)B + i * step_B,
                                QA.data(), M, delta);
    }
#endif
  }
}

void __ggml_q4_K_8x8_q8_K_GEMM(const unsigned int M, const unsigned int N,
                               const unsigned int K, const float *A,
                               const unsigned int lda, const void *B,
                               const unsigned int ldb, float *C,
                               const unsigned int ldc) {
  if (M == 1) { // GEMV
    int blocks_per_row = (K + QK_K - 1) / QK_K;
    int qa_size = sizeof(block_q8_K) * blocks_per_row;
    std::vector<char> QA = std::vector<char>(qa_size);

    ::quantize_row_q8_K(A, QA.data(), K);

    ::ggml_gemv_q4_K_8x8_q8_K(K, C, ldc, B, QA.data(), M, N);
  } else { // GEMM
    unsigned int blocks_per_4_rows = (K + QK_K - 1) / QK_K;
    unsigned int qa_4_rows_size = sizeof(block_q8_Kx4) * blocks_per_4_rows;
    unsigned int M4 = ((M + 3) / 4);
    unsigned int B_step = sizeof(block_q4_K) * (K / QK_K);
    unsigned int thread_num = std::thread::hardware_concurrency();

    unsigned int qa_size = qa_4_rows_size * M4;
    std::vector<char> QA = std::vector<char>(qa_size);

    // Quantization of activations
/// @note Heuristic inspection conducted that applying multithreading on run-time quantization hurts model latency 
// #pragma omp parallel for collapse(1) num_threads(16)
    for (unsigned int i = 0; i < M4; i++) {
      ::ggml_quantize_mat_q8_K_4x8(A + 4 * i * K,
                                   QA.data() + i * qa_4_rows_size, K);
    }

#pragma omp parallel for collapse(1) num_threads(thread_num)
    for (unsigned int i = 0; i < thread_num; i++) {
    unsigned int src0_start = (i * N) / thread_num;
    unsigned int src0_end = ((i + 1) * N) / thread_num;

      src0_start =
        (src0_start % 8) ? src0_start + 8 - (src0_start % 8) : src0_start;
      src0_end = (src0_end % 8) ? src0_end + 8 - (src0_end % 8) : src0_end;

      ::ggml_gemm_q4_K_8x8_q8_K(K, (float*)(C + src0_start), ldc, (void *)((char *)B + src0_start * B_step),
                                QA.data(), M, src0_end - src0_start);
    }
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
