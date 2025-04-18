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

#include <stdint.h>
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-common.h"
#include <stdio.h>
#include <vector>
#include <string>
#include <iostream>

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
    printf("sizeof(block_q8_Kx4):%li\n", sizeof(block_q8_Kx4));

    int blocks_per_4_rows = (K + QK_K - 1) / QK_K;
    int qa_4_rows_size = sizeof(block_q8_Kx4) * blocks_per_4_rows;
    int M4 = ((M + 3) / 4);

    int qa_size = qa_4_rows_size * M4;
    std::vector<char> QA = std::vector<char>(qa_size);

    // Quantization of activations
#pragma omp parallel for collapse(1) num_threads(16)
    for (int i = 0; i < M4; i++) {
      ::ggml_quantize_mat_q8_K_4x8(A + 4 * i * K, QA.data() + i * qa_4_rows_size, K);
    }

#if 0
    // single thread
    ggml_gemm_q4_K_8x8_q8_K(K, C, ldc, B, QA.data(), M, N);
#else
    // TODO check beter multithreading
    int delta = 8;
    // int delta = 384 / 4;
    int step_N = N / delta;
    int step_C = delta;
    int step_B = blocks_per_4_rows * sizeof(block_q4_K) * delta;
#pragma omp parallel for collapse(1) num_threads(16)
    for (int i = 0; i < step_N; i++) {
      ::ggml_gemm_q4_K_8x8_q8_K(K, C + i * step_C, ldc, (char *)B + i * step_B,
                              QA.data(), M, delta);
    }
#endif
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
