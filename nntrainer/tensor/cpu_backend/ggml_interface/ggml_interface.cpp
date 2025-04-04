// SPDX-License-Identifier: Apache-2.0
/**
 * @file	ggml_interface.cpp
 * @date	03 April 2025
 * @brief	This is ggml interface
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Sungsik Kong <ss.kong@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#include "ggml_cpu_impl.h"
#include <stdint.h>

size_t nntr_quantize_q4_K(const float *src, void *dst, int64_t nrow,
                          int64_t n_per_row, const float *quant_weights) {
  return ggml_quantize_q4_K(src, dst, nrow, n_per_row, quant_weights);
}

void nntr_q4_K_8x8_q8_K_GEMM(const unsigned int M, const unsigned int N,
                             const unsigned int K, const float *A,
                             const unsigned int lda, const void *B,
                             const unsigned int ldb, float *C,
                             const unsigned int ldc) {
  ggml_q4_K_8x8_q8_K_GEMM(M, N, K, A, lda, B, ldb, C, ldc);
}

void nntr_dequantize_row_q4_K(const void * x_raw, float * y, int64_t k){
  ggml_dequantize_row_q4_K(x_raw, y, k);
}
void nntr_dequantize_row_q8_K(const void * x, float * y, int64_t k){
  ggml_dequantize_row_q8_K(x, y, k);
}

void nntr_repack_q4_K_to_q8_K(void* W, void* repacked_W, size_t data_size, const unsigned int M, const unsigned int N){
  ggml_repack_q4_K_to_q8_K(W, repacked_W, data_size, M, N);
}
