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