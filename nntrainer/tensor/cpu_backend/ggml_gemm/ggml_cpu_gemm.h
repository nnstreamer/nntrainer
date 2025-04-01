#pragma once

#include "ggml-cpu-traits.h"
#include "ggml.h"

// GGML internal header

void nntr_q4_K_8x8_q8_K_GEMM(const unsigned int M, const unsigned int N, const unsigned int K,
               const float *A, const unsigned int lda, const void *B,
               const unsigned int ldb, float *C, const unsigned int ldc);
