#pragma once

#include "ggml-cpu-traits.h"
#include "ggml.h"

// GGML internal header
// template <typename BLOC_TYPE, int64_t INTER_SIZE, int64_t NB_COLS, ggml_type PARAM_TYPE> class nntr_gemm_ggml_traits {
//     bool compute_forward(struct ggml_compute_params * params, struct ggml_tensor * op);
//     void forward_mul_mat(ggml_compute_params * params, ggml_tensor * op);
//     int repack(struct ggml_tensor * t, const void * data, size_t data_size);
// };

void nntr_q4_K_8x8_q8_K_GEMM(const unsigned int M, const unsigned int N, const unsigned int K,
               const float *A, const unsigned int lda, const void *B,
               const unsigned int ldb, float *C, const unsigned int ldc);
