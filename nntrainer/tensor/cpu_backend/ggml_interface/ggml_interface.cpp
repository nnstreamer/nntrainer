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

#include <ggml.h>
#include <ggml_interface.h>
#include "ggml-cpu-quants.h"

#include "ggml_cpu_impl.h"
#include <stdint.h>
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-common.h"
#include <stdio.h>
#include <vector>
#include <string>
#include <iostream>

// it is taken from ./nntrainer-mw/subprojects/ggml/src/ggml-cpu/ggml-cpu-aarch64.cpp

#define QK_K 256
struct block_q8_Kx4 {
  float d[4];              // delta
  int8_t qs[QK_K * 4];     // quants
  int16_t bsums[QK_K / 4]; // sum of quants in groups of 16
};





namespace nntrainer {

size_t nntr_quantize_q4_K(const float *src, void *dst, int64_t nrow,
                          int64_t n_per_row, const float *quant_weights) {
  return ggml_quantize_q4_K(src, dst, nrow, n_per_row, quant_weights);
}




template <typename T>
static inline void print_matrix(const std::string &name, const T* src, int M, int N, int partial_m = 5, int partial_n = 5){
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

void set_value(float *data, int Y, int X, int offY, int offX, float value) {
    data[X * offY + offX] = value;
}


void nntr_q4_K_8x8_q8_K_GEMM(const unsigned int M, const unsigned int N,
                             const unsigned int K, const float *A,
                             const unsigned int lda, const void *B,
                             const unsigned int ldb, float *C,
                             const unsigned int ldc) {
  printf("++++++++++++++++++ nntr_q4_K_8x8_q8_K_GEMM(M:%u, N:%u, K:%u, A:%p, lda:%u, B:%p, ldb:%u, C:%p, ldc:%u)\n", M, N, K, A, lda, B, ldb, C, ldc);

  if (M == 1) {
    // GEMV implementation
    int blocks_per_row = (K + QK_K - 1) / QK_K;
    int qa_size = sizeof(block_q8_K) * blocks_per_row;
    std::vector<char> QA = std::vector<char>(qa_size);

    quantize_row_q8_K(A, QA.data(), K);

    ggml_gemv_q4_K_8x8_q8_K(K, C, ldc, B, QA.data(), M, N);
  } else {
    // GEMM implementation
#if 0
    // Test2: multiopication on floats
    static size_t buf_size = 100000000;
    static std::vector<uint8_t> buf(buf_size);

    ggml_init_params init_params;
    init_params.mem_size = buf_size;
    init_params.mem_buffer = buf.data();
    init_params.no_alloc = true;

    ggml_context *ctx = ggml_init(init_params);

    ggml_tensor *aa = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, M, K, 1);
    aa->data = (float*)A;
    //print_matrix<float>("aa", (float*)aa->data, 16, 16);

    ggml_tensor *bb = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, N, 1);
    bb->data = (float*)B;
    //print_matrix<float>("bb", (float*)bb->data, 16, 16);

    ggml_tensor *cc = ggml_mul_mat(ctx, aa, bb);
    cc->data = (float*)C;
    ggml_cgraph *cgraph = ggml_new_graph(ctx);
    ggml_build_forward_expand(cgraph, cc);

    ggml_status status = ggml_graph_compute_with_ctx(ctx, cgraph, 1);
    if (status != GGML_STATUS_SUCCESS) {
        printf("ERROR: ggml_graph_compute_with_ctx failed!!!\n");
    }

    //print_matrix<float>("cc", (float*)cc->data, 16, 16, 16, 16);

#elif 0
    // Test3: Quantized - to dzialalo ale wolno
    static size_t buf_size = 100000000;
    static std::vector<uint8_t> buf(buf_size);

    ggml_init_params init_params;
    init_params.mem_size = buf_size;
    init_params.mem_buffer = buf.data();
    init_params.no_alloc = true;

    ggml_context *ctx = ggml_init(init_params);

    ggml_tensor *aa = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, M, 1);
    aa->data = (float*)A;
    //print_matrix<float>("aa", (float*)aa->data, 16, 16);
    printf("ggml_nelements(aa): %li\n", ggml_nelements(aa));
    printf("ggml_element_size(aa): %li\n", ggml_element_size(aa));

    ggml_tensor *bb = ggml_new_tensor_3d(ctx, GGML_TYPE_Q4_K, K, N, 1);
    bb->data = (float*)B;
    //print_matrix<float>("bb", (float*)bb->data, 16, 16);
    printf("ggml_nelements(bb): %li\n", ggml_nelements(bb));
    printf("ggml_element_size(bb): %li\n", ggml_element_size(bb));



    //ggml_tensor *cc = ggml_mul_mat(ctx, aa, bb);
    ggml_tensor *cc = ggml_mul_mat(ctx, bb, aa);
    cc->data = (float*)C;
    ggml_cgraph *cgraph = ggml_new_graph(ctx);
    ggml_build_forward_expand(cgraph, cc);

    ggml_status status = ggml_graph_compute_with_ctx(ctx, cgraph, 1);
    if (status != GGML_STATUS_SUCCESS) {
        printf("ERROR: ggml_graph_compute_with_ctx failed!!!\n");
    }

#elif 1
  
    printf("sizeof(block_q8_Kx4):%li\n", sizeof(block_q8_Kx4));

    int blocks_per_4_rows = (K + QK_K - 1) / QK_K;
    int qa_4_rows_size = sizeof(block_q8_Kx4) * blocks_per_4_rows;
    int M4 = ((M + 3) / 4);

    int qa_size = qa_4_rows_size * M4;
    std::vector<char> QA = std::vector<char>(qa_size);

    // Quantization of activations
    for (int i = 0; i < M4; i++) {
      ggml_quantize_mat_q8_K_4x8(A + 4 * i * K, QA.data() + i * qa_4_rows_size, K);
    }

#if 1
    // single thread
    ggml_gemm_q4_K_8x8_q8_K(K, C, ldc, B, QA.data(), M, N);

#else
    // TODO beter multithreading
    int delta = 384 / 4;
    int step_N = N / delta;
    int step_C = delta;
    int step_B = blocks_per_4_rows * 144 * delta;
    #pragma omp parallel for num_threads(16)
    for (int i = 0; i < step_N; i++) {
      ggml_gemm_q4_K_8x8_q8_K(K, C + i * step_C, ldc, B + i * step_B, QA.data(), M, delta);
    }
#endif

#else
    // Old solution - it works but slow
    ggml_q4_K_8x8_q8_K_GEMM(M, N, K, A, lda, B, ldb, C, ldc);
#endif
  }
  printf("------------------ nntr_q4_K_8x8_q8_K_GEMM()\n");
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

}
