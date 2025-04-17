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

#ifndef __GGML_INTERFACE_H__
#define __GGML_INTERFACE_H__
#ifdef __cplusplus

#include <stdint.h>
#include <stdlib.h>


namespace nntrainer {

 /**
  * @brief Quantize float to q4_K Quantization format
  * 
  * @param src 
  * @param dst 
  * @param nrow 
  * @param n_per_row 
  * @param quant_weights 
  * @return size_t 
  */
 size_t __ggml_quantize_q4_K(const float * src, void * dst, int64_t nrow, int64_t n_per_row, const float * quant_weights);

/**
 * @brief 
 * 
 * @param M 
 * @param N 
 * @param K 
 * @param A 
 * @param lda 
 * @param B 
 * @param ldb 
 * @param C 
 * @param ldc 
 */
void __ggml_q4_K_8x8_q8_K_GEMM(const unsigned int M, const unsigned int N, const unsigned int K,
    const float *A, const unsigned int lda, const void *B,
    const unsigned int ldb, float *C, const unsigned int ldc);

void __ggml_dequantize_row_q4_K(const void * x_raw, float * y, int64_t k);
void __ggml_dequantize_row_q8_K(const void * x, float * y, int64_t k);

void __ggml_repack_q4_K_to_q8_K(void* W, void* repacked_W, size_t data_size, const unsigned int M, const unsigned int N);

}

#endif
#endif
