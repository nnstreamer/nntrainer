// SPDX-License-Identifier: Apache-2.0
/**
 * @file	ggml_cpu_impl.h
 * @date	03 April 2025
 * @brief	This is ggml cpu implemenation
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Sungsik Kong <ss.kong@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#include <stddef.h>
#include <stdint.h>

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
void ggml_q4_K_8x8_q8_K_GEMM(const unsigned int M, const unsigned int N, const unsigned int K,
               const float *A, const unsigned int lda, const void *B,
               const unsigned int ldb, float *C, const unsigned int ldc);
/**
 * @brief 
 * 
 * @param M 
 * @param N 
 * @param W 
 * @param repacked_W 
 * @param data_size 
 */
void ggml_repack_q4_K_to_q8_K(void* W, void* repacked_W, size_t data_size, const unsigned int M, const unsigned int N);

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
size_t ggml_quantize_q4_K(const float * src, void * dst, int64_t nrow, int64_t n_per_row, const float * quant_weights);

/**
 * @brief 
 * 
 * @param x_raw 
 * @param y 
 * @param k 
 */
void ggml_dequantize_row_q4_K(const void * x_raw, float * y, int64_t k);

/**
 * @brief 
 * 
 * @param x 
 * @param y 
 * @param k 
 */
void ggml_dequantize_row_q8_K(const void * x, float * y, int64_t k);

