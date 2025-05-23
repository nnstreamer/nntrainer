// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Michal Wlasiuk <testmailsmtp12345@gmail.com>
 * Copyright (C) 2025 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file   ggml_interface.h
 * @date   15 April 2025
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Michal Wlasiuk <testmailsmtp12345@gmail.com>
 * @author Sungsik Kong <ss.kong@samsung.com>
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
 * @brief Initialization of ggml backend
 */
void __ggml_init();

/**
 * @brief Quantize float to q4_0 Quantization format
 *
 * @param src input src to be quantized
 * @param dst output destination for quantized data
 * @param nrow number of row
 * @param n_per_row number of elements per row
 * @param quant_weights additional information for quantization. Currently in no
 * use.
 * @return size_t total size of quantized data
 */
size_t __ggml_quantize_q4_0(const float *src, void *dst, int64_t nrow,
                            int64_t n_per_row, const float *quant_weights);

/**
 * @brief Quantize float to q4_K Quantization format
 *
 * @param src input src to be quantized
 * @param dst output destination for quantized data
 * @param nrow number of row
 * @param n_per_row number of elements per row
 * @param quant_weights additional information for quantization. Currently in no
 * use.
 * @return size_t total size of quantized data
 */
size_t __ggml_quantize_q4_K(const float *src, void *dst, int64_t nrow,
                            int64_t n_per_row, const float *quant_weights);

/**
 * @brief Quantize float to q6_K Quantization format
 *
 * @param src
 * @param dst
 * @param k
 */
void __ggml_quantize_row_q6_K(const float *src, void *dst, int64_t k);

/**
 * @brief Quantize float to q6_K Quantization format
 *
 * @param src
 * @param dst
 * @param k
 */
void __ggml_quantize_row_q8_K(const float *src, void *dst, int64_t k);

/**
 * @brief A(M, K) * W.T(N, K) = (M, N)
 *
 * @param M as descripted above
 * @param N as descripted above
 * @param K as descripted above
 * @param A Activation
 * @param lda leading dimension of A
 * @param B offline quantized and packed q4_0x8 Weight
 * @param ldb leading dimension of B
 * @param C dst matrix
 * @param ldc leading dimension of C
 */
void __ggml_q4_0_8x8_q8_0_GEMM(const unsigned int M, const unsigned int N,
                               const unsigned int K, const float *A,
                               const unsigned int lda, const void *B,
                               const unsigned int ldb, float *C,
                               const unsigned int ldc);

/**
 * @brief A(M, K) * W.T(N, K) = (M, N)
 *
 * @param M as descripted above
 * @param N as descripted above
 * @param K as descripted above
 * @param A Activation
 * @param lda leading dimension of A
 * @param B offline quantized and packed q4_kx8 Weight
 * @param ldb leading dimension of B
 * @param C dst matrix
 * @param ldc leading dimension of C
 */
void __ggml_q4_K_8x8_q8_K_GEMM(const unsigned int M, const unsigned int N,
                               const unsigned int K, const float *A,
                               const unsigned int lda, const void *B,
                               const unsigned int ldb, float *C,
                               const unsigned int ldc);

/**
 * @brief
 *
 * @param K
 * @param v_q6_K
 * @param v_q8_K
 * @return float
 */
float __ggml_vec_dot_q6_K_q8_K(const unsigned int K, const void *v_q6_K,
                               const void *v_q8_K);

/**
 * @brief q4K to float dequantize
 *
 * @param x_raw input src to be dequantized
 * @param y output destination for dequantized data
 * @param k data length
 */
void __ggml_dequantize_row_q4_K(const void *x_raw, float *y, int64_t k);

/**
 * @brief dequantize row of q6_K data to float
 *
 * @param x input to be dequantized from q6_K to float
 * @param y dequantized data output
 * @param k number of elements in x
 */
void __ggml_dequantize_row_q6_K(const void *x, float *y, int64_t k);

/**
 * @brief q8K to float dequantize
 *
 * @param x_raw input src to be dequantized
 * @param y output destination for dequantized data
 * @param k data length
 */
void __ggml_dequantize_row_q8_K(const void *x, float *y, int64_t k);

/**
 * @brief repack q40 to q40x8
 *
 * @param W input q40
 * @param repacked_W output q40x8
 * @param data_size total weight size
 * @param M number of rows
 * @param N number of columns
 */
void __ggml_repack_q4_0_to_q4_0_8(void *W, void *repacked_W, size_t data_size,
                                  const unsigned int M, const unsigned int N);

/**
 * @brief repack q4K to q4Kx8
 *
 * @param W input q4K
 * @param repacked_W output q4Kx8
 * @param data_size total weight size
 * @param M number of rows
 * @param N number of columns
 */
void __ggml_repack_q4_K_to_q4_K_8(void *W, void *repacked_W, size_t data_size,
                                  const unsigned int M, const unsigned int N);
} // namespace nntrainer

#endif
#endif
