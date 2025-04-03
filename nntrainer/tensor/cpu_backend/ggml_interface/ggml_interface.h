#include <stdint.h>
#include <stdlib.h>

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
size_t nntr_quantize_q4_K(const float * src, void * dst, int64_t nrow, int64_t n_per_row, const float * quant_weights);

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
void nntr_q4_K_8x8_q8_K_GEMM(const unsigned int M, const unsigned int N, const unsigned int K,
               const float *A, const unsigned int lda, const void *B,
               const unsigned int ldb, float *C, const unsigned int ldc);

void nntr_dequantize_row_q4_K(const void * x_raw, float * y, int64_t k);
void nntr_dequantize_row_q8_K(const void * x, float * y, int64_t k);