// SPDX-License-Identifier: Apache-2.0
/**
 * @file	q4_0_utils.h
 * @date	15 October 2025
 * @brief	This is Q4_0Utils class for utils for Q4_0 quantization format.
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Maciej Nalewaj <m.nalewaj@samsung.com>
 * @bug		No known bugs
 */

#ifndef __NNTRAINER_Q4_0_UTILS_H__
#define __NNTRAINER_Q4_0_UTILS_H__

#include <algorithm>
#include <cstdint>
#include <vector>

#include "nntr_ggml_impl_common.h"

namespace nntrainer {

/**
 * @class Q4_0Utils class
 * @brief Q4_0Utils class with helpers for Q4_0 format calculation, quantization
 * and dequantization methods.
 */
class Q4_0Utils {
public:
  /**
   * @brief     Unpack one Q4_0x8 block to 8 Q4_0 blocks
   * @param[in] in block_q4_0x8* input Q4_0x8 block
   * @param[out] dst block_q4_0* output vector of 8 Q4_0 blocks
   */
  static void unpackOneBlockQ4_0x8(const block_q4_0x8 *in, block_q4_0 *dst);

  /**
   * @brief     Unpack Q4_0x8 blocks data to Q4_0 format
   * @param[in] src block_q4_0x8 * input data in Q4_0x8 blocks format
   * @param[in] data_size number of Q4_0x8 blocks * sizeof(block_q4_0x8)
   * @param[in] nrow number of rows
   * @param[in] K number of columns
   * @param[out] dst block_q4_0 * output data in Q4_0 blocks format
   */
  static void unpackBlocksQ4_0x8(const block_q4_0x8 *__restrict src,
                                 size_t data_size, size_t nrow, size_t K,
                                 block_q4_0 *__restrict dst);

  /**
   * @brief     Dequantize weights in block_q4_0x8 format to matrix of floats
   * @param[in] q4_weight_repacked void * input data in format block_q4_0x8
   * @param[in] N number of rows
   * @param[in] K number of columns
   * @param[out] dequantized_weights float * dequantized weights matrix
   */
  static void dequantizeQ4_0x8(const void *q4_weight_repacked, int N, int K,
                               float *dequantized_weights);

  /**
   * @brief Transforms data from in-memory layout osv32_isv2 to block_q4_0x8
   * in-memory layout.
   * @param N number of rows
   * @param K number of columns
   * @param osv32_weights uint8_t* data of weights in osv32_isv2 layout
   * @param osv32_scales fp16* scales
   * @param scale_group_size group size (32 or 64 or 128)
   * @param dst_q4_0x8 void * output data in block_q4_0x8 layout
   */
  static void transformQ4_0x8FromInt4(size_t N, size_t K,
                                      const uint8_t *osv32_weights,
                                      const uint16_t *osv32_scales,
                                      size_t scale_group_size,
                                      void *dst_q4_0x8);

  /**
   * @brief     Create a Q4_0 quantization block from int4 weights and scale
   * @param[in] int4_weight Pointer to the input 4-bit quantized weights array.
   * The array should contain 16 bytes representing 32 4-bit values. Each byte
   * contains two 4-bit quantized values packed together.
   * @param[in] scale Half-precision floating point scale factor (FP16) used for
   * dequantization.
   * @param[out] block Pointer to the output block_q4_0 structure that will be
   * populated.
   * @note      The input int4_weight array should contain exactly 32 4-bit
   * values (16 bytes) to match the QK4_0 block size (32 elements per block).
   */
  static void transformQ4_0Block(const uint8_t *int4_weight, uint16_t scale,
                                 block_q4_0 *block);

  /**
   * @brief     Print the Q4_0 block data
   * @param[in] block Pointer to the Q4_0 block
   */
  static void printBlockQ4_0(const block_q4_0 *block);
};
} // namespace nntrainer

#endif // __NNTRAINER_INT4_UTILS_H__
