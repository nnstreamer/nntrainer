// SPDX-License-Identifier: Apache-2.0
/**
 * @file	q4_0_utils.h
 * @date	15 October 2025
 * @brief	This is Q4_0Utils class for some utils for Q4_0 quantization format.
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
};

} // namespace nntrainer

#endif // __NNTRAINER_INT4_UTILS_H__
