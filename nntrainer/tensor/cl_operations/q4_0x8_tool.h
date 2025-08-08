/**
 * Copyright (C) 2025 Daekyoung Jung <daekyoung.jung@gmail.com>
 *
 * @file	q4_0x8_tool.h
 * @date	08 August 2025
 * @brief	Declare functions to process q4_0x8 format data
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Daekyoung Jung <daekyoung.jung@gmail.com>
 * @bug		No known bugs except for NYI items
 *
 */

namespace nntrainer {
/**
 * @brief Convert array of block_q4_0x8 into two arrays of scale, qs (st)
 * @param[in] x Array of struct block_q4_0x8
 * @param[out] d Array of unsigned short
 * @param[out] qs Array of unsigned char
 * @param N size of array @param x
 * @param K number of rows of the original weight matrix
 */
void convert_q4_0x8_shuffle(const void *x, unsigned short *d, unsigned char *qs,
                            int N, int K);

/**
 * @brief Convert array of block_q4_0x8 into two arrays of scale, qs (st)
 * @param[in] x Array of struct block_q4_0x8
 * @param[out] d Array of unsigned short
 * @param[out] qs Array of unsigned char
 * @param N size of array @param x
 * @param K number of rows of the original weight matrix
 */
void convert_q4_0x8_shuffle_omp(const void *src, unsigned short *d,
                                unsigned char *qs, int N, int K);

} // namespace nntrainer
