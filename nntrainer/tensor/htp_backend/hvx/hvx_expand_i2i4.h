// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file   hvx_expand_i2i4.h
 * @date   23 Sep 2026
 * @brief  2-bit expert codes -> int4 WH bytes, in VTCM, before the HMX
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * QS2CX_WH stores two bits a weight and four int4 codes a tensor
 * (htp_wh_palette.h). The HMX reads int4 and only int4, so the codes have
 * to become nibbles somewhere, and the only place that pays is between the
 * DMA and the matmul: half the bytes cross DDR, the full width lands in
 * VTCM, and everything downstream of the matmul is unchanged.
 *
 * The budget is what shapes this. Doc 54 4: halving the weight DMA is worth
 * about 1.4 ms a layer, and a layer expands 168 MiB, so the expansion has
 * to produce roughly 120 GB/s to break even. That rules out anything with a
 * per-column palette -- the codes would have to be spread one to a byte and
 * looked up in a 128-entry table, about seventeen ops an output vector --
 * and it is why the format carries one palette for the whole tensor.
 *
 * With one palette, the table can be indexed by a PAIR of codes:
 *
 *     table[(hi << 2) | lo] = (pal[lo] & 0xF) | ((pal[hi] & 0xF) << 4)
 *
 * The table is a REGISTER operand: V6_vlutvvb takes it as an HVX_Vector,
 * so hvx_expand_i2i4 loads it once per call and the lookup never touches
 * memory again. Only the four palette bytes need keeping per weight; the
 * 128-byte replicated form is scratch, rebuilt in a few instructions.
 *
 * Sixteen entries, and one entry IS one output byte, because whPack2 lays
 * the codes out in whPack's nibble order: source byte j carries slots
 * 4j..4j+3, which are exactly output bytes 2j and 2j+1. So a source vector
 * splits into its two nibble planes, each plane is one vlut32, and a shuffle
 * interleaves them -- three ops an output vector, nothing to gather.
 *
 * MEASURED, not assumed. The obvious reading of V6_vlutvvb -- byte i of
 * the result is table[index[i]] -- is wrong on this hardware: the device
 * fetches entry 2*i for index i (EXPAND_I2I4_MAP, doc 54 5.7-5.8). It is a
 * plain permutation of the index, the same in every lane, so the table is
 * baked with entry i at byte 2*i and the loop is unchanged -- still three
 * ops an output vector, the scaling costs nothing at run time. The 32-byte
 * block is then repeated across the vector, which covers the other half of
 * the original guess (whether the table comes from the low bytes or from
 * each lane group) without having to know which it is.
 *
 * hvx_expand_i2i4_scalar reads the same table through the same scaling, so
 * there is one table and not two. HvxExpandI2I4.MatchesScalarBitExact is
 * the device gate and it passes: bad=0 of 131072 on four palettes.
 */

#ifndef __NNTRAINER_HVX_EXPAND_I2I4_H__
#define __NNTRAINER_HVX_EXPAND_I2I4_H__

#include <stdint.h>

/* Forward declared rather than including hvx_worker_pool.h: that header
   has _Atomic members, and expand_i2i4_host_check is C++. Same typedef,
   which C and C++ both allow to repeat. */
typedef struct hvx_worker_pool_s hvx_worker_pool;

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Table bytes: 16 entries replicated to a full HVX vector. */
#define HVX_EXPAND_TABLE_BYTES 128u

/**
 * @brief Builds the code-pair table the expansion looks up
 *
 * @param pal   four int4 codes, ascending -- the tensor's palette
 * @param table HVX_EXPAND_TABLE_BYTES bytes, 128-byte aligned when it will
 *              be handed to hvx_expand_i2i4
 */
void hvx_expand_i2i4_table(const int8_t *pal, uint8_t *table);

/**
 * @brief The specification: one output byte per table lookup, scalar
 *
 * Kept in the shipped file rather than a test, because it is what the HVX
 * path is required to equal and what the host checks run.
 *
 * @param src       2-bit codes in whPack2 order
 * @param src_bytes how many, a multiple of 128 for the HVX path
 * @param table     from hvx_expand_i2i4_table
 * @param dst       2 * src_bytes int4 bytes in whPack order
 */
void hvx_expand_i2i4_scalar(const uint8_t *src, uint32_t src_bytes,
                            const uint8_t *table, uint8_t *dst);

/**
 * @brief Expands src_bytes of codes into 2 * src_bytes of int4, HVX
 *
 * src and dst are VTCM; src_bytes must be a multiple of 128. A WH tile is
 * 256 code bytes, so every shape this is called with already is.
 */
void hvx_expand_i2i4(const uint8_t *src, uint32_t src_bytes,
                     const uint8_t *table, uint8_t *dst);

/**
 * @brief Same, split across the worker pool. NULL pool runs it inline.
 *
 * Split by whole 128-byte source vectors, so no worker ever writes a byte
 * another worker's shuffle also produces.
 */
void hvx_expand_i2i4_pool(const uint8_t *src, uint32_t src_bytes,
                          const uint8_t *table, uint8_t *dst,
                          hvx_worker_pool *pool);

#ifdef __cplusplus
}
#endif

#endif /* __NNTRAINER_HVX_EXPAND_I2I4_H__ */
