// SPDX-License-Identifier: Apache-2.0
/**
 * @file hvx_gemv_u8i2_lut.h
 * @brief Exact palette-coded u8 GEMV, with an experimental decode layout.
 * @author NNTrainer contributors
 * @bug See the repository issue tracker for known issues.
 */
#ifndef __NNTRAINER_HVX_GEMV_U8I2_LUT_H__
#define __NNTRAINER_HVX_GEMV_U8I2_LUT_H__

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** K is a positive multiple of 32, N of 128. No padding is implicit.
 * Layout: [N/128][K/32][8][128] bytes, still two bits per weight.
 * Byte j's output is (j%4)*32+j/4 within its 128-column panel. For pair
 * t=0..7, bits 0/2 encode rows 2t/2t+1; bits 4/6 rows 16+2t/17+2t.
 * This column permutation cancels VLUT's even/odd split and the widening
 * add's even/odd split. The four output vectors need no final shuffle.
 * Packing is lossless: the palette, channel scales and column sums stay
 * unchanged. This is NOT QS2CX_WH and must never masquerade as that type.
 */
int hvx_i2_lut_pack_wh(const uint8_t *wh, uint32_t K, uint32_t N,
                       uint8_t *packed);
int hvx_i2_lut_unpack_wh(const uint8_t *packed, uint32_t K, uint32_t N,
                         uint8_t *wh);

/** Vector repack of whole K rows. Offsets/strides are bytes, multiples of
 * 128. src contains compact WH with src_stride between 32-K rows; dst
 * contains LUT tiles with dst_k_stride/dst_n_stride between K/N tiles.
 * Disjoint input/output or in-place lower/upper half of each row is safe.
 * Each worker must own whole K rows; N must be divisible by 128.
 */
void hvx_i2_lut_repack(const uint8_t *src, uint32_t N, uint32_t src_stride,
                       uint8_t *dst, uint32_t dst_k_stride,
                       uint32_t dst_n_stride, uint32_t kt_begin,
                       uint32_t kt_end);

/** Convert a row of offline LUT tiles in place back to compact WH.
 * bytes is a multiple of 1024; row is 128-byte aligned. Used only by
 * prefill/fallback expansion; decode reads the offline image directly. */
void hvx_i2_lut_unrepack_row(uint8_t *row, uint32_t bytes);

/** Physical tables: K*32 bytes, aligned to 128. Each vector holds two
 * 16-entry tables in its even halfwords, at offsets 0 and 32 halfwords.
 * Table preparation is per activation AND palette, not per output panel.
 * Activation is the existing AH row: 32 useful bytes per 2048-byte tile.
 * All buffers passed to HVX kernels must be 128-byte aligned.
 */
void hvx_i2_lut_prepare(const uint8_t *act_ah, const int8_t *palette,
                        uint32_t kt_begin, uint32_t kt_end, int16_t *tables);

/** Exact centered activation LUT: tables encode a-127. Dequantization
 * can use zp-127 mathematically; restore int32 sums first when exact
 * matching of the original floating-point rounding is required.
 * Same storage/layout as the uncentered tables. */
void hvx_i2_lut_prepare_centered(const uint8_t *act_ah, const int8_t *palette,
                                 uint32_t kt_begin, uint32_t kt_end,
                                 int16_t *tables);
/** Two panels using centered tables; full K32 partial sums fit in int16. */
void hvx_gemv_u8i2_lut_pair_centered(const uint8_t *src0, const uint8_t *src1,
                                     uint32_t ktiles, uint32_t k_stride,
                                     const int16_t *tables, int32_t *out0,
                                     int32_t *out1);

/** Restore uncentered int32 sums before the existing floating epilogue.
 * N is a multiple of 32; out is 128-byte aligned. */
void hvx_i2_lut_restore(int32_t *out, const int32_t *colsum, uint32_t N);

/** Dense LUT: K*16 bytes, four distinct pair tables per 128-byte vector.
 * The weight layout is unchanged. No activation requantization occurs. */
void hvx_i2_lut_prepare_dense(const uint8_t *act_ah, const int8_t *palette,
                              uint32_t kt_begin, uint32_t kt_end,
                              int16_t *tables);
/** Share four tables between two output panels. src1/out1 may equal
 * src0/out0 for a single-panel tail. Strides and alignment match strided. */
void hvx_gemv_u8i2_lut_dense_pair(const uint8_t *src0, const uint8_t *src1,
                                  uint32_t ktiles, uint32_t k_stride,
                                  const int16_t *tables, int32_t *out0,
                                  int32_t *out1);

/** Computes one 128-column panel, returning 128 contiguous int32 sums.
 * Eight pair lookups fit in int16 ([-32640,28560]); the two groups are
 * combined by a widening add into int32. No LUT requantization or rounding.
 */
void hvx_gemv_u8i2_lut_panel(const uint8_t *packed, uint32_t K, uint32_t panel,
                             const int16_t *tables, int32_t *out);

/** Same 128 outputs with a configurable byte stride between K tiles.
 * Used for in-place VTCM conversion of the existing MoE DMA rows. */
void hvx_gemv_u8i2_lut_strided(const uint8_t *panel, uint32_t ktiles,
                               uint32_t k_stride, const int16_t *tables,
                               int32_t *out);

/** Two independent 128-column panels sharing each LUT register load.
 * Both sources have the same K stride; outputs are separate 128-int32
 * arrays. This supports gate/up pairs as well as two down panels. */
void hvx_gemv_u8i2_lut_pair(const uint8_t *src0, const uint8_t *src1,
                            uint32_t ktiles, uint32_t k_stride,
                            const int16_t *tables, int32_t *out0,
                            int32_t *out1);

/** Comparison kernel: retain the compact WH layout, expand only in
 * registers and feed the existing u8*i4 vrmpy arithmetic. Returns 32 sums.
 * table is hvx_expand_i2i4_table's 128-byte lookup table.
 */
void hvx_gemv_u8i2_wh_fused_col(const uint8_t *act_ah, uint32_t k_tiles,
                                const uint8_t *wh, uint32_t n_col, uint32_t nt,
                                const uint8_t *table, int32_t *out);
/** Same compact WH reader with byte stride between K rows (VTCM feed). */
void hvx_gemv_u8i2_wh_fused_strided(const uint8_t *act_ah, uint32_t k_tiles,
                                    const uint8_t *wh, uint32_t k_stride,
                                    uint32_t nt, const uint8_t *table,
                                    int32_t *out);

#ifdef __cplusplus
}
#endif
#endif
