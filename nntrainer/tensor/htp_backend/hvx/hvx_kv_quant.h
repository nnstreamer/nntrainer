// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Haehun Yang <haehun.yang@ax.samsung.com>
 *
 * @file   hvx_kv_quant.h
 * @date   28 Sep 2026
 * @brief  HVX quantization of one appended KV row (per head) for hexkl_kv_q
 * @see    https://github.com/nntrainer/nntrainer
 * @author Haehun Yang <haehun.yang@ax.samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * The vector counterpart of hexkl_kv_q_quant_k_row / _v_row: the fp16 ->
 * f32 conversion, the absolute maximum, the multiply, the round-to-nearest-
 * even and the clamp run on HVX; the scale and its reciprocal are computed
 * on the scalar unit from the vector's maximum exactly as the scalar
 * routines do, so the scales are bit-identical and the values differ from
 * the scalar path only where the f32 product rounds differently (rare ties,
 * one step). The registry's device test allows for that.
 */

#ifndef __NNTRAINER_HVX_KV_QUANT_H__
#define __NNTRAINER_HVX_KV_QUANT_H__

#include "hvx_worker_pool.h"
#include <stdint.h>

/** Quantize dense attention heads on the DSP, with the CPU Q8 wire's f32
 * rounding. K/V are fp16, Q is f32; hd is a multiple of 32, <= 256.
 * Outputs are private scratch until success (0); non-finite input returns -1.
 * kv_scales interleaves K/V scales for each head. */
int hvx_attn_quant_rows_q8(const uint16_t *k, const uint16_t *v, const float *q,
                           uint32_t kv_heads, uint32_t q_heads, uint32_t hd,
                           uint8_t *k8, uint8_t *v8, float *kv_scales,
                           uint8_t *q8, float *q_scales, int32_t *q_zp,
                           hvx_worker_pool *pool);

/**
 * @brief K row of one head: symmetric over @a hd fp16 values -> int8
 *        values in [-qmax, qmax], scale = amax / qmax (1 for an all-zero
 *        row), colsum = sum of the values. hd a multiple of 32, <= 512.
 */
void hvx_kv_quant_k_row(const uint16_t *x_hf, uint32_t hd, int32_t qmax,
                        int8_t *q, float *scale, int32_t *colsum);

/**
 * @brief V row of one head: symmetric over all @a hd values, one scale.
 *
 * @param[out] scale  one entry
 */
void hvx_kv_quant_v_row(const uint16_t *x_hf, uint32_t hd, int32_t qmax,
                        int8_t *q, float *scale);

/** @brief K row at a given scale: q = rne(x * inv) clamped, colsum. */
void hvx_kv_quant_k_row_fixed(const uint16_t *x_hf, uint32_t hd, int32_t qmax,
                              float inv, int8_t *q, int32_t *colsum);

/** @brief V row at a given scale per dim: q[d] = rne(x[d] * inv[d]) clamped.
 *         @a inv is hd f32, 128-byte aligned or not (loaded unaligned). */
void hvx_kv_quant_v_row_fixed(const uint16_t *x_hf, uint32_t hd, int32_t qmax,
                              const float *inv, int8_t *q);

#endif /* __NNTRAINER_HVX_KV_QUANT_H__ */
