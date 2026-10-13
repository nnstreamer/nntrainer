// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Haehun Yang <haehun.yang@ax.samsung.com>
 *
 * @file   hvx_attn_decode_f16.h
 * @date   14 Sep 2026
 * @brief  Pure-HVX flash attention for a handful of query rows (decode)
 * @see    https://github.com/nntrainer/nntrainer
 * @author Haehun Yang <haehun.yang@ax.samsung.com>
 * @bug    No known bugs except for NYI items
 */

#ifndef __NNTRAINER_HVX_ATTN_DECODE_F16_H__
#define __NNTRAINER_HVX_ATTN_DECODE_F16_H__

#include <stdint.h>

#include "hexkl_attn_f16.h"
#include "hvx_worker_pool.h"

/**
 * @brief Attention for few query rows, one (row, head) per work unit, no HMX.
 *
 * Computes QK and PV without padding query rows to an HMX tile. It
 * supports the same causal/window ranges, softcap and sinks as the FP16
 * prefill kernel. Dot products and probabilities use FP16 operands;
 * output accumulation uses FP32. Performance relative to HMX depends on
 * the cache length and GQA shape and must be measured at the call site.
 *
 * K and V are read straight from DDR with vector loads; nothing touches
 * VTCM or the DMA engine, which is what lets every pool worker run its
 * own (row, head) units with no shared state. Requires head_dim in
 * [32, 512], in multiples of 32.
 *
 * @param pool  may be NULL
 * @param us    may be NULL; total on-DSP microseconds
 * @return AEE_SUCCESS or AEE_EBADPARM
 */
int hvx_attn_decode_f16(const hexkl_attn_f16_shape *s,
                        const hexkl_attn_f16_io *io, hvx_worker_pool *pool,
                        uint64_t *us);

#endif /* __NNTRAINER_HVX_ATTN_DECODE_F16_H__ */
