// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Haehun Yang <haehun.yang@ax.samsung.com>
 *
 * @file   hexkl_attn_q.h
 * @date   28 Sep 2026
 * @brief  A8W8 / A8W4 flash attention on HMX over a quantized KV cache
 * @see    https://github.com/nntrainer/nntrainer
 * @author Haehun Yang <haehun.yang@ax.samsung.com>
 * @bug    No known bugs except for NYI items
 */

#ifndef __NNTRAINER_HEXKL_ATTN_Q_H__
#define __NNTRAINER_HEXKL_ATTN_Q_H__

#include <stdint.h>

#include "hexkl_attn_q_plan.h"
#include "hexkl_kv_q.h"
#include "hvx_worker_pool.h"

/**
 * @brief Operands. Q and out as MHACoreLayer has them (f32, post-RoPE, head
 *        h at column h*head_dim); the cache is a registered hexkl_kv_q,
 *        which fixes n_head_kv, head_dim and the kind, and must hold at
 *        least cache_to rows.
 */
typedef struct {
  const float *q;
  uint32_t q_stride;
  const float *sinks; /**< NULL or n_head_q per-head sink logits */
  float *out;
  uint32_t out_stride;
  const hexkl_kv_q *kv;
  uint32_t decode_flags; /**< 0: generic; 1: four-row epilogues/byte planes;
                             3: also overlap heads when one KV block fits.
                             Nonzero requires Q8, n_q=1, GQA=4, hd=64. */
} hexkl_attn_q_io;

/** @brief Already quantized dense Q; metadata is per (query row, head). */
typedef struct {
  const uint8_t *values;
  const float *scales;
  const int32_t *zero_points;
} hexkl_attn_q8_input;

/** @brief On-DSP phase times in microseconds. In the decode head pipeline,
 * us_softmax includes exposed dequant/softmax/P wait; us_dequant and
 * us_pquant report overlapping worker busy time. These counters must not be
 * added to estimate total wall time.
 */
typedef struct {
  uint64_t us_qprep;   /**< Q gather + uint8 quantization into AH tiles */
  uint64_t us_dma;     /**< K^T / V tile DMA incl. the drain, block scales */
  uint64_t us_qk;      /**< HMX Q.K^T incl. accumulator reads */
  uint64_t us_dequant; /**< int32 accumulator -> fp16 score tiles */
  uint64_t us_softmax; /**< HVX online softmax (exposed time) */
  uint64_t us_pquant;  /**< P * s_v -> uint8 P' tiles */
  uint64_t us_pv;      /**< HMX P'.V incl. accumulator reads */
  uint64_t us_oupd;    /**< f32 partial output rescale + accumulate */
  uint64_t us_store;   /**< O/l -> f32 rows in DDR */
  uint64_t us_total;   /**< whole call, qtimer */
  uint64_t pcycles;    /**< whole call, processor cycles: the ratio to
                            us_total is the clock the DSP actually ran at */
  uint32_t n_blocks;   /**< (kv_head, q_block, kv_block) iterations run */
} hexkl_attn_q_stats;

/**
 * @brief Runs causal (optionally windowed) attention for one step over the
 *        quantized cache: the math of hexkl_attn_f16_prefill with Q as
 *        per-row uint8, K^T and V as the cache's int8 / int4 tiles, P as
 *        per-row uint8 with the per-token V scales folded in, and the
 *        partial output accumulated in f32 on HVX.
 *
 * Requires the HMX lock and the session's int32 accumulator config at
 * @a config_off (the arena top); this kernel writes no HMX config.
 *
 * @param pool  may be NULL: everything runs on the calling thread
 * @param st    may be NULL
 * @return AEE_SUCCESS, AEE_EBADPARM, AEE_ENOMEMORY, or a HexKL error
 */
int hexkl_attn_q_prefill(uint8_t *vtcm_base, uint32_t config_off,
                         const hexkl_attn_f16_shape *s,
                         const hexkl_attn_f16_tiling *t,
                         const hexkl_attn_q_io *io, hvx_worker_pool *pool,
                         hexkl_attn_q_stats *st);

/** @brief Same kernel with Q quantization performed by the producer. */
int hexkl_attn_q_prefill_u8(uint8_t *vtcm_base, uint32_t config_off,
                            const hexkl_attn_f16_shape *s,
                            const hexkl_attn_f16_tiling *t,
                            const hexkl_attn_q_io *io,
                            const hexkl_attn_q8_input *q8,
                            hvx_worker_pool *pool, hexkl_attn_q_stats *st);

#endif /* __NNTRAINER_HEXKL_ATTN_Q_H__ */
