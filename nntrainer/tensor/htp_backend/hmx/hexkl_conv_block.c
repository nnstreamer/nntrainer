// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file   hexkl_conv_block.c
 * @date   22 Sep 2026
 * @brief  An LFM2 conv block -- in_proj, gate, conv1d, gate, out_proj -- in
 *         one call
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * The header says what and why; this file is the two loops. Everything
 * about buffer reuse follows hexkl_mm_u8i4_moe.c and its rules apply:
 * two accumulator staging buffers alternate under the HMX issue, the
 * pool runs one batch behind, and every wait sits where the dependency
 * is. The one new hazard is g: phase 1's epilogues write it row-block by
 * row-block and phase 2 reads it with a two-row look-back, so phase 2
 * starts only after phase 1's last epilogue is retired.
 */

#include "hexkl_conv_block.h"

#include <stdatomic.h>
#include <string.h>

#include <AEEStdErr.h>

#include "hexkl_acc_tile.h"
#include "hexkl_dma_ring.h"
#include "hexkl_micro.h"
#include "hexkl_probe.h"
#include "hvx_conv_gate_f32.h"
#include "hvx_dequant_i32.h"
#include "hvx_gemm_u8i4_wh.h"
#include "hvx_quant_u8.h"

#define ROUND_UP_U32(v, a) ((((v) + ((a)-1)) / (a)) * (a))
#define ROUND_UP_SZ(v, a) ((((v) + ((a)-1)) / (a)) * (a))
#define CB_SCRATCH_ALIGN 128u

/** @brief Kept in sync with hexkl_mm_u8i4_dma.c's copies by inspection. */
#define WEIGHT_TILE_BYTES_U8I4 512u
#define ACC_TILE_BYTES 8192u

/** @brief Bounded chunk-index arrays; a shape needing more is refused. */
#define CB_MAX_CHUNKS 16u

/** Input quantization and AH packing share 16-row background units. Four
 * units publish one HMX row block, keeping the first block parallel while
 * later blocks can be produced under earlier blocks' HMX work. Each row
 * keeps the same min/max and rounding arithmetic. The pool's release/acquire
 * done byte publishes both the parameters and the tiles. */
#define CB_PACK_UNIT_ROWS 16u
#define CB_PACK_UNITS_THROUGH(row)                                             \
  (((row) + HEXKL_HMX_INT8_BLOCK_N_ROW) / CB_PACK_UNIT_ROWS)

typedef struct {
  const float *act_c;
  float *scale;
  int32_t *zp;
  uint8_t *act_ah;
  uint32_t M, K;
} cb_pack_ctx;

static void cb_pack_unit(uint32_t n_units, uint32_t u, void *v) {
  (void)n_units;
  const cb_pack_ctx *c = (const cb_pack_ctx *)v;
  const uint32_t r0 = u * CB_PACK_UNIT_ROWS;
  const uint32_t remaining = r0 < c->M ? c->M - r0 : 0u;
  const uint32_t n =
    remaining < CB_PACK_UNIT_ROWS ? remaining : CB_PACK_UNIT_ROWS;
  /** The full activation DMA and zero padding completed before submit.
     Do not submit nested work to the worker pool from a background unit. */
  hvx_quant_rows_u8_params(c->act_c + (size_t)r0 * c->K, n, CB_PACK_UNIT_ROWS,
                           c->K, c->scale + r0, c->zp + r0, NULL);
  hvx_quant_pack_u8_ah_rows(c->act_c, NULL, r0, r0 + CB_PACK_UNIT_ROWS, c->K,
                            c->scale, c->zp, c->act_ah);
}

/** ---- phase 2's background stage: z *= conv1d(g), then z -> mid (u8) ----
 *
 * Independent 16-row units gate, derive quantization parameters and pack
 * their own AH rows. There is no whole-block gate barrier or single-worker
 * requantization tail. Four-row pack groups stay on one worker, so stores
 * are disjoint. The next block's b matmul and the previous block's out_proj
 * can run on HMX while the HVX workers produce this block's mid tiles. */
#define CB_STAGE_UNIT_ROWS 16u

typedef struct {
  float *z;            /**< this block's [64 x C] f32, VTCM */
  const float *g;      /**< two carry rows followed by the conv input, heap */
  const float *conv_w; /**< [3 x C], VTCM */
  uint8_t *mid;        /**< this block's AH tiles, VTCM */
  float *rq_scale;     /**< this block's row params, 64 each */
  int32_t *rq_zp;
  uint32_t mb, m_blk, C;
} cb_stage_ctx;

/** @brief Summed worker time, not wall time. Atomic because the units run
 *         concurrently; HEXKL_PROBE_ADD is not. */
static inline void cb_stage_probe_add(uint64_t t0) {
  if (hexkl_probe_on) {
    atomic_fetch_add_explicit(
      (_Atomic uint64_t *)&hexkl_probe_us[HEXKL_PROBE_SWIGLU],
      hexkl_probe_now() - t0, memory_order_relaxed);
  }
}

static void cb_stage_unit(uint32_t n_units, uint32_t u, void *v) {
  (void)n_units;
  const cb_stage_ctx *c = (const cb_stage_ctx *)v;
  const uint32_t r0 = u * CB_STAGE_UNIT_ROWS;
  if (r0 >= c->m_blk) {
    return;
  }
  const uint32_t n =
    (c->m_blk - r0 < CB_STAGE_UNIT_ROWS) ? (c->m_blk - r0) : CB_STAGE_UNIT_ROWS;
  const uint32_t n4 = ROUND_UP_U32(n, 4u);
  float *z = c->z + (size_t)r0 * c->C;
  uint64_t t0 = 0;
  HEXKL_PROBE_T0(t0);
  hvx_conv_gate_f32(z, c->C, c->g, c->C, c->mb + r0, n, c->C, c->conv_w, NULL);
  /* The pack reads complete four-row groups, including the final tail. */
  if (n4 > n) {
    memset(z + (size_t)n * c->C, 0, sizeof(float) * (size_t)(n4 - n) * c->C);
  }
  hvx_quant_rows_u8_params(z, n, n4, c->C, c->rq_scale + r0, c->rq_zp + r0,
                           NULL);
  hvx_quant_pack_u8_ah_rows(c->z, NULL, r0, r0 + n4, c->C, c->rq_scale,
                            c->rq_zp, c->mid);
  cb_stage_probe_add(t0);
}

/** @brief Maximum units in one block's background stage. */
#define CB_STAGE_UNITS                                                         \
  ((HEXKL_HMX_INT8_BLOCK_N_ROW + CB_STAGE_UNIT_ROWS - 1u) / CB_STAGE_UNIT_ROWS)

int hexkl_conv_block_layout_get(uint32_t K, uint32_t C, uint32_t N_out,
                                uint32_t arena_bytes,
                                hexkl_conv_block_layout *out) {
  if (!out || K == 0u || C == 0u || N_out == 0u) {
    return AEE_EBADPARM;
  }
  if ((K % HEXKL_HMX_INT8_BLOCK_N_INNER) != 0u ||
      (C % HEXKL_HMX_INT8_BLOCK_N_COL) != 0u ||
      (C % HEXKL_HMX_INT8_BLOCK_N_INNER) != 0u ||
      (N_out % HEXKL_HMX_INT8_BLOCK_N_COL) != 0u) {
    return AEE_EBADPARM;
  }
  const uint32_t k_tiles = K / HEXKL_HMX_INT8_BLOCK_N_INNER;
  const uint32_t c_ntiles = C / HEXKL_HMX_INT8_BLOCK_N_COL;
  const uint32_t c_ktiles = C / HEXKL_HMX_INT8_BLOCK_N_INNER;
  const uint32_t o_ntiles = N_out / HEXKL_HMX_INT8_BLOCK_N_COL;

  const uint32_t act_bytes = k_tiles * HEXKL_HMX_ACTIVATION_ALIGNMENT;
  const uint32_t w_in_bytes = k_tiles * c_ntiles * WEIGHT_TILE_BYTES_U8I4;
  const uint32_t w_out_bytes = c_ktiles * o_ntiles * WEIGHT_TILE_BYTES_U8I4;
  /* Slot A holds W_a then W_b (both K x C); slot B holds W_c then W_out. */
  const uint32_t slot_b_bytes =
    w_in_bytes > w_out_bytes ? w_in_bytes : w_out_bytes;
  const uint32_t z_bytes = HEXKL_HMX_INT8_BLOCK_N_ROW * C * 4u;
  const uint32_t mid_bytes = c_ktiles * HEXKL_HMX_ACTIVATION_ALIGNMENT;
  const uint32_t conv_w_bytes = 3u * C * 4u;

  hexkl_conv_block_layout L;
  L.act_off = 0u;
  L.w_a_off =
    ROUND_UP_U32(L.act_off + act_bytes, HEXKL_HMX_ACTIVATION_ALIGNMENT);
  L.w_b_off =
    ROUND_UP_U32(L.w_a_off + w_in_bytes, HEXKL_HMX_ACTIVATION_ALIGNMENT);
  L.z_off =
    ROUND_UP_U32(L.w_b_off + slot_b_bytes, HEXKL_HMX_ACTIVATION_ALIGNMENT);
  /** z and mid twice: phase 2 pipelines blocks (block b's gate and
     requant run on the pool's background lane under block b+1's b matmul
     and block b's out_proj), so each has a live reader while the other
     is being written. */
  L.mid_off =
    ROUND_UP_U32(L.z_off + 2u * z_bytes, HEXKL_HMX_ACTIVATION_ALIGNMENT);
  L.conv_w_off =
    ROUND_UP_U32(L.mid_off + 2u * mid_bytes, HEXKL_HMX_ACTIVATION_ALIGNMENT);
  L.result_off =
    ROUND_UP_U32(L.conv_w_off + conv_w_bytes, HEXKL_HMX_ACTIVATION_ALIGNMENT);
  {
    /** Same cap and pairing rule as hexkl_mm_u8i4_moe_layout: 32 tiles a
       staging buffer, even, at least one pair. */
    uint32_t want = 2u * c_ntiles;
    uint32_t fits = 0u;
    if (arena_bytes > L.result_off) {
      fits = (arena_bytes - L.result_off) / (2u * ACC_TILE_BYTES);
    }
    if (want > 32u) {
      want = 32u;
    }
    L.acc_tiles = (fits < want) ? fits : want;
    L.acc_tiles &= ~1u;
    if (L.acc_tiles < 2u) {
      return AEE_ENOMEMORY;
    }
  }
  L.total = L.result_off + 2u * L.acc_tiles * ACC_TILE_BYTES;
  if (L.total > arena_bytes) {
    return AEE_ENOMEMORY;
  }
  *out = L;
  return AEE_SUCCESS;
}

/** @brief A registered weight of the expected shape, or NULL. */
static const hexkl_weight_u8i4 *cb_weight(const hexkl_weight_u8i4_table *tbl,
                                          uint32_t h, uint32_t K, uint32_t N) {
  if (h >= HEXKL_MM_U8I4_MAX_WEIGHTS) {
    return NULL;
  }
  const hexkl_weight_u8i4 *w = &tbl->slots[h];
  return (w->in_use && w->bits != 2u && w->K == K && w->N == N) ? w : NULL;
}

/**
 * @brief Pushes a weight in n-tile chunks of @a cn into @a dst_off.
 * @return chunks pushed; idx_out gets each chunk's ring index
 */
static uint32_t cb_push_chunks(uint8_t *vtcm_base, uint32_t dst_off,
                               const hexkl_weight_u8i4 *h, uint32_t k_tiles,
                               uint32_t n_col, uint32_t cn, uint32_t *idx_out,
                               int weight_src_bypass) {
  uint32_t n = 0u;
  for (uint32_t nt0 = 0; nt0 < n_col; nt0 += cn) {
    const uint32_t c = (n_col - nt0 < cn) ? (n_col - nt0) : cn;
    idx_out[n++] = hexkl_moe_push_weight_chunk(
      vtcm_base, dst_off, h, k_tiles, n_col, nt0, c, weight_src_bypass);
  }
  return n;
}

/**
 * @brief Enqueue an immutable output row block without stalling HMX issue.
 *
 * The cached source is written by HVX, so both DDR endpoints use the same
 * non-bypass flags as hexkl_moe_dma_copy. Chunk/row sizing follows that
 * helper; the caller drains before returning or reusing scratch. No ring
 * index is retained across blocks, so output copies can wrap the ring.
 */
static void cb_push_output(float *dst, const float *src, uint32_t row,
                           uint32_t rows, uint32_t cols) {
  const size_t begin = (size_t)row * cols;
  const size_t bytes = sizeof(float) * (size_t)rows * cols;
  for (size_t off = 0; off < bytes;) {
    const uint32_t n =
      bytes - off > (1u << 20) ? (1u << 20) : (uint32_t)(bytes - off);
    uint32_t rs = 16384u;
    while ((n % rs) != 0u) {
      rs >>= 1;
    }
    hexkl_dma_ring_push2d((uint8_t *)(dst + begin) + off,
                          (const uint8_t *)(src + begin) + off, rs, rs, rs,
                          n / rs, 0, 0);
    off += n;
  }
}

/** M=1 uses compact column slabs instead of the HMX 64-row accumulator.
 * Only the calling thread owns the DMA ring. A pool join retires all readers
 * of a slot before its next DMA; at most two groups of three descriptors are
 * live, including across ring wrap. AH keeps its existing 2048-byte pitch. */
typedef struct {
  const hexkl_weight_u8i4 *w[3]; /* A, C, B, or just O */
  const uint8_t *act, *weights;
  const float *scale;
  const int32_t *zp;
  float *g, *out;
  uint32_t k_tiles, cols, nt0, plane_bytes, width, rows1, n_weights, chunk;
} cb_m1_ctx;

static void cb_m1_worker(uint32_t n_lanes, uint32_t lane, void *v) {
  const cb_m1_ctx *c = (const cb_m1_ctx *)v;
  const uint32_t lo = c->cols * lane / n_lanes;
  const uint32_t hi = c->cols * (lane + 1u) / n_lanes;
  int32_t tiles[3u * 32u] __attribute__((aligned(128)));
  for (uint32_t j = lo; j < hi; ++j) {
    const uint32_t col = (c->nt0 + j) * 32u;
    if (c->n_weights == 3u && c->rows1) {
      hvx_gemv_u8i4_wh_3col_nopf(c->act, c->k_tiles, c->weights, c->plane_bytes,
                                 c->cols, j, tiles);
    } else if (c->n_weights == 1u && c->rows1) {
      hvx_gemv_u8i4_wh_col_aligned_nopf(c->act, c->k_tiles, c->weights, c->cols,
                                        j, tiles);
    } else {
      for (uint32_t w = 0; w < c->n_weights; ++w)
        hvx_gemm_u8i4_wh_col_nopf(c->act, 1u, c->k_tiles,
                                  c->weights + (size_t)w * c->plane_bytes,
                                  c->cols, j, c->rows1, tiles + w * 32u);
    }
    if (c->n_weights == 3u) {
      /** Reuse the HMX epilogue, including every FP32 rounding point. This
       * direct worker call does not recursively submit to the pool. */
      hvx_dq_mul_job mul = {.tiles_base = (const uint8_t *)tiles,
                            .tile_stride = 32u * sizeof(int32_t),
                            .n_pairs = 1u,
                            .c0 = col,
                            .row_stride = 32u,
                            .m_count = 1u,
                            .act_scale = c->scale,
                            .act_zp = c->zp,
                            .colsum_a = c->w[0]->colsum_w,
                            .w_scale_a = c->w[0]->w_scale,
                            .bias_a = c->w[0]->bias,
                            .colsum_b = c->w[1]->colsum_w,
                            .w_scale_b = c->w[1]->w_scale,
                            .bias_b = c->w[1]->bias,
                            .dst = c->g,
                            .dst_stride = c->width};
      hvx_dq_mul_worker(1u, 0u, &mul);
    }
    const uint32_t last = c->n_weights - 1u;
    const hexkl_weight_u8i4 *w = c->w[last];
    hvx_dequant_acc_tile_to_f32(tiles + last * 32u, 32u, 1u, c->scale, c->zp,
                                w->colsum_w + col, w->w_scale + col,
                                w->bias + col, c->out + col, c->width, 0);
  }
}

/** Queue a contiguous transfer with the same row/chunk limits as the common
 * DMA copy helper, but leave it in flight for double-buffered decode. */
static uint32_t cb_push_linear(uint8_t *dst, const uint8_t *src, uint32_t bytes,
                               int src_bypass, int dst_bypass) {
  uint32_t idx = 0u;
  for (uint32_t off = 0u; off < bytes;) {
    const uint32_t n = bytes - off > (1u << 20) ? (1u << 20) : bytes - off;
    uint32_t rs = 16384u;
    while (n % rs)
      rs >>= 1;
    idx = hexkl_dma_ring_next_idx();
    hexkl_dma_ring_push2d(dst + off, src + off, rs, rs, rs, n / rs, src_bypass,
                          dst_bypass);
    off += n;
  }
  return idx;
}

static int cb_ranges_overlap(const uint8_t *a, uint64_t a_bytes,
                             const uint8_t *b, uint64_t b_bytes) {
  const uint64_t x = (uintptr_t)a, y = (uintptr_t)b;
  return a && b && x < y + b_bytes && y < x + a_bytes;
}

int hexkl_conv_block_prepare_decode(hexkl_weight_u8i4_table *tbl, uint8_t *vtcm,
                                    uint32_t arena, uint32_t handle,
                                    uint8_t *dst, uint32_t chunk) {
  if (!tbl || handle >= HEXKL_MM_U8I4_MAX_WEIGHTS)
    return AEE_EBADPARM;
  hexkl_weight_u8i4 *w = &tbl->slots[handle];
  if (!w->in_use || w->bits == 2u)
    return AEE_EBADPARM;
  if (chunk == 0u) {
    w->conv_decode_bytes = NULL;
    w->conv_decode_chunk = 0u;
    return AEE_SUCCESS;
  }
  if (!vtcm || !dst || (uintptr_t)dst % 512u || !w->K || !w->N || w->K % 32u ||
      w->N % 32u || w->K > 8192u || w->N > 8192u ||
      (chunk != 4u && chunk != 8u && chunk != 16u && chunk != 24u &&
       chunk != 32u))
    return AEE_EBADPARM;
  const uint32_t kt = w->K / 32u, nt = w->N / 32u;
  const uint32_t cols = nt < chunk ? nt : chunk;
  if ((uint64_t)kt * cols * 512u > arena)
    return AEE_ENOMEMORY;
  const uint32_t bytes = kt * nt * 512u;
  /** A preparation must never overwrite live weight bytes. Repacking the
   * same destination is allowed, since callers serialize registration. */
  for (uint32_t i = 0u; i < HEXKL_MM_U8I4_MAX_WEIGHTS; ++i) {
    const hexkl_weight_u8i4 *other = &tbl->slots[i];
    if (!other->in_use)
      continue;
    const uint64_t size = (uint64_t)other->K * other->N / 2u;
    if (cb_ranges_overlap(dst, bytes, other->wh_bytes,
                          other->bits == 2u ? size / 2u : size) ||
        (cb_ranges_overlap(dst, bytes, other->conv_decode_bytes, size) &&
         (other != w || dst != other->conv_decode_bytes)))
      return AEE_EBADPARM;
  }
  hexkl_dma_ring_reset();
  hexkl_dcache_inva(dst, bytes);
  for (uint32_t nt0 = 0u; nt0 < nt; nt0 += chunk) {
    const uint32_t n = nt - nt0 < chunk ? nt - nt0 : chunk;
    hexkl_dma_ring_push2d(vtcm, w->wh_bytes + (size_t)nt0 * 512u, n * 512u,
                          nt * 512u, n * 512u, kt,
                          hexkl_weight_u8i4_src_bypass(w, 1), 1);
    hexkl_dma_ring_drain();
    cb_push_linear(dst + (size_t)nt0 * kt * 512u, vtcm, n * kt * 512u, 1, 1);
    hexkl_dma_ring_drain();
  }
  w->conv_decode_bytes = dst;
  w->conv_decode_chunk = chunk;
  return AEE_SUCCESS;
}

/** Pack [kt][nt0..nt0+cols) into [kt][0..cols) in the slot. The GEMV uses
 * cols (not the original weight's N/32) as its new k-tile row pitch. */
static uint32_t cb_m1_push(const cb_m1_ctx *c, uint8_t *slot, uint32_t nt0,
                           uint32_t cols, int bypass) {
  uint32_t idx = 0u;
  for (uint32_t i = 0; i < c->n_weights; ++i) {
    const hexkl_weight_u8i4 *w = c->w[i];
    if (w->conv_decode_bytes && w->conv_decode_chunk == c->chunk) {
      idx =
        cb_push_linear(slot + (size_t)i * c->plane_bytes,
                       w->conv_decode_bytes + (size_t)nt0 * c->k_tiles * 512u,
                       cols * 512u * c->k_tiles, bypass, 1);
    } else {
      idx = hexkl_dma_ring_next_idx();
      hexkl_dma_ring_push2d(slot + (size_t)i * c->plane_bytes,
                            w->wh_bytes + (size_t)nt0 * 512u, cols * 512u,
                            (w->N / 32u) * 512u, cols * 512u, c->k_tiles,
                            hexkl_weight_u8i4_src_bypass(w, bypass), 1);
    }
    HEXKL_PROBE_COUNT(HEXKL_PROBE_DMA_KB,
                      ((uint64_t)cols * 512u * c->k_tiles) >> 10);
  }
  return idx;
}

/** The first group is queued by the caller before quantization. Queue the
 * second here and then alternate: wait current, run/join current, refill
 * its retired slot while the other slot is ready or still being filled. */
static void cb_m1_project(cb_m1_ctx *c, uint8_t *feed, uint32_t slot_bytes,
                          uint32_t chunk, uint32_t first_idx,
                          hvx_worker_pool *pool, int bypass) {
  const uint32_t nt = c->width / 32u;
  uint32_t idx[2] = {first_idx, 0u};
  if (chunk < nt) {
    const uint32_t n = nt - chunk < chunk ? nt - chunk : chunk;
    idx[1] = cb_m1_push(c, feed + slot_bytes, chunk, n, bypass);
  }
  uint64_t t0 = 0;
  for (uint32_t nt0 = 0, seq = 0; nt0 < nt; nt0 += chunk, ++seq) {
    const uint32_t slot = seq & 1u;
    uint64_t wait0 = 0;
    HEXKL_PROBE_T0(wait0);
    hexkl_dma_ring_wait(idx[slot]);
    HEXKL_PROBE_ADD(HEXKL_PROBE_DRAIN, wait0);
    c->nt0 = nt0;
    c->cols = nt - nt0 < chunk ? nt - nt0 : chunk;
    c->weights = feed + slot * slot_bytes;
    HEXKL_PROBE_T0(t0);
    hvx_worker_pool_run(pool, cb_m1_worker, c, c->cols);
    HEXKL_PROBE_ADD(HEXKL_PROBE_MM, t0);
    const uint32_t next = nt0 + 2u * chunk;
    if (next < nt) {
      const uint32_t n = nt - next < chunk ? nt - next : chunk;
      idx[slot] = cb_m1_push(c, feed + slot * slot_bytes, next, n, bypass);
    }
  }
}

static int cb_m1_run(uint8_t *vtcm, uint32_t arena, uint32_t K, uint32_t C,
                     uint32_t N, const hexkl_weight_u8i4 *wa,
                     const hexkl_weight_u8i4 *wb, const hexkl_weight_u8i4 *wc,
                     const hexkl_weight_u8i4 *wo, const float *conv_w,
                     const float *x, float *out, float *state,
                     hvx_worker_pool *pool, hexkl_moe_scratch *scratch,
                     int bypass, uint32_t flags) {
  const uint32_t chunk = htp_conv_chunk_tiles(flags);
  const uint32_t out_chunk = htp_conv_out_chunk_tiles(flags);
  HEXKL_PROBE_COUNT(HEXKL_PROBE_PATH, 1u);
  HEXKL_PROBE_COUNT(HEXKL_PROBE_M1_FEED, 1u);
  const uint32_t max_k = K > C ? K : C;
  const uint32_t plane = (max_k / 32u) * chunk * 512u;
  /** O has one plane, so it can use the whole retired A/C/B slot. Keep
   * slot pitch independent of projection pitch, including a short final
   * slab. At K=C=2048, in8/out16 and in8/out24 use the original 1.5 MiB. */
  const uint32_t out_plane = (C / 32u) * out_chunk * 512u;
  const uint32_t slot_bytes = 3u * plane > out_plane ? 3u * plane : out_plane;
  /** All shapes here are multiples of 32. Every region is 128-byte aligned;
   * AH is separately allocated in cached scratch. Four FP32 rows are needed
   * only by the existing pack helper. It writes rows 0..3; GEMV reads row 0
   * only. */
  const uint64_t need = 16ull * K + 40ull * C + 4ull * N + 2ull * slot_bytes;
  if (need > arena) {
    return AEE_ENOMEMORY;
  }
  /** GEMV broadcasts activation words using scalar loads. Keep AH in cached
   * DDR (as MoE does), not VTCM: scalar VTCM loads serialize this loop. */
  const int rc = hexkl_moe_scratch_reserve(scratch, 64u * (size_t)(K + C));
  if (rc != AEE_SUCCESS)
    return rc;
  uint8_t *act = scratch->base, *mid = act + 64u * K;
  float *xc = (float *)vtcm;
  float *z = xc + 4u * K;
  float *g = z + 4u * C;
  float *cw = g + 3u * C;
  float *yc = cw + 3u * C;
  uint8_t *feed = (uint8_t *)(yc + N);
  float scale[4], rq_scale[4];
  int32_t zp[4], rq_zp[4];
  uint64_t t0 = 0;
  HEXKL_PROBE_T0(t0);
  hexkl_dma_ring_reset();
  hexkl_moe_dma_copy(xc, x, (size_t)K * 4u, 0, 1);
  hexkl_moe_dma_copy(g, state, (size_t)C * 8u, 0, 1);
  hexkl_moe_dma_copy(cw, conv_w, (size_t)C * 12u, 0, 1);
  memset(xc + K, 0, (size_t)K * 12u);
  memset(z + C, 0, (size_t)C * 12u);
  HEXKL_PROBE_ADD(HEXKL_PROBE_ACC_COPY, t0);

  cb_m1_ctx c = {.w = {wa, wc, wb},
                 .act = act,
                 .scale = scale,
                 .zp = zp,
                 .g = g + 2u * C,
                 .out = z,
                 .k_tiles = K / 32u,
                 .plane_bytes = plane,
                 .width = C,
                 .rows1 = (flags & HTP_CONV_FLAG_ROWS4) ? 0u : 1u,
                 .n_weights = 3u,
                 .chunk = chunk};
  uint32_t cols = C / 32u < chunk ? C / 32u : chunk;
  uint32_t first = cb_m1_push(&c, feed, 0u, cols, bypass);
  HEXKL_PROBE_T0(t0);
  hvx_quant_rows_u8_params(xc, 1u, 4u, K, scale, zp, NULL);
  hvx_quant_pack_u8_ah_rows(xc, NULL, 0u, 4u, K, scale, zp, act);
  HEXKL_PROBE_ADD(HEXKL_PROBE_QUANT, t0);
  cb_m1_project(&c, feed, slot_bytes, chunk, first, pool, bypass);

  /** Reuse the now-retired input slots for O. Its first DMA overlaps gate
   * and whole-row requantization, which must finish before any O GEMV. */
  c.w[0] = wo;
  c.n_weights = 1u;
  c.k_tiles = C / 32u;
  c.width = N;
  c.act = mid;
  c.scale = rq_scale;
  c.zp = rq_zp;
  c.out = yc;
  c.chunk = out_chunk;
  cols = N / 32u < out_chunk ? N / 32u : out_chunk;
  first = cb_m1_push(&c, feed, 0u, cols, bypass);
  HEXKL_PROBE_T0(t0);
  hvx_conv_gate_f32(z, C, g, C, 2u, 1u, C, cw, NULL);
  hvx_quant_rows_u8_params(z, 1u, 4u, C, rq_scale, rq_zp, NULL);
  hvx_quant_pack_u8_ah_rows(z, NULL, 0u, 4u, C, rq_scale, rq_zp, mid);
  HEXKL_PROBE_ADD(HEXKL_PROBE_REQUANT, t0);
  cb_m1_project(&c, feed, slot_bytes, out_chunk, first, pool, bypass);
  HEXKL_PROBE_T0(t0);
  hexkl_moe_dma_copy(out, yc, (size_t)N * 4u, 1, 0);
  hexkl_moe_dma_copy(state, g + C, (size_t)C * 8u, 1, 0);
  HEXKL_PROBE_ADD(HEXKL_PROBE_SCATTER, t0);
  return AEE_SUCCESS;
}

int hexkl_conv_block_run(hexkl_weight_u8i4_table *tbl, uint8_t *vtcm_base,
                         uint32_t vtcm_size, uint32_t config_off, uint32_t M,
                         uint32_t K, uint32_t C, uint32_t N_out, uint32_t h_a,
                         uint32_t h_b, uint32_t h_c, uint32_t h_out,
                         const float *conv_w, const float *act_f32,
                         float *out_f32, float *state_f32,
                         hvx_worker_pool *pool, hexkl_moe_scratch *scratch,
                         int weight_src_bypass, uint32_t flags) {
  if (!tbl || !vtcm_base || !conv_w || !act_f32 || !out_f32 || !state_f32 ||
      !scratch || M == 0u) {
    return AEE_EBADPARM;
  }
  const uint32_t arena = vtcm_size < config_off ? vtcm_size : config_off;
  if (K == 0u || C == 0u || N_out == 0u || K % 32u || C % 32u || N_out % 32u) {
    return AEE_EBADPARM;
  }
  const hexkl_weight_u8i4 *wa = cb_weight(tbl, h_a, K, C);
  const hexkl_weight_u8i4 *wb = cb_weight(tbl, h_b, K, C);
  const hexkl_weight_u8i4 *wc = cb_weight(tbl, h_c, K, C);
  const hexkl_weight_u8i4 *wo = cb_weight(tbl, h_out, C, N_out);
  if (!wa || !wb || !wc || !wo) {
    return AEE_EBADPARM;
  }
  if (M == 1u && (flags & HTP_CONV_FLAG_M1_HVX) && K <= 8192u && C <= 8192u &&
      N_out <= 8192u) {
    return cb_m1_run(vtcm_base, arena, K, C, N_out, wa, wb, wc, wo, conv_w,
                     act_f32, out_f32, state_f32, pool, scratch,
                     weight_src_bypass, flags);
  }
  hexkl_conv_block_layout L;
  int rc = hexkl_conv_block_layout_get(K, C, N_out, arena, &L);
  if (rc != AEE_SUCCESS) {
    return rc;
  }
  const hexkl_acc_layout *acc = hexkl_acc_layout_get(vtcm_base, L.result_off);
  if (!acc->usable) {
    return AEE_EUNSUPPORTED;
  }
  hexkl_probe_us[HEXKL_PROBE_ACC_STRIDE] = acc->row_stride;

  const uint32_t BR = HEXKL_HMX_INT8_BLOCK_N_ROW;
  const uint32_t k_tiles = K / HEXKL_HMX_INT8_BLOCK_N_INNER;
  const uint32_t c_ntiles = C / HEXKL_HMX_INT8_BLOCK_N_COL;
  const uint32_t c_ktiles = C / HEXKL_HMX_INT8_BLOCK_N_INNER;
  const uint32_t o_ntiles = N_out / HEXKL_HMX_INT8_BLOCK_N_COL;
  const uint32_t half = L.acc_tiles / 2u; /* a/c pairs per batch */
  const uint32_t m_pad = ROUND_UP_U32(M, BR);
  if ((c_ntiles + half - 1u) / half > CB_MAX_CHUNKS ||
      (c_ntiles + L.acc_tiles - 1u) / L.acc_tiles > CB_MAX_CHUNKS ||
      (o_ntiles + L.acc_tiles - 1u) / L.acc_tiles > CB_MAX_CHUNKS) {
    return AEE_EUNSUPPORTED;
  }

  /** Heap scratch, session-lifetime (hexkl_moe_scratch): the cached copy of
     the activation -- m_pad rows, the padding rows zero, since the pack
     reads whole 64-row blocks -- its per-row quant params, its AH tiles,
     g, the cached output, and the pack's done bytes. */
  const size_t sz_mpad_f32 = sizeof(float) * m_pad;
  const size_t sz_act_c = sizeof(float) * (size_t)m_pad * K;
  const size_t sz_act_ah = (size_t)m_pad * K;
  const size_t sz_g = sizeof(float) * ((size_t)M + 2u) * C;
  const size_t sz_out_c = sizeof(float) * (size_t)M * N_out;
  const size_t sz_pack_done = m_pad / CB_PACK_UNIT_ROWS + 1u;
  const size_t sz_rq = sizeof(float) * BR;
  const size_t need = 2u * ROUND_UP_SZ(sz_mpad_f32, CB_SCRATCH_ALIGN) +
                      ROUND_UP_SZ(sz_act_c, CB_SCRATCH_ALIGN) +
                      ROUND_UP_SZ(sz_act_ah, CB_SCRATCH_ALIGN) +
                      ROUND_UP_SZ(sz_g, CB_SCRATCH_ALIGN) +
                      ROUND_UP_SZ(sz_out_c, CB_SCRATCH_ALIGN) +
                      ROUND_UP_SZ(sz_pack_done, CB_SCRATCH_ALIGN) +
                      4u * ROUND_UP_SZ(sz_rq, CB_SCRATCH_ALIGN);
  uint64_t p0 = 0;
  HEXKL_PROBE_T0(p0);
  rc = hexkl_moe_scratch_reserve(scratch, need);
  HEXKL_PROBE_ADD(HEXKL_PROBE_ALLOC, p0);
  if (rc != AEE_SUCCESS) {
    return rc;
  }
  uint8_t *cur = scratch->base;
  float *scale_all = (float *)hexkl_moe_carve(&cur, sz_mpad_f32);
  int32_t *zp_all = (int32_t *)hexkl_moe_carve(&cur, sz_mpad_f32);
  float *act_c = (float *)hexkl_moe_carve(&cur, sz_act_c);
  uint8_t *act_ah = (uint8_t *)hexkl_moe_carve(&cur, sz_act_ah);
  float *g = (float *)hexkl_moe_carve(&cur, sz_g);
  float *out_c = (float *)hexkl_moe_carve(&cur, sz_out_c);
  uint8_t *pack_done = (uint8_t *)hexkl_moe_carve(&cur, sz_pack_done);
  float *rq_scale[2], *zbuf[2];
  int32_t *rq_zp[2];
  uint8_t *midbuf[2];
  for (uint32_t i = 0; i < 2u; ++i) {
    rq_scale[i] = (float *)hexkl_moe_carve(&cur, sz_rq);
    rq_zp[i] = (int32_t *)hexkl_moe_carve(&cur, sz_rq);
    zbuf[i] = (float *)(vtcm_base + L.z_off + i * BR * C * 4u);
    midbuf[i] =
      vtcm_base + L.mid_off + i * c_ktiles * HEXKL_HMX_ACTIVATION_ALIGNMENT;
  }
  const float *const conv_w_v = (const float *)(vtcm_base + L.conv_w_off);

  hvx_bg_job pack_job;
  cb_pack_ctx pack;
  int pack_submitted = 0;
  hvx_dq_mul_job mul_job[2];
  hvx_dq_tiles_job dq_job[2];
  /** Phase 2's background stage, two blocks in flight: jobs and their done
     bytes live here because a job outlives its submit until the wait
     that retires it. */
  cb_stage_ctx stage_ctx[2];
  hvx_bg_job stage_job[2];
  uint8_t stage_done[2][CB_STAGE_UNITS];
  int stage_submitted[2] = {0, 0};
  uint32_t idx_ac[CB_MAX_CHUNKS], idx_b[CB_MAX_CHUNKS], idx_o[CB_MAX_CHUNKS];
  uint32_t act_idx;
  uint64_t mm_t0 = 0, mm_acc0 = 0, mm_dq0 = 0;

  /** In: the FastRPC buffers are uncached (see hexkl_mm_u8i4_moe.c on why
     each is touched exactly once). conv_w goes to VTCM: phase 2 reads it
     once per row. */
  HEXKL_PROBE_T0(p0);
  hexkl_dma_ring_reset();
  hexkl_moe_dma_copy(g, state_f32, 2u * C * sizeof(float), 0, 0);
  hexkl_moe_dma_copy(act_c, act_f32, sizeof(float) * (size_t)M * K, 0, 0);
  hexkl_moe_dma_copy(vtcm_base + L.conv_w_off, conv_w, 3u * C * sizeof(float),
                     0, 1);
  if (m_pad > M) {
    memset(act_c + (size_t)M * K, 0, sizeof(float) * (size_t)(m_pad - M) * K);
  }
  HEXKL_PROBE_ADD(HEXKL_PROBE_ACC_COPY, p0);

  /** Phase 1's weights go out before the quantization so the scan hides
     them: W_a and W_c in paired chunks -- chunk ci is a's n-tile columns
     [ci*half, +half) and c's opposite them, one index each (the c push's,
     the ring retires in order). */
  uint32_t n_ac = 0u;
  for (uint32_t g0 = 0; g0 < c_ntiles; g0 += half) {
    const uint32_t cn = (c_ntiles - g0 < half) ? (c_ntiles - g0) : half;
    (void)hexkl_moe_push_weight_chunk(vtcm_base, L.w_a_off, wa, k_tiles,
                                      c_ntiles, g0, cn, weight_src_bypass);
    idx_ac[n_ac++] = hexkl_moe_push_weight_chunk(
      vtcm_base, L.w_b_off, wc, k_tiles, c_ntiles, g0, cn, weight_src_bypass);
  }

  HEXKL_PROBE_T0(p0);
  pack.act_c = act_c;
  pack.scale = scale_all;
  pack.zp = zp_all;
  pack.act_ah = act_ah;
  pack.M = M;
  pack.K = K;
  pack_job.func = cb_pack_unit;
  pack_job.ctx = &pack;
  pack_job.n_units = m_pad / CB_PACK_UNIT_ROWS;
  pack_job.done = pack_done;
  hvx_worker_pool_submit_bg(pool, &pack_job);
  pack_submitted = 1;
  hvx_worker_pool_wait_bg(pool, &pack_job, CB_PACK_UNITS_THROUGH(0u));
  HEXKL_PROBE_ADD(HEXKL_PROBE_QUANT, p0);
  act_idx =
    hexkl_moe_push_act_block(vtcm_base, L.act_off, act_ah, 0u, K, k_tiles);

  /* ---- phase 1: g = dq(x . W_a) * dq(x . W_c), block by block ---------- */
  /** The last epilogue remains in flight across row blocks. Keep staging
     parity across that boundary: a shape with an odd number of column
     batches must not reuse its last accumulator buffer for the next block. */
  uint32_t ac_batch = 0u;
  for (uint32_t mb = 0; mb < M; mb += BR) {
    const uint32_t m_blk = (M - mb < BR) ? (M - mb) : BR;
    HEXKL_PROBE_COUNT(HEXKL_PROBE_BLOCKS, 1);
    HEXKL_PROBE_T0(p0);
    hexkl_dma_ring_wait(act_idx);
    HEXKL_PROBE_ADD(HEXKL_PROBE_GATHER, p0);

    for (uint32_t g0 = 0, ci = 0; g0 < c_ntiles; g0 += half, ++ci, ++ac_batch) {
      const uint32_t np = (c_ntiles - g0 < half) ? (c_ntiles - g0) : half;
      const uint32_t stage_off =
        L.result_off + (ac_batch & 1u) * L.acc_tiles * ACC_TILE_BYTES;
      if (mb == 0u) {
        HEXKL_PROBE_T0(p0);
        hexkl_dma_ring_wait(idx_ac[ci]);
        HEXKL_PROBE_ADD(HEXKL_PROBE_DRAIN, p0);
        if (ci == 0u) {
          hexkl_probe_us[HEXKL_PROBE_DMA_FIRST] =
            hexkl_probe_us[HEXKL_PROBE_DRAIN];
          HEXKL_PROBE_COUNT(HEXKL_PROBE_DMA_FIRST_KB,
                            (2u * np * k_tiles * WEIGHT_TILE_BYTES_U8I4) >> 10);
        }
      }
      /* Slots [0, np) get a's tiles, [np, 2np) c's, at the same columns. */
      HEXKL_MOE_MM_BEGIN();
      for (uint32_t j = 0; j < 2u * np; ++j) {
        const uint32_t col = g0 + ((j < np) ? j : (j - np));
        const uint32_t w_off = (j < np) ? L.w_a_off : L.w_b_off;
        hexkl_micro_hmx_acc_clear_int32();
        for (uint32_t kt = 0; kt < k_tiles; ++kt) {
          rc = hexkl_micro_hmx_mm_u8i4(
            vtcm_base, L.act_off + kt * HEXKL_HMX_ACTIVATION_ALIGNMENT,
            w_off + (kt * c_ntiles + col) * WEIGHT_TILE_BYTES_U8I4);
          if (rc != AEE_SUCCESS) {
            goto out;
          }
        }
        HEXKL_PROBE_T0(p0);
        rc = hexkl_micro_hmx_acc_read_int32(vtcm_base, config_off,
                                            stage_off + j * ACC_TILE_BYTES);
        HEXKL_PROBE_ADD(HEXKL_PROBE_ACC_READ, p0);
        if (rc != AEE_SUCCESS) {
          goto out;
        }
      }
      HEXKL_MOE_MM_END();

      /** Retire the epilogue that read this staging buffer's partner
         (the previous batch), then launch this batch's straight into g. */
      HEXKL_PROBE_T0(p0);
      hvx_worker_pool_wait(pool);
      HEXKL_PROBE_ADD(HEXKL_PROBE_DEQUANT, p0);
      {
        hvx_dq_mul_job *jb = &mul_job[ac_batch & 1u];
        jb->tiles_base =
          (const uint8_t *)((const int32_t *)(vtcm_base + stage_off) +
                            acc->base);
        jb->tile_stride = ACC_TILE_BYTES;
        jb->n_pairs = np;
        jb->c0 = g0 * HEXKL_HMX_INT8_BLOCK_N_COL;
        jb->row_stride = acc->row_stride;
        jb->m_count = m_blk;
        jb->act_scale = scale_all + mb;
        jb->act_zp = zp_all + mb;
        jb->colsum_a = wa->colsum_w;
        jb->w_scale_a = wa->w_scale;
        jb->bias_a = wa->bias;
        jb->colsum_b = wc->colsum_w;
        jb->w_scale_b = wc->w_scale;
        jb->bias_b = wc->bias;
        jb->dst = g + ((size_t)mb + 2u) * C;
        jb->dst_stride = C;
        hvx_worker_pool_submit(pool, hvx_dq_mul_worker, jb, np);
      }
    }
    /** The activation slot is free once the last acc_read returned; the
       next block rides under this block's last epilogue. */
    if (mb + BR < M) {
      HEXKL_PROBE_T0(p0);
      hvx_worker_pool_wait_bg(pool, &pack_job, CB_PACK_UNITS_THROUGH(mb + BR));
      HEXKL_PROBE_ADD(HEXKL_PROBE_QUANT, p0);
      act_idx = hexkl_moe_push_act_block(vtcm_base, L.act_off, act_ah, mb + BR,
                                         K, k_tiles);
    }
  }
  /* g complete: phase 2 looks two rows back across blocks. */
  HEXKL_PROBE_T0(p0);
  hvx_worker_pool_wait(pool);
  HEXKL_PROBE_ADD(HEXKL_PROBE_DEQUANT, p0);

  /* ---- phase 2: out = (dq(x . W_b) * conv1d(g)) . W_out ---------------- */
  /** Both slots are dead (every HMX read of them returned). Block 0's
     activation first, then W_b into A and W_out into B, chunked so the
     first batch of each starts before the rest has landed. */
  act_idx =
    hexkl_moe_push_act_block(vtcm_base, L.act_off, act_ah, 0u, K, k_tiles);
  const uint32_t n_b =
    cb_push_chunks(vtcm_base, L.w_a_off, wb, k_tiles, c_ntiles, L.acc_tiles,
                   idx_b, weight_src_bypass);
  const uint32_t n_o =
    cb_push_chunks(vtcm_base, L.w_b_off, wo, c_ktiles, o_ntiles, L.acc_tiles,
                   idx_o, weight_src_bypass);
  (void)n_b;
  (void)n_o;

  /** The pipeline, one block ahead: block n's b matmul and dequant into
     z[n&1], then its gate and requant submitted to the background lane;
     block n-1's out_proj from mid[(n-1)&1] issued while they run. Every
     buffer has one writer and one reader a block apart:
       z[i]    written by the dequant of block n, read by its gate and
               requant, both retired before block n+2's dequant writes it
       mid[i]  written by block n's requant, read by block n's out_proj
       rq[i]   written by block n's requant, read by block n's out_proj
               epilogues, retired before block n+2's requant */
  /** Keep accumulator and job parity across b/out and row-block boundaries.
     The last out epilogue may still read the partner buffer when the next
     b batch writes this one. */
  uint32_t bo_batch = 0u;
  uint32_t pending_out_row = 0u, pending_out_rows = 0u;
  for (uint32_t mb = 0, n = 0; mb < M + BR; mb += BR, ++n) {
    const uint32_t cur = n & 1u;
    if (mb < M) {
      const uint32_t m_blk = (M - mb < BR) ? (M - mb) : BR;
      HEXKL_PROBE_T0(p0);
      hexkl_dma_ring_wait(act_idx);
      HEXKL_PROBE_ADD(HEXKL_PROBE_GATHER, p0);

      /* --- b: dequantized into z[cur] --- */
      for (uint32_t nt0 = 0, ci = 0; nt0 < c_ntiles;
           nt0 += L.acc_tiles, ++ci, ++bo_batch) {
        const uint32_t nb =
          (c_ntiles - nt0 < L.acc_tiles) ? (c_ntiles - nt0) : L.acc_tiles;
        const uint32_t stage_off =
          L.result_off + (bo_batch & 1u) * L.acc_tiles * ACC_TILE_BYTES;
        if (mb == 0u) {
          HEXKL_PROBE_T0(p0);
          hexkl_dma_ring_wait(idx_b[ci]);
          HEXKL_PROBE_ADD(HEXKL_PROBE_DRAIN_DN, p0);
        }
        HEXKL_MOE_MM_BEGIN();
        for (uint32_t j = 0; j < nb; ++j) {
          hexkl_micro_hmx_acc_clear_int32();
          for (uint32_t kt = 0; kt < k_tiles; ++kt) {
            rc = hexkl_micro_hmx_mm_u8i4(
              vtcm_base, L.act_off + kt * HEXKL_HMX_ACTIVATION_ALIGNMENT,
              L.w_a_off + (kt * c_ntiles + nt0 + j) * WEIGHT_TILE_BYTES_U8I4);
            if (rc != AEE_SUCCESS) {
              goto out;
            }
          }
          HEXKL_PROBE_T0(p0);
          rc = hexkl_micro_hmx_acc_read_int32(vtcm_base, config_off,
                                              stage_off + j * ACC_TILE_BYTES);
          HEXKL_PROBE_ADD(HEXKL_PROBE_ACC_READ, p0);
          if (rc != AEE_SUCCESS) {
            goto out;
          }
        }
        HEXKL_MOE_MM_END();
        HEXKL_PROBE_T0(p0);
        hvx_worker_pool_wait(pool);
        HEXKL_PROBE_ADD(HEXKL_PROBE_DEQUANT, p0);
        if (pending_out_rows != 0u) {
          /** The previous output block's last writer has now retired. Its
             cached rows are immutable for the rest of this invocation. */
          HEXKL_PROBE_T0(p0);
          cb_push_output(out_f32, out_c, pending_out_row, pending_out_rows,
                         N_out);
          pending_out_rows = 0u;
          HEXKL_PROBE_ADD(HEXKL_PROBE_ACC_COPY, p0);
        }
        {
          hvx_dq_tiles_job *jb = &dq_job[bo_batch & 1u];
          jb->tiles_base =
            (const uint8_t *)((const int32_t *)(vtcm_base + stage_off) +
                              acc->base);
          jb->tile_stride = ACC_TILE_BYTES;
          jb->nt0 = nt0;
          jb->row_stride = acc->row_stride;
          jb->m_count = m_blk;
          jb->act_scale = scale_all + mb;
          jb->act_zp = zp_all + mb;
          jb->colsum_w = wb->colsum_w;
          jb->w_scale = wb->w_scale;
          jb->bias = wb->bias;
          jb->dst_a = zbuf[cur];
          jb->dst_b = NULL;
          jb->split = C;
          jb->dst_stride = C;
          jb->n_tiles = nb;
          hvx_worker_pool_submit(pool, hvx_dq_tiles_worker, jb, nb);
        }
      }
      HEXKL_PROBE_T0(p0);
      hvx_worker_pool_wait(pool);
      HEXKL_PROBE_ADD(HEXKL_PROBE_DEQUANT, p0);

      /** The activation slot is dead until the next b matmul; the next
         block's DMA rides under this block's stage and the previous
         block's out_proj. */
      if (mb + BR < M) {
        act_idx = hexkl_moe_push_act_block(vtcm_base, L.act_off, act_ah,
                                           mb + BR, K, k_tiles);
      }

      /* This block's gate and requant, to the background lane. */
      {
        cb_stage_ctx *c = &stage_ctx[cur];
        c->z = zbuf[cur];
        c->g = g;
        c->conv_w = conv_w_v;
        c->mid = midbuf[cur];
        c->rq_scale = rq_scale[cur];
        c->rq_zp = rq_zp[cur];
        c->mb = mb + 2u;
        c->m_blk = m_blk;
        c->C = C;
        stage_job[cur].func = cb_stage_unit;
        stage_job[cur].ctx = c;
        stage_job[cur].n_units =
          (m_blk + CB_STAGE_UNIT_ROWS - 1u) / CB_STAGE_UNIT_ROWS;
        stage_job[cur].done = stage_done[cur];
        hvx_worker_pool_submit_bg(pool, &stage_job[cur]);
        stage_submitted[cur] = 1;
      }
    }

    /* --- out_proj of the PREVIOUS block, from mid[prev] --- */
    if (mb >= BR) {
      const uint32_t pmb = mb - BR;
      const uint32_t prev = cur ^ 1u;
      const uint32_t pm_blk = (M - pmb < BR) ? (M - pmb) : BR;
      /** Its requant is usually long done; what this wait reads is the
         exposure, filed under REQUANT as the synchronous path's was. */
      HEXKL_PROBE_T0(p0);
      hvx_worker_pool_wait_bg(pool, &stage_job[prev], UINT32_MAX);
      HEXKL_PROBE_ADD(HEXKL_PROBE_REQUANT, p0);
      for (uint32_t nt0 = 0, ci = 0; nt0 < o_ntiles;
           nt0 += L.acc_tiles, ++ci, ++bo_batch) {
        const uint32_t nb =
          (o_ntiles - nt0 < L.acc_tiles) ? (o_ntiles - nt0) : L.acc_tiles;
        const uint32_t stage_off =
          L.result_off + (bo_batch & 1u) * L.acc_tiles * ACC_TILE_BYTES;
        if (pmb == 0u) {
          HEXKL_PROBE_T0(p0);
          hexkl_dma_ring_wait(idx_o[ci]);
          HEXKL_PROBE_ADD(HEXKL_PROBE_DRAIN_DN, p0);
        }
        HEXKL_MOE_MM_BEGIN();
        for (uint32_t j = 0; j < nb; ++j) {
          hexkl_micro_hmx_acc_clear_int32();
          for (uint32_t kt = 0; kt < c_ktiles; ++kt) {
            rc = hexkl_micro_hmx_mm_u8i4(vtcm_base,
                                         (uint32_t)(midbuf[prev] - vtcm_base) +
                                           kt * HEXKL_HMX_ACTIVATION_ALIGNMENT,
                                         L.w_b_off + (kt * o_ntiles + nt0 + j) *
                                                       WEIGHT_TILE_BYTES_U8I4);
            if (rc != AEE_SUCCESS) {
              goto out;
            }
          }
          HEXKL_PROBE_T0(p0);
          rc = hexkl_micro_hmx_acc_read_int32(vtcm_base, config_off,
                                              stage_off + j * ACC_TILE_BYTES);
          HEXKL_PROBE_ADD(HEXKL_PROBE_ACC_READ, p0);
          if (rc != AEE_SUCCESS) {
            goto out;
          }
        }
        HEXKL_MOE_MM_END();
        HEXKL_PROBE_T0(p0);
        hvx_worker_pool_wait(pool);
        HEXKL_PROBE_ADD(HEXKL_PROBE_DEQUANT, p0);
        if (pending_out_rows != 0u) {
          /** The previous output block's last writer has now retired. Its
             cached rows are immutable for the rest of this invocation. */
          HEXKL_PROBE_T0(p0);
          cb_push_output(out_f32, out_c, pending_out_row, pending_out_rows,
                         N_out);
          pending_out_rows = 0u;
          HEXKL_PROBE_ADD(HEXKL_PROBE_ACC_COPY, p0);
        }
        {
          hvx_dq_tiles_job *jb = &dq_job[bo_batch & 1u];
          jb->tiles_base =
            (const uint8_t *)((const int32_t *)(vtcm_base + stage_off) +
                              acc->base);
          jb->tile_stride = ACC_TILE_BYTES;
          jb->nt0 = nt0;
          jb->row_stride = acc->row_stride;
          jb->m_count = pm_blk;
          jb->act_scale = rq_scale[prev];
          jb->act_zp = rq_zp[prev];
          jb->colsum_w = wo->colsum_w;
          jb->w_scale = wo->w_scale;
          jb->bias = wo->bias;
          jb->dst_a = out_c + (size_t)pmb * N_out;
          jb->dst_b = NULL;
          jb->split = N_out;
          jb->dst_stride = N_out;
          jb->n_tiles = nb;
          hvx_worker_pool_submit(pool, hvx_dq_tiles_worker, jb, nb);
        }
      }
      pending_out_row = pmb;
      pending_out_rows = pm_blk;
      /** Leave the last out epilogue under the next b batch's HMX work.
         Its post-HMX wait retires this reader of rq[prev] before the next
         gate/requant can overwrite that row-parameter buffer. */
    }
  }

  /* No next b batch exists to retire the final output epilogue. */
  HEXKL_PROBE_T0(p0);
  hvx_worker_pool_wait(pool);
  HEXKL_PROBE_ADD(HEXKL_PROBE_DEQUANT, p0);

  HEXKL_PROBE_T0(p0);
  cb_push_output(out_f32, out_c, pending_out_row, pending_out_rows, N_out);
  /** Commit the carry only after every stage succeeded. With M=1, the
     first exported row is the previous newest row; chunked prefill uses
     exactly the same recurrence as token-at-a-time decode. */
  hexkl_moe_dma_copy(state_f32, g + (size_t)M * C, 2u * C * sizeof(float), 0,
                     0);
  HEXKL_PROBE_ADD(HEXKL_PROBE_ACC_COPY, p0);

out:
  /* An error path may leave jobs in flight over VTCM this call owns. */
  hvx_worker_pool_wait(pool);
  if (pack_submitted) {
    hvx_worker_pool_wait_bg(pool, &pack_job, UINT32_MAX);
  }
  for (uint32_t i = 0; i < 2u; ++i) {
    if (stage_submitted[i]) {
      hvx_worker_pool_wait_bg(pool, &stage_job[i], UINT32_MAX);
    }
  }
  /** Output copies and weight/activation prefetches may still own their
     source or destination on an error path. Retire them before reuse. */
  hexkl_dma_ring_drain();
  return rc;
}
