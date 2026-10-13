// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Haehun Yang <haehun.yang@ax.samsung.com>
 *
 * @file   hexkl_attn_q.c
 * @date   28 Sep 2026
 * @brief  A8W8 / A8W4 flash attention on HMX over a quantized KV cache
 * @see    https://github.com/nntrainer/nntrainer
 * @author Haehun Yang <haehun.yang@ax.samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * hexkl_attn_f16's FA-2 pipeline with the operands on HMX's 8-bit ports:
 *
 *   Qu     = per-row asymmetric uint8 of Q         (hvx_quant_u8)
 *   S      = (Qu.Kq^T - zp*colsum_k) * s_q*s_k     int32 acc -> f32 -> fp16
 *   P      = online softmax of S                    (shared fp16 routine)
 *   P'_d   = uint8 of P * s_v[k][d], per row, per 32-dim group d
 *   O_d    = a*O_d + s_p[d] * (P'_d.Vq_d)           int32 acc -> f32, on HVX
 *   out    = O / l
 *
 * The V scale is per cache row -- the reduction axis of P.V -- so it
 * cannot be a weight-side dequant constant; folding it into P before P's
 * own quantization is what lets V carry per-token, per-group scales on an
 * int accumulator. Everything the accumulator produces is read in place
 * (hexkl_acc_tile: a plain row-major 64x32 int32 tile on every part
 * measured) and never leaves VTCM until the final store. S / P tiles are
 * the fp16 kernel's layout so its softmax runs unmodified. See
 * docs/backend_guide/htp_backend/21_quantized_attention_plan.md.
 *
 * Pipeline per (kv head, q block), as the fp16 kernel: with block k's
 * scores in S[k&1], the pool runs its softmax while the calling thread
 * lands block k+1's tiles, runs its Q.K^T and dequantizes into S[(k+1)&1];
 * then P' quantization on the pool, and P'.V plus the f32 O update on the
 * calling thread. Q is quantized once per kv head for a chunk of q blocks.
 *
 * VTCM is fast for vectors and slow for the scalar unit: a scalar load or
 * store into VTCM costs tens of cycles and does not pipeline, and the
 * first version of this kernel kept its per-row metadata (scales, zero
 * points, rescale factors) there and gathered Q rows and block scales into
 * VTCM with memcpy -- measured at 60-140 cycles per tile row against ~15
 * of vector work. So: per-row metadata lives in cached heap memory, block
 * scales are read from the registry (DDR) as unaligned vectors with the
 * registry laid out to make them contiguous, Q rows are gathered with
 * vector loads and stores, and the accumulator and tile regions are only
 * ever touched by vector instructions. Loops run four rows per iteration
 * through restrict-qualified pointers.
 *
 * Single-token Q8 decode with GQA=4/hd=64 processes only four real rows.
 * A one-block head pipeline keeps two independent arenas: the caller
 * issues HMX QK/PV, while a worker reconstructs the four live accumulator
 * rows, dequantizes, softmaxes and packs P for the other head. K/V and
 * vector metadata are prefetched into alternating banks. SDK accumulator
 * configuration is checked before replacing its full 64-row int32 zip
 * with byte planes; unsupported configurations retain the SDK reader.
 * Longer contexts retain the KV-block pipeline with four-row epilogues.

 */

#include "hexkl_attn_q.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include <AEEStdErr.h>
#include <HAP_perf.h>

#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>

#include "hexkl_acc_tile.h"
#include "hexkl_cvt.h"
#include "hexkl_dma_ring.h"
#include "hexkl_hmx_mm.h"
#include "hexkl_micro.h"
#include "hvx_attn_softmax_f16.h"
#include "hvx_convert.h"
#include "hvx_quant_u8.h"
#include "hvx_tile_f16.h"

#define TB HEXKL_ATTN_TILE_BYTES
#define LOG2E 1.4426950408889634f
/** @brief Row cap; bounds the lane-mask arithmetic, not any buffer. */
#define MAX_G_BR 256u
/** @brief Vectors per 2 KiB tile. */
#define TILE_VECS (TB / sizeof(HVX_Vector))
/** @brief VTCM the Q staging (f32 rows + uint8 tiles) may take per head
 *         chunk; the kernel divides it by the per-q-block cost. */
#define Q_CHUNK_BUDGET (1536u * 1024u)

static int map_plan_rc(int rc) {
  switch (rc) {
  case HEXKL_ATTN_OK:
    return AEE_SUCCESS;
  case HEXKL_ATTN_ENOMEM:
    return AEE_ENOMEMORY;
  default:
    return AEE_EBADPARM;
  }
}

typedef struct {
  uint8_t *vb;
  uint32_t cfg;
  const hexkl_attn_f16_shape *s;
  const hexkl_attn_f16_tiling *t;
  const hexkl_attn_q_io *io;
  const hexkl_attn_q8_input *q8;
  const hexkl_kv_q *kv;
  hvx_worker_pool *pool;
  hexkl_attn_q_layout L;
  const hexkl_acc_layout *acc_layout;
  float scale;         /**< log2e / sqrt(hd), folded into the S dequant */
  float *q_scale;      /**< [chunk*g_br] uint8 Q scale times c->scale */
  int32_t *q_zp;       /**< [chunk*g_br] */
  float *l_f32;        /**< [g_br] running sum */
  float *a_f32;        /**< [g_br] this block's rescale, f32 */
  uint16_t *m_hf;      /**< [g_br] running max, fp16 bits */
  uint16_t *a_hf;      /**< [g_br] this block's rescale, fp16 bits */
  HVX_Vector *p_scale; /**< [g_br] uint8 P' scale, each a splat vector so
                            the O update loads it as is. One per row, not
                            per V group: the registry's V scale covers the
                            whole row, so P' is the same for every group */
  HVX_Vector col_idx;
  HVX_Vector cap_hf, inv_cap_hf;
  int softcap_on;
  uint32_t prefill_acc[2]; /**< Disjoint QK accumulator blocks for workers. */
  uint32_t acc_stage; /**< Optional per-column QK planes for the head worker. */
  uint32_t scale_stage; /**< Double-buffered K scale, colsum and V scale. */
  uint64_t worker_dequant_us;
  uint32_t head_buf; /**< Current KV bank; the other bank may be prefetched. */
  int acc_planes;    /**< Reconstruct only live rows from SDK byte planes. */
  uint32_t
    active_rows; /**< Real decode rows, or the full padded prefill tile. */
  uint64_t
    worker_pquant_us; /**< Written by the worker, read only after join. */
  uint32_t qi;        /**< q block index within the current chunk */
  /** per-block parameters handed to the pool workers; frozen while a job
      is in flight */
  uint32_t w_n, w_qb, w_kb, w_buf, w_sbuf;
  int w_masked;
  hexkl_attn_q_stats *st;
} attn_ctx;

/* --- addressing ------------------------------------------------------- */

/** @brief Q tile (row tile i, dot tile d) of q block qi of the chunk. */
static inline uint8_t *q_tile(const attn_ctx *c, uint32_t qi, uint32_t i,
                              uint32_t d) {
  return c->vb + c->L.q_ah +
         ((qi * c->L.n_row_tiles + i) * c->L.n_dot_tiles + d) * TB;
}
static inline uint8_t *kt_tile(const attn_ctx *c, uint32_t buf, uint32_t col,
                               uint32_t d) {
  return c->vb + c->L.kt_wh[buf] +
         (col * c->L.n_dot_tiles + d) * c->L.tile_bytes;
}
static inline uint8_t *v_tile(const attn_ctx *c, uint32_t buf, uint32_t col,
                              uint32_t d) {
  return c->vb + c->L.v_wh[buf] +
         (col * c->L.n_dot_tiles + d) * c->L.tile_bytes;
}
/** Contiguous 32-lane metadata vectors beside each bank of KV tiles. */
static inline void *scale_tile(const attn_ctx *c, uint32_t buf, uint32_t field,
                               uint32_t col) {
  return c->vb + c->scale_stage + (buf * 3u + field) * c->t->bc * 4u +
         col * 128u;
}
/** @brief fp16 score tile (i32 in [0, 2*rt)) of S[sbuf]. */
static inline HVX_Vector *s_tile(const attn_ctx *c, uint32_t sbuf, uint32_t i32,
                                 uint32_t col) {
  return (HVX_Vector *)(c->vb + c->L.s_hf[sbuf] +
                        (i32 * c->L.n_col_tiles + col) * TB);
}
/** @brief P' tile (row tile i, column tile col); shared by every group. */
static inline uint8_t *p_tile(const attn_ctx *c, uint32_t i, uint32_t col) {
  return c->vb + c->L.p_ah + (i * c->L.n_col_tiles + col) * TB;
}
/** @brief Row 0 of the accumulator tile; rows are row_stride int32 apart. */
static inline const HVX_Vector *acc_rows(const attn_ctx *c) {
  return (const HVX_Vector *)(c->vb + c->L.acc) +
         c->acc_layout->base / (sizeof(HVX_Vector) / sizeof(int32_t));
}
/** Validate the immutable SDK configuration before bypassing its full-tile
 * byte-plane zip. Copy vectors to cached stack memory for scalar checks.
 */
static int decode_planes_supported(const attn_ctx *c) {
  if (!c->io->decode_flags || c->acc_layout->base != 0u ||
      c->acc_layout->row_stride != 32u ||
      ((uintptr_t)(c->vb + c->cfg) & (HEXKL_CVT_BLOCK_BYTES - 1u)))
    return 0;
  const uint32_t scales[4] = {0x6000u, 0x4000u, 0x2000u, 0u};
  for (uint32_t p = 0; p < 4u; ++p) {
    uint32_t words[256] __attribute__((aligned(128)));
    const HVX_Vector *src =
      (const HVX_Vector *)(c->vb + c->cfg + p * HEXKL_CVT_BLOCK_BYTES);
    for (uint32_t v = 0; v < 8u; ++v)
      ((HVX_Vector *)words)[v] = src[v];
    for (uint32_t w = 0; w < 256u; ++w)
      if (words[w] != (w < 64u ? scales[p] : 0u))
        return 0;
  }
  return 1;
}

static int read_acc(attn_ctx *c) {
  if (!c->acc_planes)
    return hexkl_micro_hmx_acc_read_int32(c->vb, c->cfg, c->L.acc);
  for (uint32_t p = 0; p < 4u; ++p)
    hexkl_cvt_issue(c->vb + c->cfg + p * HEXKL_CVT_BLOCK_BYTES,
                    c->vb + c->L.acc + p * HEXKL_CVT_PLANE_BYTES);
  // Complete all convert writes before the next HMX accumulator clear.
  (void)*(const volatile HVX_Vector *)(c->vb + c->L.acc +
                                       HEXKL_ATTN_Q_ACC_BYTES - 128u);
  return AEE_SUCCESS;
}

/** Fuse SDK byte-plane interleaving into the four-row dequant epilogue. */
static inline void acc_four_rows(const attn_ctx *c, uint32_t r,
                                 HVX_Vector rows[4], int planes) {
  const HVX_Vector *p = acc_rows(c);
  if (!planes) {
    const uint32_t stride = c->acc_layout->row_stride / 32u;
    for (uint32_t j = 0; j < 4u; ++j)
      rows[j] = p[(r + j) * stride];
    return;
  }
  const uint32_t v = r / 4u;
  const HVX_VectorPair low = Q6_W_vshuff_VVR(p[16u + v], p[v], -1);
  const HVX_VectorPair high = Q6_W_vshuff_VVR(p[48u + v], p[32u + v], -1);
  const HVX_VectorPair first =
    Q6_W_vshuff_VVR(Q6_V_lo_W(high), Q6_V_lo_W(low), -2);
  const HVX_VectorPair last =
    Q6_W_vshuff_VVR(Q6_V_hi_W(high), Q6_V_hi_W(low), -2);
  rows[0] = Q6_V_lo_W(first);
  rows[1] = Q6_V_hi_W(first);
  rows[2] = Q6_V_lo_W(last);
  rows[3] = Q6_V_hi_W(last);
}

static inline float *o_row(const attn_ctx *c, uint32_t r) {
  return (float *)(c->vb + c->L.o_f32) + (size_t)r * c->s->head_dim;
}
static inline float *qx_row(const attn_ctx *c, uint32_t r) {
  return (float *)(c->vb + c->L.qx_f32) + (size_t)r * c->s->head_dim;
}

static inline uint64_t now_us(void) { return HAP_perf_get_time_us(); }

static inline void unit_range(uint32_t n, uint32_t n_threads, uint32_t i,
                              uint32_t *lo, uint32_t *hi) {
  *lo = (uint32_t)((uint64_t)n * i / n_threads);
  *hi = (uint32_t)((uint64_t)n * (i + 1) / n_threads);
}

/** @brief Column tiles of the registry that block kb actually covers. */
static inline uint32_t resident_cols(const attn_ctx *c, uint32_t kb) {
  const uint32_t ct = c->L.n_col_tiles;
  const uint32_t c0 = kb * ct;
  if (c0 >= c->kv->n_col_tiles) {
    return 0;
  }
  const uint32_t left = c->kv->n_col_tiles - c0;
  return left < ct ? left : ct;
}

/* --- phases ----------------------------------------------------------- */

/** Each unit owns 32 tile rows and complete metadata vectors. */
typedef struct {
  attn_ctx *c;
  uint32_t n, qb0, n_units;
} q8_pack_ctx;

static void q8_pack_worker(uint32_t n_threads, uint32_t i, void *opaque) {
  const q8_pack_ctx *ctx = (const q8_pack_ctx *)opaque;
  attn_ctx *c = ctx->c;
  const uint32_t g_br = c->t->g_br, hd = c->s->head_dim;
  const uint32_t units_per_qb = g_br / 32u;
  const uint32_t lo = (uint32_t)((uint64_t)ctx->n_units * i / n_threads);
  const uint32_t hi = (uint32_t)((uint64_t)ctx->n_units * (i + 1u) / n_threads);
  for (uint32_t u = lo; u < hi; ++u) {
    const uint32_t qi = u / units_per_qb, r0 = (u % units_per_qb) * 32u;
    float scales[32] __attribute__((aligned(128)));
    int32_t zero_points[32] __attribute__((aligned(128)));
    for (uint32_t rr = 0; rr < 32u; rr += 4u) {
      const uint32_t r = r0 + rr;
      const uint8_t *src[4];
      for (uint32_t j = 0; j < 4u; ++j) {
        uint32_t q, g;
        const int valid = hexkl_attn_f16_tile_row_to_qg(
          c->s, c->t, ctx->qb0 + qi, r + j, &q, &g);
        const size_t row =
          valid ? (size_t)q * c->s->n_head_q + ctx->n * c->t->g + g : 0;
        scales[rr + j] = (valid ? c->q8->scales[row] : 1.0f) * c->scale;
        zero_points[rr + j] = valid ? c->q8->zero_points[row] : 0;
        src[j] = valid ? c->q8->values + row * hd : NULL;
      }
      // Four LFM2 heads form one contiguous 256-byte run. Deinterleave
      // its 32-byte groups directly into the two HMX activation tiles.
      // A padded query falls through to the bounded generic gather.
      if (hd == 64u && c->t->g == 4u && src[0] && src[3]) {
        const HVX_VectorPair packed = Q6_W_vdeal_VVR(
          hvx_tile_load_u(src[0] + 128u), hvx_tile_load_u(src[0]), -32);
        const uint32_t tile_row = r % HEXKL_ATTN_Q_ROWS;
        hvx_tile_store_u(q_tile(c, qi, r / HEXKL_ATTN_Q_ROWS, 0u) +
                           tile_row * 32u,
                         Q6_V_lo_W(packed));
        hvx_tile_store_u(q_tile(c, qi, r / HEXKL_ATTN_Q_ROWS, 1u) +
                           tile_row * 32u,
                         Q6_V_hi_W(packed));
        continue;
      }
      for (uint32_t d = 0; d < c->L.n_dot_tiles; ++d) {
        uint8_t packed[128] __attribute__((aligned(128)));
        for (uint32_t j = 0; j < 4u; ++j) {
          if (src[j])
            memcpy(packed + 32u * j, src[j] + 32u * d, 32u);
          else
            memset(packed + 32u * j, 0, 32u);
        }
        uint8_t *dst = q_tile(c, qi, r / HEXKL_ATTN_Q_ROWS, d) +
                       (r % HEXKL_ATTN_Q_ROWS) * 32u;
        hvx_tile_store_u(dst, hvx_tile_load_u(packed));
      }
    }
    hvx_tile_store_u(c->q_scale + qi * g_br + r0, hvx_tile_load_u(scales));
    hvx_tile_store_u(c->q_zp + qi * g_br + r0, hvx_tile_load_u(zero_points));
  }
}

/**
 * @brief Q rows of kv head n for q blocks [qb0, qb0 + n_qb) -> uint8 AH
 *        tiles with per-row scale and zero point, one quantizer pass.
 *
 * Tile row r of q block qb is query row qb*br + r/G, head n*G + r%G,
 * gathered into a contiguous f32 block so hvx_quant_u8's row quantizer
 * runs as it does for the FC layers (one pool fork per chunk, not per q
 * block); pad rows are zero, so they quantize to 0 with scale 1.
 */
static int phase_qprep(attn_ctx *c, uint32_t n, uint32_t qb0, uint32_t n_qb) {
  const uint32_t hd = c->s->head_dim;
  const uint32_t g_br = c->t->g_br;
  const uint32_t rows = n_qb * g_br;
  const uint32_t dt = c->L.n_dot_tiles;
  if (c->q8 && c->io->decode_flags) {
    // Q already arrives quantized. Gather only the four live rows; padded
    // rows were cleared once for this arena, not once per KV head.
    for (uint32_t r = 0; r < 4u; ++r) {
      const uint32_t h = n * 4u + r;
      c->q_scale[r] = c->q8->scales[h] * c->scale;
      c->q_zp[r] = c->q8->zero_points[h];
    }
    for (uint32_t d = 0; d < dt; ++d) {
      uint8_t packed[128] __attribute__((aligned(128)));
      for (uint32_t r = 0; r < 4u; ++r)
        memcpy(packed + 32u * r, c->q8->values + (n * 4u + r) * hd + 32u * d,
               32u);
      *(HVX_Vector *)q_tile(c, 0u, 0u, d) = hvx_tile_load_u(packed);
    }
    return AEE_SUCCESS;
  }
  if (c->q8) {
    q8_pack_ctx ctx = {c, n, qb0, n_qb * (g_br / 32u)};
    hvx_worker_pool_run(c->pool, q8_pack_worker, &ctx, ctx.n_units);
    return AEE_SUCCESS;
  }
  for (uint32_t qi = 0; qi < n_qb; ++qi) {
    for (uint32_t r = 0; r < g_br; ++r) {
      uint32_t q, g;
      HVX_Vector *restrict dst = (HVX_Vector *)qx_row(c, qi * g_br + r);
      if (hexkl_attn_f16_tile_row_to_qg(c->s, c->t, qb0 + qi, r, &q, &g)) {
        const float *src = c->io->q + (size_t)q * c->io->q_stride +
                           (size_t)(n * c->t->g + g) * hd;
        for (uint32_t d = 0; d < dt; ++d) {
          dst[d] = hvx_tile_load_u(src + 32u * d);
        }
      } else {
        for (uint32_t d = 0; d < dt; ++d) {
          dst[d] = Q6_V_vzero();
        }
      }
    }
  }
  hvx_quant_rows_u8_params(qx_row(c, 0), rows, rows, hd, c->q_scale, c->q_zp,
                           c->pool);
  int rc = hvx_quant_pack_u8_ah(qx_row(c, 0), rows, rows, hd, c->q_scale,
                                c->q_zp, q_tile(c, 0, 0, 0), c->pool);
  if (rc != AEE_SUCCESS) {
    return rc;
  }
  // Fold the logit scale into the dequant multiplier.
  for (uint32_t r = 0; r < rows; ++r) {
    c->q_scale[r] *= c->scale;
  }
  return AEE_SUCCESS;
}

/** @brief Softmax state of one q block: nothing seen, or the sink seen. */
static void softmax_state_init(attn_ctx *c, uint32_t n, uint32_t qb) {
  for (uint32_t r = 0; r < c->active_rows; ++r) {
    uint32_t q, g;
    if (c->io->sinks &&
        hexkl_attn_f16_tile_row_to_qg(c->s, c->t, qb, r, &q, &g)) {
      c->m_hf[r] = hvx_attn_f32_to_hf(c->io->sinks[n * c->t->g + g] * LOG2E);
      c->l_f32[r] = 1.0f;
    } else {
      c->m_hf[r] = (uint16_t)HVX_ATTN_HF_NEG_INF;
      c->l_f32[r] = 0.0f;
    }
  }
}

/**
 * @brief Queues the DMA of the ready tiles of block kb of head n: the
 *        block's tiles are one contiguous run in the registry and in
 *        VTCM, so each of K^T and V is a single descriptor.
 */
static void dma_issue(attn_ctx *c, uint32_t n, uint32_t kb, uint32_t buf) {
  const uint32_t ncol = resident_cols(c, kb);
  if (ncol == 0) {
    return;
  }
  const uint32_t bytes = ncol * c->L.n_dot_tiles * c->L.tile_bytes;
  const size_t src = hexkl_kv_q_tile_off(c->kv, n, kb * c->L.n_col_tiles, 0);
  hexkl_dma_ring_push2d(kt_tile(c, buf, 0, 0), c->kv->kt + src, bytes, bytes,
                        bytes, 1u, /*src_vtcm=*/0, /*dst_vtcm=*/1);
  hexkl_dma_ring_push2d(v_tile(c, buf, 0, 0), c->kv->v + src, bytes, bytes,
                        bytes, 1u, /*src_vtcm=*/0, /*dst_vtcm=*/1);
  if (c->scale_stage) {
    const uint32_t mbytes = ncol * 128u;
    const size_t index = hexkl_kv_q_sk_index(c->kv, n, kb * c->t->bc);
    const void *src_meta[3] = {c->kv->s_k + index, c->kv->colsum_k + index,
                               c->kv->s_v + index};
    for (uint32_t field = 0; field < 3u; ++field)
      hexkl_dma_ring_push2d(scale_tile(c, buf, field, 0u), src_meta[field],
                            mbytes, mbytes, mbytes, 1u, 0, 1);
  }
}

/**
 * @brief Waits for the DMA and zeroes the tiles past the registry's last
 *        column, so HMX multiplies finite zeros there. Vector stores: a
 *        memset into VTCM would go through the scalar unit.
 */
static void dma_land(attn_ctx *c, uint32_t kb, uint32_t buf) {
  hexkl_dma_ring_drain();
  const uint32_t ncol = resident_cols(c, kb);
  const uint32_t ct = c->L.n_col_tiles;
  if (ncol < ct) {
    const size_t bytes =
      (size_t)(ct - ncol) * c->L.n_dot_tiles * c->L.tile_bytes;
    HVX_Vector *restrict kt = (HVX_Vector *)kt_tile(c, buf, ncol, 0);
    HVX_Vector *restrict vt = (HVX_Vector *)v_tile(c, buf, ncol, 0);
    for (size_t v = 0; v < bytes / sizeof(HVX_Vector); ++v) {
      kt[v] = Q6_V_vzero();
      vt[v] = Q6_V_vzero();
    }
  }
}

/**
 * @brief The int32 accumulator of output tile (i, col) -> fp16 score tiles
 *        (2i, col) and (2i+1, col) of S[sbuf], dequantized per row:
 *        (acc - zp*colsum_k) * (s_q*log2e/sqrt(hd)) * s_k.
 *
 * Exact up to the f32 products: |acc| and |zp*colsum| are below 2^24 for
 * head_dim <= 256. Two rows go into one tile vector through the widening
 * narrow (low vector -> even lanes), as the fp16 tile builders do; four
 * rows -- two output vectors -- per iteration.
 */
static inline __attribute__((always_inline)) void
dequant_s_impl(attn_ctx *c, uint32_t n, uint32_t kb, uint32_t i, uint32_t col,
               uint32_t sbuf, int planes) {
  const uint32_t k0 = kb * c->t->bc + 32u * col;
  const float *restrict qs =
    c->q_scale + (size_t)c->qi * c->t->g_br + HEXKL_ATTN_Q_ROWS * i;
  const int32_t *restrict qz =
    c->q_zp + (size_t)c->qi * c->t->g_br + HEXKL_ATTN_Q_ROWS * i;
  HVX_Vector sk, cs;
  if (k0 + 32u <= c->kv->max_rows) {
    const void *sk_src = c->scale_stage
                           ? scale_tile(c, c->head_buf, 0u, col)
                           : c->kv->s_k + hexkl_kv_q_sk_index(c->kv, n, k0);
    const void *cs_src =
      c->scale_stage ? scale_tile(c, c->head_buf, 1u, col)
                     : c->kv->colsum_k + hexkl_kv_q_sk_index(c->kv, n, k0);
    sk = hvx_tile_load_u(sk_src);
    cs = Q6_Vsf_equals_Vw(hvx_tile_load_u(cs_src));
  } else {
    // Past the registry: the tiles were zeroed, the lanes will be masked;
    // keep the arithmetic finite.
    sk = Q6_V_vzero();
    cs = Q6_V_vzero();
  }
  for (uint32_t h = 0; h < 2u; ++h) {
    HVX_Vector *restrict dst = s_tile(c, sbuf, 2u * i + h, col);
    const uint32_t r0 = 32u * h;
    for (uint32_t v = 0; v < TILE_VECS; v += 2u) {
      const uint32_t r = r0 + 2u * v;
      if (HEXKL_ATTN_Q_ROWS * i + r >= c->active_rows)
        break;
      HVX_Vector q[4], raw[4];
      if (planes)
        acc_four_rows(c, r, raw, 1);
      for (uint32_t p = 0; p < 4u; ++p) {
        const HVX_Vector a = Q6_Vsf_equals_Vw(
          planes ? raw[p]
                 : acc_rows(c)[(r + p) * (c->acc_layout->row_stride / 32u)]);
        const HVX_Vector zc =
          Q6_Vsf_vmpy_VsfVsf(Q6_Vsf_equals_Vw(Q6_V_vsplat_R(qz[r + p])), cs);
        const HVX_Vector corr = Q6_Vsf_vsub_VsfVsf(a, zc);
        const HVX_Vector scaled =
          Q6_Vsf_vmpy_VsfVsf(corr, hvx_splat_sf(qs[r + p]));
        q[p] = Q6_Vqf32_vmpy_VsfVsf(scaled, sk);
      }
      dst[v] = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(q[1], q[0]));
      dst[v + 1u] = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(q[3], q[2]));
    }
  }
}

static void dequant_s(attn_ctx *c, uint32_t n, uint32_t kb, uint32_t i,
                      uint32_t col, uint32_t sbuf) {
  if (c->acc_planes)
    dequant_s_impl(c, n, kb, i, col, sbuf, 1);
  else
    dequant_s_impl(c, n, kb, i, col, sbuf, 0);
}

/** @brief S[sbuf] = Q.K^T for the block in @a buf, dequantized. */
static int phase_qk(attn_ctx *c, uint32_t n, uint32_t kb, uint32_t buf,
                    uint32_t sbuf) {
  const uint64_t head_t0 = c->acc_stage ? now_us() : 0;
  const int q4 = c->kv->kind == HEXKL_KV_Q4;
  for (uint32_t i = 0; i < c->L.n_row_tiles; ++i) {
    for (uint32_t col = 0; col < c->L.n_col_tiles; ++col) {
      uint64_t t0 = c->acc_stage ? 0 : now_us();
      hexkl_micro_hmx_acc_clear_int32();
      if (q4) {
        for (uint32_t d = 0; d < c->L.n_dot_tiles; ++d) {
          hexkl_hmx_mm_u8i4(q_tile(c, c->qi, i, d), kt_tile(c, buf, col, d));
        }
      } else {
        for (uint32_t d = 0; d < c->L.n_dot_tiles; ++d) {
          hexkl_hmx_mm_u8i8(q_tile(c, c->qi, i, d), kt_tile(c, buf, col, d));
        }
      }
      const uint32_t acc = c->L.acc;
      if (c->acc_stage)
        c->L.acc = c->acc_stage + col * HEXKL_ATTN_Q_ACC_BYTES;
      int rc;
      if (c->prefill_acc[0]) {
        const uint32_t staged =
          c->prefill_acc[sbuf] +
          (i * c->L.n_col_tiles + col) * HEXKL_ATTN_Q_ACC_BYTES;
        rc = hexkl_micro_hmx_acc_read_int32(c->vb, c->cfg, staged);
      } else {
        rc = read_acc(c);
      }
      c->L.acc = acc;
      if (rc != AEE_SUCCESS) {
        return rc;
      }
      if (!c->acc_stage) {
        uint64_t t1 = now_us();
        c->st->us_qk += t1 - t0;
        if (!c->prefill_acc[0]) {
          dequant_s(c, n, kb, i, col, sbuf);
          c->st->us_dequant += now_us() - t1;
        }
      }
    }
  }
  if (c->acc_stage)
    c->st->us_qk += now_us() - head_t0;
  return AEE_SUCCESS;
}

/** @brief One fp16 row pair of S[w_sbuf] through the shared online softmax. */
static void softmax_row_pair(attn_ctx *c, uint32_t i32, uint32_t v) {
  const uint32_t r0 = 32u * i32 + 2u * v;
  hvx_attn_softmax_pair_args a;
  a.n_col_tiles = c->L.n_col_tiles;
  a.k0 = c->w_kb * c->t->bc;
  a.masked = c->w_masked;
  a.lo_pair = Q6_V_vzero();
  a.hi_pair = Q6_V_vzero();
  if (a.masked) {
    uint32_t lo[2] = {0, 0}, hi[2] = {0xFFFFu, 0xFFFFu};
    for (uint32_t p = 0; p < 2u; ++p) {
      uint32_t q, g;
      if (hexkl_attn_f16_tile_row_to_qg(c->s, c->t, c->w_qb, r0 + p, &q, &g)) {
        hexkl_attn_f16_row_range(c->s, q, &lo[p], &hi[p]);
      }
    }
    a.lo_pair = hvx_attn_splat_pair_hf(lo[0], lo[1]);
    a.hi_pair = hvx_attn_splat_pair_hf(hi[0], hi[1]);
  }
  a.col_idx = c->col_idx;
  a.softcap_on = c->softcap_on;
  a.cap_hf = c->cap_hf;
  a.inv_cap_hf = c->inv_cap_hf;
  hvx_attn_online_softmax_pair(s_tile(c, c->w_sbuf, i32, 0) + v, TILE_VECS, &a,
                               c->m_hf + r0, c->a_hf + r0, c->l_f32 + r0);
}

static void softmax_worker(uint32_t n_threads, uint32_t i, void *ctx_) {
  attn_ctx *c = (attn_ctx *)ctx_;
  const uint32_t n = c->active_rows / 2u;
  uint32_t lo, hi;
  unit_range(n, n_threads, i, &lo, &hi);
  for (uint32_t u = lo; u < hi; ++u) {
    softmax_row_pair(c, u / TILE_VECS, u % TILE_VECS);
  }
}

static void softmax_pquant_worker(uint32_t n_threads, uint32_t i, void *ctx_);

static void softmax_start(attn_ctx *c, uint32_t n, uint32_t qb, uint32_t kb,
                          uint32_t sbuf, int masked) {
  c->w_n = n;
  c->w_qb = qb;
  c->w_kb = kb;
  c->w_sbuf = sbuf;
  c->w_masked = masked;
  const int fused = c->s->n_q >= 5u;
  const uint32_t units = c->active_rows / (fused ? 4u : 2u);
  hvx_worker_pool_submit(
    c->pool, fused ? softmax_pquant_worker : softmax_worker, c, units);
}

/**
 * @brief P' tiles of V group d for the 32 rows of fp16 row tile i32:
 *        P'[r][k] = P[r][k] * s_v[k][d], per-row scale = row max / 255,
 *        uint8 round-to-nearest-even, stored flat into the 64-row tile.
 *
 * Four rows (two fp16 vectors) at a time: their 4 x 32 bytes are one
 * vector store, the same packing hvx_quant_pack_u8_ah uses. The block's
 * V scales are one vector per column tile, staged by dma_land.
 *
 * One pass for the whole block: the registry gives V one scale per row,
 * covering every 32-dim group, so the quantized P' feeds all of the P.V
 * matmuls rather than needing a separate pass per group.
 */
static void pquant_rows(attn_ctx *c, uint32_t i32, uint32_t v_begin,
                        uint32_t v_end) {
  const HVX_Vector one_hf = Q6_Vh_vsplat_R((int)HVX_ATTN_HF_ONE);
  const HVX_Vector zero = Q6_V_vzero();
  const uint32_t ct = c->L.n_col_tiles;
  const uint32_t i64 = i32 / 2u;
  const uint32_t rbase = 32u * (i32 & 1u);
  // Only resident column tiles have scales. Decode blocks can exceed the
  // registry's padding; never load a scale beyond this head's allocation.
  const uint32_t ncol = resident_cols(c, c->w_kb);
  const float *sv =
    c->scale_stage
      ? (const float *)scale_tile(c, c->head_buf, 2u, 0u)
      : c->kv->s_v + hexkl_kv_q_sv_index(c->kv, c->w_n, c->w_kb * c->t->bc);
  const HVX_Vector *restrict s0 = s_tile(c, c->w_sbuf, i32, 0);
  uint8_t *restrict p0 = p_tile(c, i64, 0);
  HVX_Vector *restrict ps = c->p_scale + HEXKL_ATTN_Q_ROWS * i64 + rbase;
  const HVX_Vector inv255 = hvx_splat_sf(1.0f / 255.0f);
  const HVX_Vector one_sf = hvx_splat_sf(1.0f);

  for (uint32_t v = v_begin; v < v_end; v += 2u) {
    if (32u * i32 + 2u * v >= c->active_rows)
      break;
    // Pass 1: row max of P * s_v over the block, rows 2v..2v+3.
    HVX_Vector mx[4] = {zero, zero, zero, zero};
    for (uint32_t col = 0; col < ct; ++col) {
      const HVX_Vector svv =
        col < ncol ? hvx_tile_load_u(sv + 32u * col) : one_sf;
      for (uint32_t j = 0; j < 2u; ++j) {
        const HVX_VectorPair w =
          Q6_Wqf32_vmpy_VhfVhf(s0[col * TILE_VECS + v + j], one_hf);
        mx[2u * j] = Q6_Vsf_vmax_VsfVsf(
          mx[2u * j],
          Q6_Vsf_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(Q6_V_lo_W(w)), svv));
        mx[2u * j + 1u] = Q6_Vsf_vmax_VsfVsf(
          mx[2u * j + 1u],
          Q6_Vsf_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(Q6_V_hi_W(w)), svv));
      }
    }
    // A row whose P' is all zero (fully masked so far) keeps scale 1 and
    // quantizes to zeros; the max lane of any other row maps to 255.
    // All in vectors: the row max is broadcast by the reduction, the
    // positive test is an integer compare (P' >= 0), the reciprocal is
    // Newton's -- no vextract, no scalar divide -- and the scale is
    // stored as the splat vector the O update wants.
    HVX_Vector inv[4];
    for (uint32_t j = 0; j < 4u; ++j) {
      const HVX_Vector m = hvx_attn_max32_sf(mx[j]);
      const HVX_VectorPred pos = Q6_Q_vcmp_gt_VwVw(m, zero);
      const HVX_Vector scale =
        Q6_V_vmux_QVV(pos, Q6_Vsf_vmpy_VsfVsf(m, inv255), one_sf);
      ps[2u * v + j] = scale;
      inv[j] = hvx_attn_recip_pos_f32(scale);
    }
    // Pass 2: quantize and pack.
    for (uint32_t col = 0; col < ct; ++col) {
      const HVX_Vector svv =
        col < ncol ? hvx_tile_load_u(sv + 32u * col) : one_sf;
      HVX_Vector vq[4];
      for (uint32_t j = 0; j < 2u; ++j) {
        const HVX_VectorPair w =
          Q6_Wqf32_vmpy_VhfVhf(s0[col * TILE_VECS + v + j], one_hf);
        const HVX_Vector e =
          Q6_Vsf_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(Q6_V_lo_W(w)), svv);
        const HVX_Vector o =
          Q6_Vsf_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(Q6_V_hi_W(w)), svv);
        vq[2u * j] = hvx_sf_to_w_rne(Q6_Vsf_vmpy_VsfVsf(e, inv[2u * j]));
        vq[2u * j + 1u] =
          hvx_sf_to_w_rne(Q6_Vsf_vmpy_VsfVsf(o, inv[2u * j + 1u]));
      }
      const HVX_Vector h01 = Q6_Vh_vpack_VwVw_sat(vq[1], vq[0]);
      const HVX_Vector h23 = Q6_Vh_vpack_VwVw_sat(vq[3], vq[2]);
      *(HVX_Vector *)(p0 + col * TB + (rbase + 2u * v) * 32u) =
        Q6_Vub_vpack_VhVh_sat(h23, h01);
    }
  }
}

/** Reconstruct one worker's four score rows from a completed QK block.
 * The caller drains the other accumulator bank while these rows are read.
 * Every floating-point conversion and product matches dequant_s_impl.
 */
static void dequant_prefill_four(attn_ctx *c, uint32_t r) {
  const uint32_t i = r / HEXKL_ATTN_Q_ROWS;
  const uint32_t row = r % HEXKL_ATTN_Q_ROWS;
  const float *restrict qs = c->q_scale + (size_t)c->qi * c->t->g_br + r;
  const int32_t *restrict qz = c->q_zp + (size_t)c->qi * c->t->g_br + r;
  for (uint32_t col = 0; col < c->L.n_col_tiles; ++col) {
    const uint32_t k0 = c->w_kb * c->t->bc + 32u * col;
    HVX_Vector sk = Q6_V_vzero(), cs = Q6_V_vzero();
    if (k0 + 32u <= c->kv->max_rows) {
      const size_t index = hexkl_kv_q_sk_index(c->kv, c->w_n, k0);
      sk = hvx_tile_load_u(c->kv->s_k + index);
      cs = Q6_Vsf_equals_Vw(hvx_tile_load_u(c->kv->colsum_k + index));
    }
    const uint32_t staged =
      c->prefill_acc[c->w_sbuf] +
      (i * c->L.n_col_tiles + col) * HEXKL_ATTN_Q_ACC_BYTES;
    const HVX_Vector *restrict source = (const HVX_Vector *)(c->vb + staged) +
                                        c->acc_layout->base / 32u +
                                        row * (c->acc_layout->row_stride / 32u);
    HVX_Vector q[4];
    for (uint32_t j = 0; j < 4u; ++j) {
      const HVX_Vector a =
        Q6_Vsf_equals_Vw(source[j * (c->acc_layout->row_stride / 32u)]);
      const HVX_Vector zc =
        Q6_Vsf_vmpy_VsfVsf(Q6_Vsf_equals_Vw(Q6_V_vsplat_R(qz[j])), cs);
      const HVX_Vector corr = Q6_Vsf_vsub_VsfVsf(a, zc);
      const HVX_Vector scaled = Q6_Vsf_vmpy_VsfVsf(corr, hvx_splat_sf(qs[j]));
      q[j] = Q6_Vqf32_vmpy_VsfVsf(scaled, sk);
    }
    HVX_Vector *restrict dst =
      s_tile(c, c->w_sbuf, r / 32u, col) + (r % 32u) / 2u;
    dst[0] = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(q[1], q[0]));
    dst[1] = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(q[3], q[2]));
  }
}

/** Four rows own complete score and P vectors. Their softmax, rescale
 * conversion and P quantization can finish in the same worker while HMX
 * computes the next block's QK. No other worker reads these rows before
 * the caller joins, and each row keeps the original arithmetic order.
 */
static void softmax_pquant_worker(uint32_t n_threads, uint32_t i, void *ctx_) {
  attn_ctx *c = (attn_ctx *)ctx_;
  uint32_t lo, hi;
  unit_range(c->active_rows / 4u, n_threads, i, &lo, &hi);
  for (uint32_t u = lo; u < hi; ++u) {
    const uint32_t i32 = u / (TILE_VECS / 2u);
    const uint32_t v = (u % (TILE_VECS / 2u)) * 2u;
    if (c->prefill_acc[0])
      dequant_prefill_four(c, 4u * u);
    softmax_row_pair(c, i32, v);
    softmax_row_pair(c, i32, v + 1u);
    for (uint32_t r = 4u * u; r < 4u * u + 4u; ++r)
      c->a_f32[r] = hvx_attn_hf_to_f32(c->a_hf[r]);
    pquant_rows(c, i32, v, v + 2u);
  }
}

static void pquant_worker(uint32_t n_threads, uint32_t i, void *ctx_) {
  attn_ctx *c = (attn_ctx *)ctx_;
  const uint32_t n = (c->active_rows + 31u) / 32u;
  uint32_t lo, hi;
  unit_range(n, n_threads, i, &lo, &hi);
  for (uint32_t u = lo; u < hi; ++u) {
    pquant_rows(c, u, 0u, TILE_VECS);
  }
}

/** @brief Joins the softmax (exposed time), then P -> P' across the pool. */
static void softmax_finish_pquant(attn_ctx *c, uint32_t buf) {
  uint64_t t0 = now_us();
  hvx_worker_pool_wait(c->pool);
  uint64_t t1 = now_us();
  c->st->us_softmax += t1 - t0;
  // Prefill's worker already emitted P; its exposed wait is included in
  // us_softmax. There is no separate P-quantization fork or wait to time.
  if (c->s->n_q >= 5u)
    return;
  for (uint32_t r = 0; r < c->active_rows; ++r) {
    c->a_f32[r] = hvx_attn_hf_to_f32(c->a_hf[r]);
  }
  c->w_buf = buf;
  hvx_worker_pool_run(c->pool, pquant_worker, c, (c->active_rows + 31u) / 32u);
  c->st->us_pquant += now_us() - t1;
}

/**
 * @brief O_d[rows of tile i] = a*O_d + s_p[d] * acc, four rows per
 *        iteration; @a first assigns instead (O has no past yet).
 */
static inline __attribute__((always_inline)) void
o_update_impl(attn_ctx *c, uint32_t i, uint32_t d, int first, int planes) {
  const uint32_t hd = c->s->head_dim;
  const uint32_t R0 = HEXKL_ATTN_Q_ROWS * i;
  const HVX_Vector *restrict ps = c->p_scale + R0;
  const float *restrict af = c->a_f32 + R0;
  HVX_Vector *restrict o =
    (HVX_Vector *)((float *)(c->vb + c->L.o_f32) + (size_t)R0 * hd + 32u * d);
  const uint32_t ostride = hd / 32u; /* vectors per O row */
  for (uint32_t r = 0; r < HEXKL_ATTN_Q_ROWS && R0 + r < c->active_rows;
       r += 4u) {
    HVX_Vector t[4], raw[4];
    if (planes)
      acc_four_rows(c, r, raw, 1);
    for (uint32_t p = 0; p < 4u; ++p) {
      t[p] = Q6_Vsf_vmpy_VsfVsf(
        Q6_Vsf_equals_Vw(
          planes ? raw[p]
                 : acc_rows(c)[(r + p) * (c->acc_layout->row_stride / 32u)]),
        ps[r + p]);
    }
    if (!first) {
      for (uint32_t p = 0; p < 4u; ++p) {
        t[p] =
          Q6_Vsf_vadd_VsfVsf(t[p], Q6_Vsf_vmpy_VsfVsf(o[(r + p) * ostride],
                                                      hvx_splat_sf(af[r + p])));
      }
    }
    for (uint32_t p = 0; p < 4u; ++p) {
      o[(r + p) * ostride] = t[p];
    }
  }
}

static void o_update(attn_ctx *c, uint32_t i, uint32_t d, int first) {
  if (c->acc_planes)
    o_update_impl(c, i, d, first, 1);
  else
    o_update_impl(c, i, d, first, 0);
}

/**
 * @brief O_d = a*O_d + s_p[d] * (P'_d . V_d) for every output tile, the
 *        rescale and accumulate in f32 on HVX from the int32 accumulator.
 */
static int phase_pv(attn_ctx *c, uint32_t buf, int first) {
  const int q4 = c->kv->kind == HEXKL_KV_Q4;
  for (uint32_t i = 0; i < c->L.n_row_tiles; ++i) {
    for (uint32_t d = 0; d < c->L.n_dot_tiles; ++d) {
      uint64_t t0 = now_us();
      hexkl_micro_hmx_acc_clear_int32();
      if (q4) {
        for (uint32_t col = 0; col < c->L.n_col_tiles; ++col) {
          hexkl_hmx_mm_u8i4(p_tile(c, i, col), v_tile(c, buf, col, d));
        }
      } else {
        for (uint32_t col = 0; col < c->L.n_col_tiles; ++col) {
          hexkl_hmx_mm_u8i8(p_tile(c, i, col), v_tile(c, buf, col, d));
        }
      }
      int rc = read_acc(c);
      if (rc != AEE_SUCCESS) {
        return rc;
      }
      uint64_t t1 = now_us();
      c->st->us_pv += t1 - t0;
      o_update(c, i, d, first);
      c->st->us_oupd += now_us() - t1;
    }
  }
  return AEE_SUCCESS;
}

/** @brief out rows = O / l, f32. */
static void store_rows(attn_ctx *c, uint32_t n, uint32_t qb, uint32_t lo,
                       uint32_t hi) {
  const uint32_t hd = c->s->head_dim;
  for (uint32_t r = lo; r < hi; ++r) {
    uint32_t q, g;
    if (!hexkl_attn_f16_tile_row_to_qg(c->s, c->t, qb, r, &q, &g)) {
      continue;
    }
    const HVX_Vector inv_l = hvx_splat_sf(1.0f / c->l_f32[r]);
    const HVX_Vector *restrict o = (const HVX_Vector *)o_row(c, r);
    float *dst = c->io->out + (size_t)q * c->io->out_stride +
                 (size_t)(n * c->t->g + g) * hd;
    for (uint32_t d = 0; d < c->L.n_dot_tiles; ++d) {
      hvx_tile_store_u(dst + 32u * d, Q6_Vsf_vmpy_VsfVsf(o[d], inv_l));
    }
  }
}

typedef struct {
  attn_ctx *c;
  uint32_t n, qb;
} store_ctx;

static void store_worker(uint32_t n_threads, uint32_t i, void *opaque) {
  const store_ctx *s = opaque;
  uint32_t lo, hi;
  unit_range(s->c->active_rows, n_threads, i, &lo, &hi);
  store_rows(s->c, s->n, s->qb, lo, hi);
}

static void phase_store(attn_ctx *c, uint32_t n, uint32_t qb) {
  if (c->active_rows >= 128u && c->s->n_q >= 5u) {
    store_ctx ctx = {c, n, qb};
    hvx_worker_pool_run(c->pool, store_worker, &ctx, c->active_rows);
  } else {
    store_rows(c, n, qb, 0u, c->active_rows);
  }
}

/* --- the block pipeline for one (kv head, query block) ---------------- */

static int run_q_block(attn_ctx *c, uint32_t n, uint32_t qb) {
  hexkl_attn_q_stats *st = c->st;
  uint32_t kb_lo, kb_hi;
  hexkl_attn_f16_kv_block_range(c->s, c->t, qb, &kb_lo, &kb_hi);
  if (kb_lo >= kb_hi) {
    return AEE_SUCCESS;
  }
  softmax_state_init(c, n, qb);

  uint64_t t0 = now_us();
  dma_issue(c, n, kb_lo, kb_lo & 1u);
  dma_land(c, kb_lo, kb_lo & 1u);
  st->us_dma += now_us() - t0;
  int rc = phase_qk(c, n, kb_lo, kb_lo & 1u, 0);
  if (rc != AEE_SUCCESS) {
    return rc;
  }

  int first = 1;
  for (uint32_t kb = kb_lo; kb < kb_hi; ++kb) {
    const uint32_t buf = kb & 1u;
    const uint32_t sbuf = (kb - kb_lo) & 1u;
    const int has_next = (kb + 1u < kb_hi);

    t0 = now_us();
    if (has_next) {
      dma_issue(c, n, kb + 1u, buf ^ 1u);
    }
    const int masked = !hexkl_attn_f16_kv_block_unmasked(c->s, c->t, qb, kb);
    softmax_start(c, n, qb, kb, sbuf, masked);
    uint64_t t1 = now_us();
    st->us_softmax += t1 - t0;

    if (has_next) {
      dma_land(c, kb + 1u, buf ^ 1u);
      uint64_t t2 = now_us();
      st->us_dma += t2 - t1;
      rc = phase_qk(c, n, kb + 1u, buf ^ 1u, sbuf ^ 1u);
      if (rc != AEE_SUCCESS) {
        hvx_worker_pool_wait(c->pool);
        return rc;
      }
    }

    // Join before HMX consumes P. Prefill produces it in the same worker
    // as softmax, overlapping both phases with the next block's QK.
    softmax_finish_pquant(c, buf);

    rc = phase_pv(c, buf, first);
    if (rc != AEE_SUCCESS) {
      return rc;
    }
    first = 0;
    st->n_blocks++;
  }

  t0 = now_us();
  phase_store(c, n, qb);
  st->us_store += now_us() - t0;
  return AEE_SUCCESS;
}

/** HMX reads padded activation rows even though epilogues skip them. */
static void clear_decode_padding(attn_ctx *c) {
  HVX_Vector *p = (HVX_Vector *)(c->vb + c->L.p_ah);
  for (uint32_t v = 0; v < c->L.n_row_tiles * c->L.n_col_tiles * TILE_VECS; ++v)
    p[v] = Q6_V_vzero();
  HVX_Vector *q = (HVX_Vector *)(c->vb + c->L.q_ah);
  for (uint32_t v = 0; v < c->L.n_row_tiles * c->L.n_dot_tiles * TILE_VECS; ++v)
    q[v] = Q6_V_vzero();
}

/** Per-head metadata is disjoint while the other head's worker is active. */
static void map_meta(attn_ctx *c, uint8_t *meta_heap) {
  const uint32_t g_br = c->t->g_br, chunk = c->L.n_qb_chunk;
  uint8_t *meta = meta_heap;
  c->q_scale = (float *)meta;
  c->q_zp = (int32_t *)(meta + 4u * chunk * g_br);
  meta += 8u * chunk * g_br;
  c->l_f32 = (float *)meta;
  c->a_f32 = (float *)(meta + 4u * g_br);
  c->m_hf = (uint16_t *)(meta + 8u * g_br);
  c->a_hf = (uint16_t *)(meta + 10u * g_br);
  // The splat-vector table needs vector alignment; the +128 in the
  // allocation pays for rounding up here.
  c->p_scale =
    (HVX_Vector *)(((uintptr_t)(meta + 12u * g_br) + 127u) & ~(uintptr_t)127u);
}

/** Keep HMX on the caller; its peer dequantizes, softmaxes and packs P. */
static void decode_soft_p_worker(uint32_t n_threads, uint32_t i, void *opaque) {
  attn_ctx *c = (attn_ctx *)opaque;
  (void)n_threads;
  (void)i;
  const uint64_t dq0 = now_us();
  if (c->acc_stage) {
    const uint32_t acc = c->L.acc;
    for (uint32_t col = 0; col < c->L.n_col_tiles; ++col) {
      c->L.acc = c->acc_stage + col * HEXKL_ATTN_Q_ACC_BYTES;
      dequant_s(c, c->w_n, 0u, 0u, col, 0u);
    }
    c->L.acc = acc;
  }
  c->worker_dequant_us = now_us() - dq0;
  softmax_worker(1u, 0u, c);
  const uint64_t t0 = now_us();
  for (uint32_t r = 0; r < c->active_rows; ++r)
    c->a_f32[r] = hvx_attn_hf_to_f32(c->a_hf[r]);
  pquant_worker(1u, 0u, c);
  c->worker_pquant_us = now_us() - t0;
}

static void decode_soft_start(attn_ctx *c, uint32_t n) {
  c->w_n = n;
  c->w_qb = c->w_kb = c->w_sbuf = 0;
  c->w_masked = !hexkl_attn_f16_kv_block_unmasked(c->s, c->t, 0, 0);
  const uint64_t t0 = now_us();
  hvx_worker_pool_submit(c->pool, decode_soft_p_worker, c, 1u);
  c->st->us_softmax += now_us() - t0;
}

static void decode_soft_wait(attn_ctx *c) {
  const uint64_t t0 = now_us();
  hvx_worker_pool_wait(c->pool);
  // Exposed wait includes dequant and P packing. Their busy-time counters
  // overlap the caller; phase counters must not be summed as wall time.
  c->st->us_softmax += now_us() - t0;
  c->st->us_pquant += c->worker_pquant_us;
  c->st->us_dequant += c->worker_dequant_us;
}

static int decode_head_qk(attn_ctx *c, uint32_t n) {
  uint64_t t0 = now_us();
  int rc = phase_qprep(c, n, 0u, 1u);
  c->st->us_qprep += now_us() - t0;
  if (rc != AEE_SUCCESS)
    return rc;
  softmax_state_init(c, n, 0u);
  t0 = now_us();
  c->head_buf = (n / 2u) & 1u;
  dma_land(c, 0u, c->head_buf);
  c->st->us_dma += now_us() - t0;
  return phase_qk(c, n, 0u, c->head_buf, 0u);
}

/** One KV block: overlap head n's dequant/softmax/P with head n+1's QK.
 * Two independent arenas keep scores, P, accumulators and metadata live
 * until each consumer finishes. HMX operations are always serialized.
 */
static int run_decode_heads(attn_ctx *first, attn_ctx *second) {
  attn_ctx *slot[2] = {first, second};
  dma_issue(first, 0u, 0u, 0u);
  if (first->s->n_head_kv > 1u)
    dma_issue(second, 1u, 0u, 0u);
  int rc = decode_head_qk(first, 0u);
  if (rc != AEE_SUCCESS)
    return rc;
  decode_soft_start(first, 0u);
  if (first->s->n_head_kv > 1u)
    rc = decode_head_qk(second, 1u);
  decode_soft_wait(first);
  if (rc != AEE_SUCCESS)
    return rc;
  for (uint32_t n = 0; n < first->s->n_head_kv; ++n) {
    attn_ctx *cur = slot[n & 1u], *next = slot[(n + 1u) & 1u];
    const int has_next = n + 1u < first->s->n_head_kv;
    if (has_next)
      decode_soft_start(next, n + 1u);
    // The next use of this arena lands in the other KV bank while HMX
    // consumes the current V bank. Q/metadata are reused only after store.
    if (n + 2u < first->s->n_head_kv)
      dma_issue(cur, n + 2u, 0u, cur->head_buf ^ 1u);
    rc = phase_pv(cur, cur->head_buf, 1);
    if (rc == AEE_SUCCESS) {
      const uint64_t t0 = now_us();
      phase_store(cur, n, 0u);
      cur->st->us_store += now_us() - t0;
      cur->st->n_blocks++;
      if (n + 2u < first->s->n_head_kv)
        rc = decode_head_qk(cur, n + 2u);
    }
    if (has_next)
      decode_soft_wait(next);
    if (rc != AEE_SUCCESS) {
      hexkl_dma_ring_drain();
      return rc;
    }
  }
  return AEE_SUCCESS;
}

/* --- entry ------------------------------------------------------------ */

static int prefill_impl(uint8_t *vtcm_base, uint32_t config_off,
                        const hexkl_attn_f16_shape *s,
                        const hexkl_attn_f16_tiling *t,
                        const hexkl_attn_q_io *io, hvx_worker_pool *pool,
                        hexkl_attn_q_stats *st, const hexkl_attn_q8_input *q8) {
  if (!vtcm_base || !s || !t || !io || (!io->q && !q8) || !io->out || !io->kv) {
    return AEE_EBADPARM;
  }
  const hexkl_kv_q *kv = io->kv;
  if (kv->n_head_kv != s->n_head_kv || kv->head_dim != s->head_dim ||
      kv->max_rows < s->cache_to) {
    return AEE_EBADPARM;
  }
  if (s->cache_to > 0xFFFFu || t->g_br > MAX_G_BR || s->softcap < 0.0f) {
    return AEE_EBADPARM;
  }

  attn_ctx c;
  memset(&c, 0, sizeof(c));
  c.vb = vtcm_base;
  c.cfg = config_off;
  c.s = s;
  c.t = t;
  c.io = io;
  c.q8 = q8;
  c.kv = kv;
  c.pool = pool;
  c.active_rows = t->g_br;
  if (io->decode_flags) {
    // Four-row vector packing is exact for LFM's single-token GQA=4.
    if (!q8 || s->n_q != 1u || t->g != 4u || s->head_dim != 64u ||
        (io->decode_flags != 1u && io->decode_flags != 3u))
      return AEE_EBADPARM;
    c.active_rows = 4u;
  }

  // Q chunk: as many q blocks as the budget allows, at least one.
  const uint32_t n_qb = hexkl_attn_f16_n_q_blocks(s, t);
  const uint32_t per_qb =
    t->g_br * s->head_dim * 4u +
    (t->g_br / HEXKL_ATTN_Q_ROWS) * (s->head_dim / HEXKL_ATTN_TILE) * TB;
  uint32_t chunk = Q_CHUNK_BUDGET / per_qb;
  if (chunk == 0) {
    chunk = 1;
  }
  if (chunk > n_qb) {
    chunk = n_qb;
  }
  int rc = map_plan_rc(
    hexkl_attn_q_plan(s, t, config_off, kv->tile_bytes, chunk, &c.L));
  if (rc == AEE_ENOMEMORY && chunk > 1) {
    chunk = 1;
    rc = map_plan_rc(
      hexkl_attn_q_plan(s, t, config_off, kv->tile_bytes, chunk, &c.L));
  }
  if (rc != AEE_SUCCESS) {
    return rc;
  }
  // The HMX load instructions want every activation tile 2048-byte
  // aligned and every weight tile 128; the plan places them at multiples
  // of 2048 and of tile_bytes, so checking the four region starts here
  // covers every tile and the inline multiplies need no check of their
  // own (hexkl_hmx_mm.h).
  if (!hexkl_hmx_mm_aligned(vtcm_base + c.L.q_ah, vtcm_base + c.L.kt_wh[0]) ||
      !hexkl_hmx_mm_aligned(vtcm_base + c.L.p_ah, vtcm_base + c.L.v_wh[0]) ||
      !hexkl_hmx_mm_aligned(vtcm_base + c.L.q_ah, vtcm_base + c.L.kt_wh[1]) ||
      !hexkl_hmx_mm_aligned(vtcm_base + c.L.p_ah, vtcm_base + c.L.v_wh[1])) {
    return AEE_EBADPARM;
  }

  c.acc_layout = hexkl_acc_layout_get(vtcm_base, c.L.acc);
  if (!c.acc_layout->usable || (c.acc_layout->row_stride % 32u) != 0 ||
      (c.acc_layout->base % 32u) != 0) {
    return AEE_EUNSUPPORTED;
  }
  c.acc_planes = decode_planes_supported(&c);
  c.scale = LOG2E / sqrtf((float)s->head_dim);

  const uint32_t staged_total =
    c.L.total + c.L.n_col_tiles * HEXKL_ATTN_Q_ACC_BYTES +
    hexkl_attn_round_up(6u * t->bc * sizeof(float), TB);
  const int head_pipeline = io->decode_flags == 3u && s->cache_to <= t->bc &&
                            staged_total <= config_off / 2u;
  if (head_pipeline) {
    c.acc_stage = c.L.total;
    c.scale_stage = c.acc_stage + c.L.n_col_tiles * HEXKL_ATTN_Q_ACC_BYTES;
    c.L.total = staged_total;
  }
  // Keep complete QK blocks until their row workers finish. Both banks
  // are disjoint from the scratch accumulator used by PV. If VTCM is
  // tight, retain the validated caller-side dequantization pipeline.
  const uint32_t acc_block_bytes =
    c.L.n_row_tiles * c.L.n_col_tiles * HEXKL_ATTN_Q_ACC_BYTES;
  if (!io->decode_flags && q8 && s->n_q >= 5u && s->head_dim == 64u &&
      t->g == 4u && (uint64_t)c.L.total + 2u * acc_block_bytes <= config_off) {
    c.prefill_acc[0] = c.L.total;
    c.prefill_acc[1] = c.L.total + acc_block_bytes;
    c.L.total += 2u * acc_block_bytes;
  }
  // Per-row metadata in cached heap memory: it is read and written by the
  // scalar unit, which VTCM serves slowly (see the file comment).
  const uint32_t g_br = t->g_br;
  const uint32_t meta_bytes =
    hexkl_attn_q_meta_bytes(g_br, c.L.n_dot_tiles, chunk) + 128u;
  uint8_t *meta_heap =
    (uint8_t *)malloc(meta_bytes * (head_pipeline ? 2u : 1u));
  if (!meta_heap) {
    return AEE_ENOMEMORY;
  }
  map_meta(&c, meta_heap);

  c.col_idx = hvx_attn_col_index_pairs();
  if (s->softcap > 0.0f) {
    const float cap = s->softcap * LOG2E;
    c.cap_hf = Q6_Vh_vsplat_R((int)hvx_attn_f32_to_hf(cap));
    c.inv_cap_hf = Q6_Vh_vsplat_R((int)hvx_attn_f32_to_hf(2.0f * LOG2E / cap));
    c.softcap_on = 1;
  }
  hexkl_dma_ring_reset();

  hexkl_attn_q_stats local;
  memset(&local, 0, sizeof(local));
  c.st = st ? st : &local;
  memset(c.st, 0, sizeof(*c.st));
  const uint64_t call_t0 = now_us();
  const uint64_t call_c0 = HAP_perf_get_pcycles();

  if (io->decode_flags)
    clear_decode_padding(&c);
  if (head_pipeline) {
    attn_ctx peer = c;
    peer.vb += c.L.total;
    peer.cfg -= c.L.total;
    map_meta(&peer, meta_heap + meta_bytes);
    clear_decode_padding(&peer);
    rc = run_decode_heads(&c, &peer);
  } else
    for (uint32_t n = 0; n < s->n_head_kv && rc == AEE_SUCCESS; ++n) {
      for (uint32_t qb0 = 0; qb0 < n_qb && rc == AEE_SUCCESS; qb0 += chunk) {
        const uint32_t nq = (n_qb - qb0 < chunk) ? n_qb - qb0 : chunk;
        const uint64_t t0 = now_us();
        rc = phase_qprep(&c, n, qb0, nq);
        c.st->us_qprep += now_us() - t0;
        for (uint32_t qi = 0; qi < nq && rc == AEE_SUCCESS; ++qi) {
          c.qi = qi;
          rc = run_q_block(&c, n, qb0 + qi);
        }
      }
    }
  c.st->us_total = now_us() - call_t0;
  c.st->pcycles = HAP_perf_get_pcycles() - call_c0;
  free(meta_heap);
  return rc;
}

int hexkl_attn_q_prefill(uint8_t *vtcm_base, uint32_t config_off,
                         const hexkl_attn_f16_shape *s,
                         const hexkl_attn_f16_tiling *t,
                         const hexkl_attn_q_io *io, hvx_worker_pool *pool,
                         hexkl_attn_q_stats *st) {
  return prefill_impl(vtcm_base, config_off, s, t, io, pool, st, NULL);
}

int hexkl_attn_q_prefill_u8(uint8_t *vtcm_base, uint32_t config_off,
                            const hexkl_attn_f16_shape *s,
                            const hexkl_attn_f16_tiling *t,
                            const hexkl_attn_q_io *io,
                            const hexkl_attn_q8_input *q8,
                            hvx_worker_pool *pool, hexkl_attn_q_stats *st) {
  if (!q8 || !q8->values || !q8->scales || !q8->zero_points)
    return AEE_EBADPARM;
  return prefill_impl(vtcm_base, config_off, s, t, io, pool, st, q8);
}
