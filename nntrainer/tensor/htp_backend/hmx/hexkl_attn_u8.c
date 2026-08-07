// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file   hexkl_attn_u8.c
 * @date   07 Aug 2026
 * @brief  KV block registry and the S = Q.Kt half of the fused attention path
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * See hexkl_attn_u8.h for the contract. The host-side executable
 * specification this must agree with is mha_htp_host_forward's PHASE A in
 * test/unittest/mha_htp_host_model.cpp.
 */

#include <stdlib.h>
#include <string.h>

#include <AEEStdErr.h>
#include <HAP_perf.h>

#include "hexkl_attn_u8.h"
#include "hexkl_kv_quant.h"
#include "hvx_softmax_blocked_f32.h"

/** @brief Bytes one fp16 KV row occupies across all heads. */
static uint32_t row_stride(const hexkl_attn_u8_ctx *ctx) {
  return ctx->nch * ctx->head_dim;
}

/**
 * @brief Microsecond clock, read only when a caller asked for the breakdown.
 *
 * HAP_perf_get_qtimer_count is the DSP's own free-running counter, so this
 * measures DSP-internal time and excludes the FastRPC round trip -- which is
 * the point: the host side already times the whole call, and the difference
 * between the two is the transport.
 */
static inline uint64_t now_us(void) {
  return HAP_perf_qtimer_count_to_us(HAP_perf_get_qtimer_count());
}

#define TICK(dst)                                                              \
  do {                                                                         \
    if (stage_us) {                                                            \
      (dst) = now_us();                                                        \
    }                                                                          \
  } while (0)

#define ACCUM(slot, t0, t1)                                                    \
  do {                                                                         \
    if (stage_us) {                                                            \
      stage_us[slot] += (uint32_t)((t1) - (t0));                               \
    }                                                                          \
  } while (0)

uint32_t hexkl_attn_u8_n_blocks(const hexkl_attn_u8_ctx *ctx) {
  return (ctx->kv_len + ctx->T - 1u) / ctx->T;
}

int hexkl_attn_u8_ctx_init(hexkl_attn_u8_ctx *ctx, uint8_t *vtcm_base,
                           uint32_t vtcm_size, uint32_t config_off,
                           uint32_t nch, uint32_t gqa, uint32_t head_dim,
                           uint32_t max_kv, uint32_t T, uint32_t M_band,
                           hexkl_w_width w_k, hexkl_w_width w_v, void *table_k,
                           void *table_v) {
  uint32_t i, n_slots, biggest;

  if (ctx == NULL || nch == 0u || nch > HEXKL_ATTN_MAX_HEADS || gqa == 0u ||
      head_dim == 0u || max_kv == 0u || T == 0u) {
    return AEE_EBADPARM;
  }
  /* hexkl_mm_u8iX_plan enforces K % 32 == 0 and N % 32 == 0; for S those are
   * head_dim and T. Check here rather than letting the first run fail. */
  if ((T % 32u) != 0u || (head_dim % 32u) != 0u) {
    return AEE_EBADPARM;
  }
  /* Property 5: a band's rows must never span more than one extra block once
   * masking restricts them (MHA_HTP_PLAN.md §5.4). Rejecting it here is the
   * only place it can be caught -- violating it misbehaves silently until
   * block skipping lands, one task later. */
  if (M_band == 0u || M_band > T) {
    return AEE_EBADPARM;
  }

  memset(ctx, 0, sizeof(*ctx));
  ctx->vtcm_base = vtcm_base;
  ctx->vtcm_size = vtcm_size;
  ctx->config_off = config_off;
  ctx->nch = nch;
  ctx->gqa = gqa;
  ctx->head_dim = head_dim;
  ctx->max_kv = max_kv;
  ctx->T = T;
  ctx->M_band = M_band;
  ctx->max_blocks = (max_kv + T - 1u) / T;
  ctx->kv_len = 0u;

  if (ctx->max_blocks > HEXKL_ATTN_MAX_BLOCKS) {
    return AEE_EBADPARM;
  }
  if (hexkl_attn_ops_init(&ctx->ops_k, w_k, table_k) != 0 ||
      hexkl_attn_ops_init(&ctx->ops_v, w_v, table_v) != 0) {
    return AEE_EBADPARM;
  }

  n_slots = nch * ctx->max_blocks;
  /* Kt is [head_dim][T], V is [T][head_dim] -- same element count, so one
   * scratch buffer serves both. The scale/colsum vectors differ in length
   * (T vs head_dim); size for the larger. */
  biggest = (T > head_dim) ? T : head_dim;

  ctx->k_shadow =
    (uint16_t *)malloc((size_t)max_kv * row_stride(ctx) * sizeof(uint16_t));
  ctx->v_shadow =
    (uint16_t *)malloc((size_t)max_kv * row_stride(ctx) * sizeof(uint16_t));
  ctx->h_kt = (uint32_t *)malloc((size_t)n_slots * sizeof(uint32_t));
  ctx->h_v = (uint32_t *)malloc((size_t)n_slots * sizeof(uint32_t));
  ctx->q_rm = (int8_t *)malloc((size_t)T * head_dim);
  ctx->q_scale = (float *)malloc((size_t)biggest * sizeof(float));
  ctx->q_colsum = (int32_t *)malloc((size_t)biggest * sizeof(int32_t));
  ctx->q_bias = (float *)calloc((size_t)biggest, sizeof(float));

  ctx->s_band =
    (float *)malloc((size_t)ctx->max_blocks * M_band * T * sizeof(float));
  ctx->o_band = (float *)malloc((size_t)M_band * head_dim * sizeof(float));
  ctx->o_part = (float *)malloc((size_t)M_band * head_dim * sizeof(float));
  ctx->l_row = (float *)malloc((size_t)M_band * sizeof(float));
  ctx->sink_row = (float *)malloc((size_t)M_band * sizeof(float));
  ctx->q_gather = (float *)malloc((size_t)M_band * head_dim * sizeof(float));
  ctx->begin = (uint32_t *)malloc((size_t)M_band * sizeof(uint32_t));
  ctx->end = (uint32_t *)malloc((size_t)M_band * sizeof(uint32_t));
  ctx->seg = (float **)malloc((size_t)ctx->max_blocks * sizeof(float *));

  if (!ctx->k_shadow || !ctx->v_shadow || !ctx->h_kt || !ctx->h_v ||
      !ctx->q_rm || !ctx->q_scale || !ctx->q_colsum || !ctx->q_bias ||
      !ctx->s_band || !ctx->o_band || !ctx->o_part || !ctx->l_row ||
      !ctx->sink_row || !ctx->q_gather || !ctx->begin || !ctx->end ||
      !ctx->seg) {
    hexkl_attn_u8_ctx_fini(ctx);
    return AEE_ENOMEMORY;
  }

  /* Zero-fill the shadows: a tail block's unused rows are quantized along with
   * the real ones, and garbage there is a silent wrong answer that the mask
   * alone will not save (§1.5). */
  memset(ctx->k_shadow, 0, (size_t)max_kv * row_stride(ctx) * sizeof(uint16_t));
  memset(ctx->v_shadow, 0, (size_t)max_kv * row_stride(ctx) * sizeof(uint16_t));
  for (i = 0; i < n_slots; ++i) {
    ctx->h_kt[i] = HEXKL_ATTN_NO_HANDLE;
    ctx->h_v[i] = HEXKL_ATTN_NO_HANDLE;
  }
  return AEE_SUCCESS;
}

void hexkl_attn_u8_ctx_fini(hexkl_attn_u8_ctx *ctx) {
  if (ctx == NULL) {
    return;
  }
  if (ctx->h_kt != NULL || ctx->h_v != NULL) {
    const uint32_t n_slots = ctx->nch * ctx->max_blocks;
    uint32_t i;
    for (i = 0; i < n_slots; ++i) {
      if (ctx->h_kt != NULL && ctx->h_kt[i] != HEXKL_ATTN_NO_HANDLE) {
        (void)ctx->ops_k.rel(ctx->ops_k.table, ctx->h_kt[i]);
      }
      if (ctx->h_v != NULL && ctx->h_v[i] != HEXKL_ATTN_NO_HANDLE) {
        (void)ctx->ops_v.rel(ctx->ops_v.table, ctx->h_v[i]);
      }
    }
  }
  free(ctx->k_shadow);
  free(ctx->v_shadow);
  free(ctx->h_kt);
  free(ctx->h_v);
  free(ctx->q_rm);
  free(ctx->q_scale);
  free(ctx->q_colsum);
  free(ctx->q_bias);
  free(ctx->s_band);
  free(ctx->o_band);
  free(ctx->o_part);
  free(ctx->l_row);
  free(ctx->sink_row);
  free(ctx->q_gather);
  free(ctx->begin);
  free(ctx->end);
  free(ctx->seg);
  memset(ctx, 0, sizeof(*ctx));
}

/** @brief (Re-)registers block @a blk of head @a head, both operands. */
static int register_block(hexkl_attn_u8_ctx *ctx, uint32_t head, uint32_t blk) {
  const uint32_t slot = head * ctx->max_blocks + blk;
  const uint32_t base = blk * ctx->T;
  const uint32_t valid =
    (ctx->kv_len > base)
      ? ((ctx->kv_len - base < ctx->T) ? ctx->kv_len - base : ctx->T)
      : 0u;
  const size_t off = (size_t)base * row_stride(ctx);
  int rc;

  /* Release before re-registering: the slot's scales change when a new token
   * lands in it, and leaking the old handle would exhaust the registry after
   * T appends. */
  if (ctx->h_kt[slot] != HEXKL_ATTN_NO_HANDLE) {
    (void)ctx->ops_k.rel(ctx->ops_k.table, ctx->h_kt[slot]);
    ctx->h_kt[slot] = HEXKL_ATTN_NO_HANDLE;
  }
  if (ctx->h_v[slot] != HEXKL_ATTN_NO_HANDLE) {
    (void)ctx->ops_v.rel(ctx->ops_v.table, ctx->h_v[slot]);
    ctx->h_v[slot] = HEXKL_ATTN_NO_HANDLE;
  }
  if (valid == 0u) {
    return AEE_SUCCESS;
  }

  /* Kt block: logical [head_dim][T], N = kv position, K = head_dim. */
  hexkl_kvq_pack_kt_block(ctx->k_shadow + off, valid, ctx->T, ctx->head_dim,
                          ctx->nch, head, ctx->ops_k.width, ctx->q_rm,
                          ctx->q_scale, ctx->q_colsum);
  rc = ctx->ops_k.reg(ctx->ops_k.table, ctx->vtcm_base, ctx->vtcm_size,
                      ctx->head_dim, ctx->T, ctx->q_rm, ctx->q_scale,
                      ctx->q_colsum, ctx->q_bias, &ctx->h_kt[slot]);
  if (rc != AEE_SUCCESS) {
    return rc;
  }

  /* V block: logical [T][head_dim], N = head_dim column, K = T. */
  hexkl_kvq_pack_v_block(ctx->v_shadow + off, valid, ctx->T, ctx->head_dim,
                         ctx->nch, head, ctx->ops_v.width, ctx->q_rm,
                         ctx->q_scale, ctx->q_colsum);
  return ctx->ops_v.reg(ctx->ops_v.table, ctx->vtcm_base, ctx->vtcm_size,
                        ctx->T, ctx->head_dim, ctx->q_rm, ctx->q_scale,
                        ctx->q_colsum, ctx->q_bias, &ctx->h_v[slot]);
}

int hexkl_attn_u8_kv_append(hexkl_attn_u8_ctx *ctx, uint32_t kv_from,
                            uint32_t n_rows, const uint16_t *k_rows_f16,
                            const uint16_t *v_rows_f16) {
  uint32_t first_blk, last_blk, head, blk;
  size_t off, bytes;

  if (ctx == NULL || k_rows_f16 == NULL || v_rows_f16 == NULL) {
    return AEE_EBADPARM;
  }
  if (n_rows == 0u || kv_from + n_rows > ctx->max_kv) {
    return AEE_EBADPARM;
  }

  off = (size_t)kv_from * row_stride(ctx);
  bytes = (size_t)n_rows * row_stride(ctx) * sizeof(uint16_t);
  memcpy(ctx->k_shadow + off, k_rows_f16, bytes);
  memcpy(ctx->v_shadow + off, v_rows_f16, bytes);

  if (kv_from + n_rows > ctx->kv_len) {
    ctx->kv_len = kv_from + n_rows;
  }

  /* Only the blocks the new rows landed in change. Everything before
   * first_blk keeps the handles it already holds -- which is what makes the
   * append cost independent of kv_len. */
  first_blk = kv_from / ctx->T;
  last_blk = (ctx->kv_len - 1u) / ctx->T;
  for (head = 0; head < ctx->nch; ++head) {
    for (blk = first_blk; blk <= last_blk; ++blk) {
      const int rc = register_block(ctx, head, blk);
      if (rc != AEE_SUCCESS) {
        return rc;
      }
    }
  }
  TICK(t1);
  ACCUM(HEXKL_ATTN_T_TOTAL, t_call0, t1);
  return AEE_SUCCESS;
}

int hexkl_attn_u8_scores(hexkl_attn_u8_ctx *ctx, uint32_t head, uint32_t M,
                         const float *q_band, float *out_s) {
  uint32_t handles[HEXKL_ATTN_MAX_BLOCKS];
  uint32_t n_blocks, blk;

  if (ctx == NULL || q_band == NULL || out_s == NULL || head >= ctx->nch ||
      M == 0u) {
    return AEE_EBADPARM;
  }
  n_blocks = hexkl_attn_u8_n_blocks(ctx);
  if (n_blocks == 0u) {
    return AEE_EBADPARM;
  }

  for (blk = 0; blk < n_blocks; ++blk) {
    const uint32_t h = ctx->h_kt[head * ctx->max_blocks + blk];
    if (h == HEXKL_ATTN_NO_HANDLE) {
      return AEE_EBADSTATE;
    }
    handles[blk] = h;
  }

  /* ONE call. layer_run quantizes q_band once, shares it across every handle,
   * and prefetches block j+1's weight while j computes. out_s comes back
   * block-major, which is the layout PHASE B walks directly. */
  return ctx->ops_k.run(ctx->ops_k.table, ctx->vtcm_base, ctx->vtcm_size,
                        ctx->config_off, M, ctx->head_dim, handles, n_blocks,
                        q_band, out_s);
}

int hexkl_attn_u8_forward(hexkl_attn_u8_ctx *ctx, uint32_t kv_from,
                          uint32_t n_query, float scale, int is_causal,
                          uint32_t window, const float *sink, const float *q,
                          float *out, float *dbg_p, float *dbg_l,
                          uint32_t *stage_us) {
  uint32_t nHq, M, n_blocks, n, b0, m, j, d;
  uint64_t t0 = 0, t1 = 0, t_call0 = 0;
  int rc;

  if (ctx == NULL || q == NULL || out == NULL || n_query == 0u) {
    return AEE_EBADPARM;
  }
  if (kv_from + n_query != ctx->kv_len) {
    /* kv_len is kv_from + n_query by construction: the step rows must already
     * have been appended, so query row i sits at absolute position
     * kv_from + i. Catch the mismatch here rather than producing a plausible
     * wrong answer. */
    return AEE_EBADSTATE;
  }

  if (stage_us) {
    for (n = 0; n < (uint32_t)HEXKL_ATTN_N_STAGES; ++n) {
      stage_us[n] = 0u;
    }
  }
  TICK(t_call0);

  nHq = ctx->nch * ctx->gqa;
  M = n_query * ctx->gqa; /* GQA row fold, per kv_head (doc08 §7) */
  n_blocks = hexkl_attn_u8_n_blocks(ctx);

  for (n = 0; n < ctx->nch; ++n) {
    for (b0 = 0; b0 < M; b0 += ctx->M_band) {
      const uint32_t mb = (M - b0 < ctx->M_band) ? (M - b0) : ctx->M_band;

      /* ---- PHASE A: scores, streaming the K blocks ---------------------- */
      TICK(t0);
      for (m = 0; m < mb; ++m) {
        const uint32_t i = (b0 + m) / ctx->gqa;
        const uint32_t g = (b0 + m) % ctx->gqa;
        memcpy(ctx->q_gather + (size_t)m * ctx->head_dim,
               q + ((size_t)i * nHq + n * ctx->gqa + g) * ctx->head_dim,
               (size_t)ctx->head_dim * sizeof(float));
      }
      TICK(t1);
      ACCUM(HEXKL_ATTN_T_GATHER, t0, t1);

      TICK(t0);
      rc = hexkl_attn_u8_scores(ctx, n, mb, ctx->q_gather, ctx->s_band);
      TICK(t1);
      ACCUM(HEXKL_ATTN_T_QK, t0, t1);
      if (stage_us) {
        stage_us[HEXKL_ATTN_T_CALLS] += 1u;
      }
      if (rc != AEE_SUCCESS) {
        return rc;
      }

      /* ---- PHASE B: one masked softmax pass over the whole band --------- */
      for (m = 0; m < mb; ++m) {
        const uint32_t i = (b0 + m) / ctx->gqa;
        const uint32_t g = (b0 + m) % ctx->gqa;
        const uint32_t p = kv_from + i;
        const uint32_t e = is_causal ? (p + 1u) : ctx->kv_len;
        ctx->end[m] = e;
        ctx->begin[m] = (window != 0u && window < e) ? (e - window) : 0u;
        ctx->sink_row[m] = (sink != NULL) ? sink[n * ctx->gqa + g] : 0.0f;
      }
      for (j = 0; j < n_blocks; ++j) {
        ctx->seg[j] = ctx->s_band + (size_t)j * mb * ctx->T;
      }
      hvx_softmax_blocked_f32(
        ctx->seg, n_blocks, ctx->T, 0u, mb, mb, scale, ctx->begin, ctx->end,
        (sink != NULL) ? ctx->sink_row : NULL, ctx->l_row);

      /* ---- PHASE C: output, streaming the V blocks ---------------------- */
      memset(ctx->o_band, 0, (size_t)mb * ctx->head_dim * sizeof(float));
      for (j = 0; j < n_blocks; ++j) {
        const uint32_t hv = ctx->h_v[n * ctx->max_blocks + j];
        if (hv == HEXKL_ATTN_NO_HANDLE) {
          return AEE_EBADSTATE;
        }
        /* One handle per call, so P is quantized PER BLOCK (property 2):
         * layer_run quantizes act_f32 itself, and after PHASE B this block's
         * rows have a known range in (0, 1] with zp landing at 0 naturally. */
        rc = ctx->ops_v.run(ctx->ops_v.table, ctx->vtcm_base, ctx->vtcm_size,
                            ctx->config_off, mb, ctx->T, &hv, 1u, ctx->seg[j],
                            ctx->o_part);
        if (rc != AEE_SUCCESS) {
          return rc;
        }
        /* Accumulate in f32 across blocks (property 3): each block carries its
         * own dequant constants, so an integer accumulation would apply block
         * 0's scale to every block. */
        for (m = 0; m < mb; ++m) {
          for (d = 0; d < ctx->head_dim; ++d) {
            ctx->o_band[(size_t)m * ctx->head_dim + d] +=
              ctx->o_part[(size_t)m * ctx->head_dim + d];
          }
        }
        TICK(t1);
        ACCUM(HEXKL_ATTN_T_ACCUM, t0, t1);
      }

      /* The 1/l normalization happens ONCE, here, after every block has been
       * accumulated (property 4) -- not as a third softmax pass. */
      for (m = 0; m < mb; ++m) {
        const uint32_t i = (b0 + m) / ctx->gqa;
        const uint32_t g = (b0 + m) % ctx->gqa;
        const float inv_l =
          (ctx->l_row[m] > 0.0f) ? (1.0f / ctx->l_row[m]) : 0.0f;
        float *dst = out + ((size_t)i * nHq + n * ctx->gqa + g) * ctx->head_dim;
        for (d = 0; d < ctx->head_dim; ++d) {
          dst[d] = ctx->o_band[(size_t)m * ctx->head_dim + d] * inv_l;
        }
      }

      TICK(t1);
      ACCUM(HEXKL_ATTN_T_GATHER, t0, t1);

      if (dbg_p != NULL) {
        memcpy(dbg_p, ctx->s_band,
               (size_t)n_blocks * mb * ctx->T * sizeof(float));
      }
      if (dbg_l != NULL) {
        memcpy(dbg_l, ctx->l_row, (size_t)mb * sizeof(float));
      }
    }
  }
  TICK(t1);
  ACCUM(HEXKL_ATTN_T_TOTAL, t_call0, t1);
  return AEE_SUCCESS;
}
