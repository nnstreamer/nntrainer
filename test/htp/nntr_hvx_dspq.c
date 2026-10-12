// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   nntr_hvx_dspq.c
 * @date   28 Sep 2026
 * @brief  [#141] The M==1 MoE layer call over dspqueue: one DSP thread
 *         parked on a queue the ARM side exported, calling the FastRPC
 *         method's own C function (nntr_hvx_mm_u8i4_moe_layer[_timed])
 *         with the packet's arguments
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 *
 * Plan docs/plans/141-dspq-moe.md sections 3.2-3.4; the packet is
 * htp_dspq_wire.h. [#132 Part B E2] The same thread answers
 * HTP_DSPQ_OP_TOKEN, one decode token of the session's token driver role
 * (nntr_hvx_token.c): on S1 it waits for S2's rows on the mailbox page
 * between its rounds, on S2 it runs the token's graph -- both sessions'
 * own threads, not pool lanes (plan 132-part-b section 3.2). Same entry
 * function, same argument bytes, same session state, so the arithmetic is the
 * FastRPC path's bit for bit: the kernel resets its DMA ring per call and the
 * pool splits jobs by index, so which thread is lane 0 does not enter the
 * result. The ARM side keeps at most one DSP entry in flight (its invoke mutex
 * spans write -> response), so this thread and the FastRPC threads never run a
 * kernel at once.
 *
 * Waiting: after each response the thread spins on read_noblock with
 * pause(#255) for spin_us (the decode gaps between two MoE calls), then
 * blocks in dspqueue_read with a 100 ms timeout that re-checks the stop
 * flag. spin_us = 0 always blocks.
 *
 * The dspqueue_* symbols are weak (nntr_hvx_dspq_bench.c's five): a DSP
 * image without dspqueue still loads the skel and start returns
 * AEE_EUNSUPPORTED, which the ARM side turns into its "dspq: off" line.
 *
 * Address space: a 64 KiB thread stack of DSP heap while a queue is open
 * (lane 0 of the M=1 GEMV runs on it; the pool's workers get 32 KiB), and
 * the session's struct nntr_hvx_dspq. Both are freed by stop or close.
 */

#include "nntr_hvx_expert_update.h"
#include <AEEStdErr.h>
#include <HAP_farf.h>
#include <HAP_perf.h>
#include <qurt.h>
#include <remote.h>
#include <stdlib.h>
#include <string.h>

#include "dspqueue.h"
#include "htp_conv_opts.h"
#include "htp_dspq_wire.h"
#include "htp_fc_opts.h"
#include "nntr_hvx.h"
#include "nntr_hvx_session.h"

#pragma weak dspqueue_import
#pragma weak dspqueue_close
#pragma weak dspqueue_write
#pragma weak dspqueue_read
#pragma weak dspqueue_read_noblock

#define DSPQ_STACK (64u * 1024u)
#define DSPQ_BLOCK_TIMEOUT_US 100000u

typedef struct {
  htp_dspq_conv_req binding;
  float *data; /* conv_w[3*C], gamma[K], state[2*C], aligned for DMA. */
  int valid, gamma_valid;
} dspq_conv_cache;

/** @brief The session's queue: its thread and the counters stop returns. */
struct nntr_hvx_dspq {
  nntr_hvx_session *s;
  dspqueue_t q;
  qurt_thread_t tid;
  void *stack;
  uint32_t spin_us;
  volatile int stop;
  uint32_t served, bad, empty;
  dspq_conv_cache conv_cache[HTP_DSPQ_CONV_CACHE_SLOTS];
};

/** @brief The request message, readable as u32 and as f32 (row_weight). */
typedef union {
  uint32_t u[HTP_DSPQ_MAX_MSG / 4];
  float f[HTP_DSPQ_MAX_MSG / 4];
  htp_dspq_req_hdr h;
} dspq_msg;

static inline void dspq_pause(void) {
#if defined(__hexagon__)
  asm volatile(" pause(#255)\n");
#endif
}

static inline uint64_t dspq_now_us(void) {
  return HAP_perf_qtimer_count_to_us(HAP_perf_get_qtimer_count());
}

/** @brief The packet agrees with its own header (section 3.2 step 2); the
 *  kernel's own argument checks run after this, as under FastRPC. */
static int dspq_valid(const dspq_msg *m, uint32_t len, uint32_t nb,
                      const struct dspqueue_buffer *bufs) {
  const htp_dspq_req_hdr *h = &m->h;
  if (len < sizeof(*h))
    return 0;
  if ((h->flags & HTP_DSPQ_FLAG_EXPERT_BATCH_DMA) &&
      !(h->flags & HTP_DSPQ_FLAG_EXPERT_UPDATES))
    return 0;
  const uint64_t base = htp_dspq_req_bytes(h->n_experts, h->n_rows);
  uint64_t expected = base;
  if (h->flags & HTP_DSPQ_FLAG_EXPERT_UPDATES) {
    if (h->M != 1 || base + 4u > len)
      return 0;
    const uint32_t count = m->u[base / 4u];
    if (!count || count > HTP_DSPQ_MAX_EXPERT_UPDATES)
      return 0;
    expected += 4u + (uint64_t)count * sizeof(htp_dspq_expert_update);
  }
  return len >= sizeof(*h) && h->op == HTP_DSPQ_OP_MOE &&
         (h->flags & ~(HTP_DSPQ_FLAG_TIMED | HTP_DSPQ_FLAG_GELU |
                       HTP_DSPQ_FLAG_EXPERT_UPDATES |
                       HTP_DSPQ_FLAG_EXPERT_BATCH_DMA)) == 0 &&
         len == expected && nb == 2 && bufs[0].ptr != NULL &&
         bufs[1].ptr != NULL &&
         bufs[0].size == (uint64_t)h->M * h->K * sizeof(float) &&
         bufs[1].size == (uint64_t)h->M * h->N_out * sizeof(float);
}

/** @brief Runs one valid packet through the FastRPC method's function. */
static int dspq_run(struct nntr_hvx_dspq *d, const dspq_msg *m,
                    const struct dspqueue_buffer *bufs, htp_dspq_resp *resp) {
  const htp_dspq_req_hdr *h = &m->h;
  const uint32_t ne = h->n_experts, nr = h->n_rows;
  const uint32_t o = sizeof(*h) / 4;
  const uint32_t *h_gu = &m->u[o];
  const uint32_t *h_dn = &m->u[o + ne];
  const uint32_t *row_count = &m->u[o + 2 * ne];
  const uint32_t *row_index = &m->u[o + 3 * ne];
  const float *row_weight = &m->f[o + 3 * ne + nr];
  const remote_handle64 hs = (remote_handle64)d->s;
  const int act_len = (int)(h->M * h->K), out_len = (int)(h->M * h->N_out);
  const uint32_t act = (h->flags & HTP_DSPQ_FLAG_GELU) ? 1u : 0u;
  if (h->flags & HTP_DSPQ_FLAG_EXPERT_UPDATES) {
    const uint32_t offset = htp_dspq_req_bytes(ne, nr) / 4u;
    const uint32_t count = m->u[offset];
    const htp_dspq_expert_update *updates =
      (const htp_dspq_expert_update *)&m->u[offset + 1u];
    // Validate stable handles before changing any registry entry. All updates
    // run on the same queue thread as the kernel, including its DMA ring.
    for (uint32_t i = 0; i < count; ++i) {
      const htp_dspq_expert_update *u = updates + i;
      if (u->h_gu == u->h_dn || u->h_gu >= HEXKL_MM_U8I4_MAX_WEIGHTS ||
          u->h_dn >= HEXKL_MM_U8I4_MAX_WEIGHTS)
        return AEE_EBADPARM;
      const hexkl_weight_u8i4 *gu = &d->s->weights_u8i4.slots[u->h_gu];
      const hexkl_weight_u8i4 *dn = &d->s->weights_u8i4.slots[u->h_dn];
      if (!gu->in_use || !dn->in_use || !gu->borrowed || !dn->borrowed ||
          gu->K != h->K || gu->N != 2ull * h->inter || dn->K != h->inter ||
          dn->N != h->N_out)
        return AEE_EBADPARM;
      for (uint32_t j = 0; j < i; ++j)
        if (updates[j].h_gu == u->h_gu || updates[j].h_dn == u->h_gu ||
            updates[j].h_gu == u->h_dn || updates[j].h_dn == u->h_dn)
          return AEE_EBADPARM;
    }
    const uint64_t start = (h->flags & HTP_DSPQ_FLAG_TIMED) ? dspq_now_us() : 0;
    if (h->flags & HTP_DSPQ_FLAG_EXPERT_BATCH_DMA) {
      const int rc = nntr_hvx_expert_update_batch(hs, h->K, h->inter, h->N_out,
                                                  updates, count);
      if (rc != AEE_SUCCESS)
        return rc;
    } else
      for (uint32_t i = 0; i < count; ++i) {
        const htp_dspq_expert_update *u = updates + i;
        uint32_t gu, dn;
        const int rc = nntr_hvx_weight_swap_u2i4_arena(
          hs, u->h_gu, u->h_dn, h->K, h->inter, h->N_out, u->arena, u->off_gu,
          u->off_dn, &gu, &dn);
        if (rc != AEE_SUCCESS)
          return rc; // Never execute with an incompletely updated batch.
        if (gu != u->h_gu || dn != u->h_dn)
          return AEE_EBADSTATE;
      }
    if (h->flags & HTP_DSPQ_FLAG_TIMED)
      resp->expert_update_us = (uint32_t)(dspq_now_us() - start);
  }
  if (h->flags & HTP_DSPQ_FLAG_TIMED) {
    return nntr_hvx_mm_u8i4_moe_layer_timed(
      hs, h->M, h->K, h->inter, h->N_out, act, h_gu, (int)ne, h_dn, (int)ne,
      row_index, (int)nr, row_count, (int)ne, row_weight, (int)nr,
      (const float *)bufs[0].ptr, act_len, (float *)bufs[1].ptr, out_len,
      resp->stage_us, (int)HTP_DSPQ_STAGES);
  }
  return nntr_hvx_mm_u8i4_moe_layer(
    hs, h->M, h->K, h->inter, h->N_out, act, h_gu, (int)ne, h_dn, (int)ne,
    row_index, (int)nr, row_count, (int)ne, row_weight, (int)nr,
    (const float *)bufs[0].ptr, act_len, (float *)bufs[1].ptr, out_len);
}

/** @brief Decode math reuses the exact FastRPC kernel and its validation.
 * Check packed buffer bounds before deriving any pointer. */
static int dspq_math(struct nntr_hvx_dspq *d, const dspq_msg *m, uint32_t len,
                     uint32_t nb, const struct dspqueue_buffer *bufs,
                     htp_dspq_resp *resp) {
  const htp_dspq_math_req *h = (const htp_dspq_math_req *)m;
  const int split = (h->flags & HTP_DSPQ_MATH_AUX_BUFFER) != 0;
  if (len != sizeof(*h) || nb != (split ? 3u : 2u) || !bufs[0].ptr ||
      !bufs[1].ptr || h->M != 1 ||
      (h->flags & ~(HTP_DSPQ_FLAG_TIMED | HTP_DSPQ_MATH_AUX_BUFFER)) ||
      (split && ((h->kind != 2u && h->kind != 3u) || !bufs[2].ptr ||
                 !h->aux_len || 4ull * h->aux_len > bufs[2].size ||
                 bufs[2].size > HTP_DSPQ_BUF_BYTES)) ||
      4u * ((uint64_t)h->x_len + (split ? 0u : h->aux_len) + h->param_len) !=
        bufs[0].size ||
      bufs[0].size > HTP_DSPQ_BUF_BYTES || h->out_bytes != bufs[1].size ||
      h->out_bytes > HTP_DSPQ_BUF_BYTES)
    return AEE_EBADPARM;
  const float *x = (const float *)bufs[0].ptr;
  const float *aux =
    split ? (const float *)bufs[2].ptr : (h->aux_len ? x + h->x_len : NULL);
  const float *param =
    h->param_len ? x + h->x_len + (split ? 0u : h->aux_len) : NULL;
  return nntr_hvx_inference_math_timed_f32(
    (remote_handle64)d->s, h->kind, h->M, h->N, h->chunk, h->eps, h->scale, x,
    (int)h->x_len, aux, (int)h->aux_len, param, (int)h->param_len,
    (uint8_t *)bufs[1].ptr, (int)h->out_bytes, resp->stage_us, 1);
}

static int dspq_router(struct nntr_hvx_dspq *d, const dspq_msg *m, uint32_t len,
                       uint32_t nb, const struct dspqueue_buffer *bufs) {
  const htp_dspq_router_req *h = (const htp_dspq_router_req *)m;
  if (len != sizeof(*h) || nb != 3 || !bufs[0].ptr || !bufs[1].ptr ||
      !bufs[2].ptr || h->flags || !h->E || h->E > 128u || !h->top_k ||
      h->top_k > h->n_sel || h->n_sel > h->E ||
      (h->gamma_len && h->gamma_len != h->K) ||
      4u * ((uint64_t)h->K + h->gamma_len + h->E) != bufs[0].size ||
      4u * ((uint64_t)h->E + h->n_sel + h->top_k) != bufs[1].size ||
      4u * (uint64_t)h->K * h->E != bufs[2].size ||
      bufs[0].size > HTP_DSPQ_BUF_BYTES || bufs[1].size > HTP_DSPQ_BUF_BYTES ||
      bufs[2].size > 0x7fffffffu)
    return AEE_EBADPARM;
  const float *x = (const float *)bufs[0].ptr;
  float *out = (float *)bufs[1].ptr;
  return nntr_hvx_router_sigmoid_f32(
    (remote_handle64)d->s, 1, h->K, h->E, h->eps,
    h->gamma_len ? x + h->K : NULL, h->gamma_len, UINT32_MAX,
    (const float *)bufs[2].ptr, h->K * h->E, x, h->K, h->top_k, h->n_sel,
    x + h->K + h->gamma_len, h->E, out, h->E, (uint32_t *)(out + h->E),
    h->n_sel, out + h->E + h->n_sel, h->top_k);
}

static int dspq_router_norm(struct nntr_hvx_dspq *d, const dspq_msg *m,
                            uint32_t len, uint32_t nb,
                            const struct dspqueue_buffer *bufs) {
  const htp_dspq_router_norm_req *r = (const htp_dspq_router_norm_req *)m;
  const htp_dspq_router_req *h = &r->router;
  if (len != sizeof(*r) || nb != 3 || !bufs[0].ptr || !bufs[1].ptr ||
      !bufs[2].ptr || h->flags || !h->K || h->K > 4096 || h->K % 32 || !h->E ||
      h->E > 128 || !h->top_k || h->top_k > h->n_sel || h->n_sel > h->E ||
      (h->gamma_len && h->gamma_len != h->K) || r->add > 1)
    return AEE_EBADPARM;
  const uint32_t norm_off = (h->E + h->n_sel + h->top_k + 31u) & ~31u;
  const uint32_t input_rows = r->add ? 3u : 2u;
  if (bufs[0].size != 4ull * (h->K * input_rows + h->gamma_len + h->E) ||
      bufs[1].size != 4ull * (norm_off + 2u * h->K) ||
      bufs[2].size != 4ull * h->K * h->E || bufs[0].size > HTP_DSPQ_BUF_BYTES ||
      bufs[1].size > HTP_DSPQ_BUF_BYTES)
    return AEE_EBADPARM;
  const float *x = (const float *)bufs[0].ptr;
  float *out = (float *)bufs[1].ptr;
  float *sum = out + norm_off;
  float *norm = sum + h->K;
  if (!r->add)
    memcpy(sum, x, h->K * sizeof(float));
  const int rc = nntr_hvx_inference_math_f32(
    (remote_handle64)d->s, r->add ? 7u : 0u, 1, h->K, h->K, r->norm_eps, 1.0f,
    x, h->K, r->add ? x + h->K : NULL, r->add ? h->K : 0,
    x + h->K * (input_rows - 1u), h->K, (uint8 *)(r->add ? sum : norm),
    (r->add ? 8u : 4u) * h->K);
  if (rc)
    return rc;
  const float *extra = x + input_rows * h->K;
  return nntr_hvx_router_sigmoid_f32(
    (remote_handle64)d->s, 1, h->K, h->E, h->eps, h->gamma_len ? extra : NULL,
    h->gamma_len, UINT32_MAX, (const float *)bufs[2].ptr, h->K * h->E, norm,
    h->K, h->top_k, h->n_sel, extra + h->gamma_len, h->E, out, h->E,
    (uint32_t *)(out + h->E), h->n_sel, out + h->E + h->n_sel, h->top_k);
}

static int dspq_rope_qkv(struct nntr_hvx_dspq *d, const dspq_msg *m,
                         uint32_t len, uint32_t nb,
                         const struct dspqueue_buffer *bufs) {
  const htp_dspq_rope_qkv_req *h = (const htp_dspq_rope_qkv_req *)m;
  if (len != sizeof(*h) || nb != 2 || !bufs[0].ptr || !bufs[1].ptr ||
      h->flags || !h->head_dim || h->head_dim % 64 || !h->q_width ||
      !h->kv_width || h->q_width % h->head_dim || h->kv_width % h->head_dim ||
      bufs[0].size != 4ull * ((uint64_t)h->q_width + 2ull * h->kv_width +
                              2ull * h->head_dim) ||
      bufs[1].size != 4ull * ((uint64_t)h->q_width + h->kv_width) ||
      bufs[0].size > HTP_DSPQ_BUF_BYTES || bufs[1].size > HTP_DSPQ_BUF_BYTES)
    return AEE_EBADPARM;
  const float *q = (const float *)bufs[0].ptr;
  const float *k = q + h->q_width, *v = k + h->kv_width;
  const float *cs = v + h->kv_width;
  uint8_t *out = (uint8_t *)bufs[1].ptr;
  const remote_handle64 hs = (remote_handle64)d->s;
  int rc = nntr_hvx_inference_math_f32(hs, 4, 1, h->q_width, h->head_dim, 0, 1,
                                       q, h->q_width, NULL, 0, cs,
                                       2u * h->head_dim, out, 4u * h->q_width);
  if (rc)
    return rc;
  rc = nntr_hvx_inference_math_f32(hs, 5, 1, h->kv_width, h->head_dim, 0, 1, k,
                                   h->kv_width, NULL, 0, cs, 2u * h->head_dim,
                                   out + 4u * h->q_width, 2u * h->kv_width);
  if (rc)
    return rc;
  return nntr_hvx_inference_math_f32(
    hs, 6, 1, h->kv_width, h->head_dim, 0, 1, v, h->kv_width, NULL, 0, NULL, 0,
    out + 4u * h->q_width + 2u * h->kv_width, 2u * h->kv_width);
}

/** @brief [#132 Part B E2] Answers one HTP_DSPQ_OP_TOKEN packet: the
 *  session's token driver role on the packet's buffers (S2: 0 the
 *  embedding row, 1 the logits under HTP_DSPQ_TOKEN_LOGITS; S1: none).
 *  A malformed packet is answered AEE_EBADPARM, as a MoE packet is.
 *  @return dspqueue_write's code */
static int dspq_token(struct nntr_hvx_dspq *d, const dspq_msg *m, uint32_t len,
                      uint32_t nb, struct dspqueue_buffer *bufs) {
  const htp_dspq_token_req *q = (const htp_dspq_token_req *)m;
  htp_dspq_token_resp resp;
  uint32_t i;
  const int logits = len == sizeof(*q) && (q->flags & HTP_DSPQ_TOKEN_LOGITS);
  const int valid =
    len == sizeof(*q) && (q->flags & ~HTP_DSPQ_TOKEN_LOGITS) == 0u &&
    nb <= 2u && (nb == 2u) == (logits != 0) &&
    (nb < 1u || (bufs[0].ptr != NULL && bufs[0].size % 4u == 0u)) &&
    (nb < 2u || (bufs[1].ptr != NULL && bufs[1].size % 4u == 0u));
  memset(&resp, 0, sizeof(resp));
  resp.t_in_us = (uint32_t)dspq_now_us();
  resp.seq = len >= 8 ? m->u[1] : 0;
  if (valid) {
    resp.rc = nntr_hvx_token_run(
      d->s, q->seq, q->pos, nb >= 1u ? (const float *)bufs[0].ptr : NULL,
      nb >= 1u ? (uint32_t)(bufs[0].size / 4u) : 0u,
      nb == 2u ? (float *)bufs[1].ptr : NULL,
      nb == 2u ? (uint32_t)(bufs[1].size / 4u) : 0u, &resp);
  } else {
    resp.rc = AEE_EBADPARM;
    ++d->bad;
  }
  for (i = 0; i < nb; ++i) {
    bufs[i].flags = DSPQUEUE_BUFFER_FLAG_DEREF;
  }
  if (nb == 2u) {
    bufs[1].flags |= DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER |
                     DSPQUEUE_BUFFER_FLAG_INVALIDATE_RECIPIENT;
  }
  resp.t_out_us = (uint32_t)dspq_now_us();
  return dspqueue_write(d->q, 0, nb, bufs, sizeof(resp), (const uint8_t *)&resp,
                        DSPQUEUE_TIMEOUT_NONE);
}

static int dspq_conv_call(struct nntr_hvx_dspq *d, const htp_dspq_conv_req *h,
                          const float *input, const float *conv,
                          const float *gamma, float *state, float *output,
                          htp_dspq_resp *resp) {
  const int input_len = (int)((h->norm_kind + 1u) * h->K);
  const int timed = (h->flags & HTP_DSPQ_FLAG_TIMED) != 0u;
  const remote_handle64 hs = (remote_handle64)d->s;
  if (h->norm_kind && gamma) {
    // Same norm kernel/order as conv_block_norm, with separate cached gamma.
    // Avoid rebuilding its packed argument and copying x/residual on the DSP.
    const uint64_t start = timed ? dspq_now_us() : 0;
    const int add = h->norm_kind == 2u;
    float *norm = output + h->N_out;
    if (!add)
      memcpy(norm, input, h->K * sizeof(float));
    int rc = nntr_hvx_inference_math_f32(
      hs, add ? 7u : 0u, 1, h->K, h->K, h->eps, 1.f, input, h->K,
      add ? input + h->K : NULL, add ? h->K : 0u, gamma, h->K,
      (uint8_t *)(add ? norm : norm + h->K), (add ? 8u : 4u) * h->K);
    if (rc)
      return rc;
    const uint32_t norm_us = timed ? (uint32_t)(dspq_now_us() - start) : 0;
    htp_dspq_conv_req plain = *h;
    plain.norm_kind = 0;
    rc =
      dspq_conv_call(d, &plain, norm + h->K, conv, NULL, state, output, resp);
    if (!rc && timed)
      resp->stage_us[0] += norm_us;
    return rc;
  }
  if (h->norm_kind)
    return nntr_hvx_mm_u8i4_conv_block_norm(
      hs, h->K, h->C, h->N_out, h->h_in, 3, h->h_out, conv, 3u * h->C,
      h->norm_kind == 2u, h->eps, input, input_len, output + h->N_out,
      2u * h->K, output, h->N_out, state, 2u * h->C,
      timed ? resp->stage_us : NULL, timed ? HTP_DSPQ_STAGES : 0);
  if (timed)
    return nntr_hvx_mm_u8i4_conv_block_timed(
      hs, 1, h->K, h->C, h->N_out, h->h_in, 3, h->h_out, conv, 3u * h->C, input,
      input_len, output, h->N_out, state, 2u * h->C, resp->stage_us,
      HTP_DSPQ_STAGES);
  return nntr_hvx_mm_u8i4_conv_block(
    hs, 1, h->K, h->C, h->N_out, h->h_in, 3, h->h_out, conv, 3u * h->C, input,
    input_len, output, h->N_out, state, 2u * h->C);
}

/** Keep HMX on its owning FastRPC thread. The admitted HVX kernel uses
 * the same session, DMA ring and worker pool, with one entry in flight. */
static int dspq_conv(struct nntr_hvx_dspq *d, const dspq_msg *m, uint32_t len,
                     uint32_t nb, const struct dspqueue_buffer *bufs,
                     htp_dspq_resp *resp) {
  const htp_dspq_conv_req *h = (const htp_dspq_conv_req *)m;
  if (len != sizeof(*h) || nb != 2u || !bufs[0].ptr || !bufs[1].ptr ||
      !htp_dspq_conv_shape_valid(h) || (h->flags & ~HTP_DSPQ_FLAG_TIMED) ||
      bufs[0].size != htp_dspq_conv_input_bytes(h) ||
      bufs[1].size != htp_dspq_conv_output_bytes(h) ||
      bufs[0].size > HTP_DSPQ_BUF_BYTES || bufs[1].size > HTP_DSPQ_BUF_BYTES)
    return AEE_EBADPARM;
  if (!(d->s->moe_flags & HTP_CONV_FLAG_M1_HVX))
    return AEE_EUNSUPPORTED;
  const float *input = (const float *)bufs[0].ptr;
  const float *conv = input + (h->norm_kind + 1u) * h->K;
  float *output = (float *)bufs[1].ptr;
  float *state = output + h->N_out + (h->norm_kind ? 2u * h->K : 0u);
  memcpy(state, conv + 3u * h->C, 2u * h->C * sizeof(float));
  return dspq_conv_call(d, h, input, conv, NULL, state, output, resp);
}

/** Reuse unchanged inputs without changing Conv's observable state API.
 * All allocation, validation and cache updates run on the queue thread. */
static int dspq_conv_cached(struct nntr_hvx_dspq *d, const dspq_msg *m,
                            uint32_t len, uint32_t nb,
                            const struct dspqueue_buffer *bufs,
                            htp_dspq_resp *resp) {
  const htp_dspq_conv_cached_req *r = (const htp_dspq_conv_cached_req *)m;
  const htp_dspq_conv_req *h = &r->conv;
  if (len != sizeof(*r) || nb != 2u || !bufs[0].ptr || !bufs[1].ptr ||
      !htp_dspq_conv_shape_valid(h) || (h->flags & ~HTP_DSPQ_FLAG_TIMED) ||
      r->slot >= HTP_DSPQ_CONV_CACHE_SLOTS ||
      (r->updates & ~HTP_DSPQ_CONV_UPDATES) ||
      (!h->norm_kind && (r->updates & HTP_DSPQ_CONV_UPDATE_GAMMA)) ||
      bufs[0].size != htp_dspq_conv_cached_input_bytes(r) ||
      bufs[1].size != htp_dspq_conv_output_bytes(h) ||
      bufs[0].size > HTP_DSPQ_BUF_BYTES || bufs[1].size > HTP_DSPQ_BUF_BYTES)
    return AEE_EBADPARM;
  if (!(d->s->moe_flags & HTP_CONV_FLAG_M1_HVX))
    return AEE_EUNSUPPORTED;
  dspq_conv_cache *c = &d->conv_cache[r->slot];
  const int reuse = c->valid && htp_dspq_conv_binding_equal(&c->binding, h);
  const uint32_t required = HTP_DSPQ_CONV_UPDATE_W |
                            HTP_DSPQ_CONV_UPDATE_STATE |
                            (h->norm_kind ? HTP_DSPQ_CONV_UPDATE_GAMMA : 0u);
  if ((!reuse && (r->updates & required) != required) ||
      (h->norm_kind && !(reuse && c->gamma_valid) &&
       !(r->updates & HTP_DSPQ_CONV_UPDATE_GAMMA)))
    return AEE_EBADSTATE;
  if (!c->data || c->binding.K != h->K || c->binding.C != h->C) {
    float *data = (float *)memalign(128, (5u * h->C + h->K) * sizeof(float));
    if (!data)
      return AEE_ENOMEMORY;
    free(c->data);
    c->data = data;
    c->valid = c->gamma_valid = 0;
  }
  if (!reuse)
    c->gamma_valid = 0;
  c->valid = 0; /* A failed kernel cannot leave reusable partial history. */
  c->binding = *h;
  float *conv = c->data, *gamma = conv + 3u * h->C;
  float *state = gamma + h->K;
  const float *input = (const float *)bufs[0].ptr;
  const uint32_t dynamic_len = (h->norm_kind == 2u ? 2u : 1u) * h->K;
  const float *src = input + dynamic_len;
  if (r->updates & HTP_DSPQ_CONV_UPDATE_W) {
    memcpy(conv, src, 3u * h->C * sizeof(float));
    src += 3u * h->C;
  }
  if (r->updates & HTP_DSPQ_CONV_UPDATE_GAMMA) {
    memcpy(gamma, src, h->K * sizeof(float));
    src += h->K;
    c->gamma_valid = 1;
  }
  if (r->updates & HTP_DSPQ_CONV_UPDATE_STATE)
    memcpy(state, src, 2u * h->C * sizeof(float));
  float *output = (float *)bufs[1].ptr;
  const int rc = dspq_conv_call(d, h, input, conv, gamma, state, output, resp);
  if (!rc) {
    memcpy(output + h->N_out + (h->norm_kind ? 2u * h->K : 0u), state,
           2u * h->C * sizeof(float));
    c->valid = 1;
  }
  return rc;
}

/** Only admit HVX FC: HMX remains on its owning FastRPC thread. */
static int dspq_fc(struct nntr_hvx_dspq *d, const dspq_msg *m, uint32_t len,
                   uint32_t nb, const struct dspqueue_buffer *bufs,
                   htp_dspq_resp *resp) {
  const htp_dspq_fc_req *h = (const htp_dspq_fc_req *)m;
  if (len != sizeof(*h) || nb != 2 || !bufs[0].ptr || !bufs[1].ptr ||
      !htp_dspq_fc_shape_valid(h) || (h->flags & ~HTP_DSPQ_FLAG_TIMED) ||
      bufs[0].size != htp_dspq_fc_input_bytes(h) ||
      bufs[1].size != htp_dspq_fc_output_bytes(h) ||
      bufs[0].size > HTP_DSPQ_BUF_BYTES || bufs[1].size > HTP_DSPQ_BUF_BYTES)
    return AEE_EBADPARM;
  if (!(d->s->moe_flags & HTP_FC_FLAG_M1_HVX))
    return AEE_EUNSUPPORTED;
  uint64_t chunks = 0;
  for (unsigned i = 0; i < h->count; ++i)
    chunks += h->chunks[i];
  if (chunks != h->post_len)
    return AEE_EBADPARM;
  const float *input = (const float *)bufs[0].ptr;
  const uint32_t input_len = (h->norm_kind + 1u) * h->K;
  const float *pre = input + input_len, *post = pre + h->pre_len;
  const float *cs = post + h->post_len;
  float *output = (float *)bufs[1].ptr;
  const remote_handle64 hs = (remote_handle64)d->s;
  const int timed = (h->flags & HTP_DSPQ_FLAG_TIMED) != 0;
  if (!h->fused) {
    // FC's first nine slots are the existing timed entry's layout.
    if (timed)
      return nntr_hvx_mm_u8i4_layer_timed(hs, 1, h->K, h->handles, h->count,
                                          input, h->K, output, h->N,
                                          resp->stage_us, 9);
    return nntr_hvx_mm_u8i4_layer(hs, 1, h->K, h->handles, h->count, input,
                                  h->K, output, h->N);
  }
  const uint64_t start = timed ? dspq_now_us() : 0;
  const int rc =
    h->norm_kind ? nntr_hvx_mm_u8i4_layer_norm_ld_input(
                     hs, h->K, h->eps, pre, h->pre_len, h->chunks, h->count,
                     post, h->post_len, h->rope_hd, h->rope_handles, cs,
                     2u * h->rope_hd, h->handles, h->count, h->offsets,
                     h->count, h->strides, h->count, h->norm_kind == 2u,
                     h->norm_eps, input, input_len, output, h->N + 2u * h->K)
                 : nntr_hvx_mm_u8i4_layer_norm_ld(
                     hs, 1, h->K, h->eps, pre, h->pre_len, h->chunks, h->count,
                     post, h->post_len, h->rope_hd, h->rope_handles, cs,
                     2u * h->rope_hd, h->handles, h->count, h->offsets,
                     h->count, h->strides, h->count, input, h->K, output, h->N);
  if (timed)
    resp->stage_us[0] = (uint32_t)(dspq_now_us() - start);
  return rc;
}

static int dspq_router_q4(struct nntr_hvx_dspq *d, const dspq_msg *m,
                          uint32_t len, uint32_t nb,
                          const struct dspqueue_buffer *bufs) {
  const htp_dspq_router_q4_req *h = (const htp_dspq_router_q4_req *)m;
  if (len != sizeof(*h) || nb != 2u || !bufs[0].ptr || !bufs[1].ptr ||
      !htp_dspq_router_q4_shape_valid(h) || h->flags ||
      bufs[0].size != htp_dspq_router_q4_input_bytes(h) ||
      bufs[1].size != htp_dspq_router_q4_output_bytes(h) ||
      bufs[0].size > HTP_DSPQ_BUF_BYTES || bufs[1].size > HTP_DSPQ_BUF_BYTES)
    return AEE_EBADPARM;
  // A queue worker must never enter the HMX kernel owned by FastRPC.
  if (!(d->s->moe_flags & HTP_FC_FLAG_M1_HVX))
    return AEE_EUNSUPPORTED;
  const float *x = (const float *)bufs[0].ptr;
  float *out = (float *)bufs[1].ptr;
  float *sum = out + htp_dspq_router_q4_norm_offset(h);
  float *norm = sum + h->K;
  const uint32_t rows = h->norm_kind + 1u;
  const remote_handle64 hs = (remote_handle64)d->s;
  if (h->norm_kind != 2u)
    memcpy(sum, x, h->K * sizeof(float));
  if (h->norm_kind) {
    const int add = h->norm_kind == 2u;
    const int rc = nntr_hvx_inference_math_f32(
      hs, add ? 7u : 0u, 1, h->K, h->K, h->eps, 1.0f, x, h->K,
      add ? x + h->K : NULL, add ? h->K : 0, x + (rows - 1u) * h->K, h->K,
      (uint8 *)(add ? sum : norm), (add ? 8u : 4u) * h->K);
    if (rc)
      return rc;
  } else {
    memcpy(norm, x, h->K * sizeof(float));
  }
  return nntr_hvx_router_q4_prefill(
    hs, 1, h->K, h->E, h->w_handle, norm, h->K, h->top_k, h->n_sel,
    x + rows * h->K, h->E, (uint32_t *)out, h->n_sel, out + h->n_sel, h->top_k);
}

static void dspq_thread(void *arg) {
  struct nntr_hvx_dspq *d = (struct nntr_hvx_dspq *)arg;
  dspq_msg m;
  uint64_t spin_until = 0; /* 0: block on the next read */
  while (!d->stop) {
    uint32_t flags = 0, nb = 0, len = 0;
    struct dspqueue_buffer bufs[3];
    memset(bufs, 0, sizeof(bufs));
    int err;
    if (spin_until != 0) {
      err = dspqueue_read_noblock(d->q, &flags, 3, &nb, bufs, sizeof(m), &len,
                                  (uint8_t *)&m);
      if (err == AEE_EWOULDBLOCK) {
        ++d->empty;
        dspq_pause();
        if (dspq_now_us() >= spin_until) {
          spin_until = 0;
        }
        continue;
      }
    } else {
      err = dspqueue_read(d->q, &flags, 3, &nb, bufs, sizeof(m), &len,
                          (uint8_t *)&m, DSPQ_BLOCK_TIMEOUT_US);
      if (err == AEE_EEXPIRED) {
        continue;
      }
    }
    if (err != AEE_SUCCESS) {
      FARF(ERROR, "dspq: read failed: 0x%08x", (unsigned)err);
      ++d->bad;
      break;
    }
    if (len >= 4 && m.u[0] == HTP_DSPQ_OP_QUIT && nb == 0) {
      break;
    }
    if (len >= 4 && m.u[0] == HTP_DSPQ_OP_TOKEN) {
      err = dspq_token(d, &m, len, nb, bufs);
      if (err != AEE_SUCCESS) {
        FARF(ERROR, "dspq: write failed: 0x%08x", (unsigned)err);
        ++d->bad;
        break;
      }
      ++d->served;
      spin_until = d->spin_us ? dspq_now_us() + d->spin_us : 0;
      continue;
    }

    // A malformed packet is answered too, so the ARM side never hangs on it.
    htp_dspq_resp resp;
    memset(&resp, 0, sizeof(resp));
    resp.seq = len >= 8 ? m.u[1] : 0;
    const uint32_t op = len >= 4 ? m.u[0] : 0;
    const int valid =
      op == HTP_DSPQ_OP_CONV_CACHED   ? len == sizeof(htp_dspq_conv_cached_req)
      : op == HTP_DSPQ_OP_ROUTER_Q4   ? len == sizeof(htp_dspq_router_q4_req)
      : op == HTP_DSPQ_OP_FC          ? len == sizeof(htp_dspq_fc_req)
      : op == HTP_DSPQ_OP_CONV        ? len == sizeof(htp_dspq_conv_req)
      : op == HTP_DSPQ_OP_ROPE_QKV    ? len == sizeof(htp_dspq_rope_qkv_req)
      : op == HTP_DSPQ_OP_ROUTER_NORM ? len == sizeof(htp_dspq_router_norm_req)
      : op == HTP_DSPQ_OP_MATH        ? len == sizeof(htp_dspq_math_req)
      : op == HTP_DSPQ_OP_ROUTER      ? len == sizeof(htp_dspq_router_req)
                                      : dspq_valid(&m, len, nb, bufs);
    if (valid && op == HTP_DSPQ_OP_CONV_CACHED)
      resp.rc = dspq_conv_cached(d, &m, len, nb, bufs, &resp);
    else if (valid && op == HTP_DSPQ_OP_ROUTER_Q4)
      resp.rc = dspq_router_q4(d, &m, len, nb, bufs);
    else if (valid && op == HTP_DSPQ_OP_FC)
      resp.rc = dspq_fc(d, &m, len, nb, bufs, &resp);
    else if (valid && op == HTP_DSPQ_OP_CONV)
      resp.rc = dspq_conv(d, &m, len, nb, bufs, &resp);
    else if (valid && op == HTP_DSPQ_OP_MATH)
      resp.rc = dspq_math(d, &m, len, nb, bufs, &resp);
    else if (valid && op == HTP_DSPQ_OP_ROUTER)
      resp.rc = dspq_router(d, &m, len, nb, bufs);
    else if (valid && op == HTP_DSPQ_OP_ROUTER_NORM)
      resp.rc = dspq_router_norm(d, &m, len, nb, bufs);
    else if (valid && op == HTP_DSPQ_OP_ROPE_QKV)
      resp.rc = dspq_rope_qkv(d, &m, len, nb, bufs);
    else if (valid)
      resp.rc = dspq_run(d, &m, bufs, &resp);
    else {
      resp.rc = AEE_EBADPARM;
      ++d->bad;
    }
    uint32_t rlen =
      HTP_DSPQ_RESP_BASE_BYTES + ((valid && (m.h.flags & HTP_DSPQ_FLAG_TIMED))
                                    ? HTP_DSPQ_STAGES * 4u
                                    : 0u);
    if (valid && op == HTP_DSPQ_OP_MOE && (m.h.flags & HTP_DSPQ_FLAG_TIMED) &&
        (m.h.flags & HTP_DSPQ_FLAG_EXPERT_UPDATES))
      rlen += sizeof(resp.expert_update_us);
    // Every reference the request took is released, malformed or not; the
    // output goes back flushed from the DSP's caches.
    for (uint32_t i = 0; i < nb; ++i) {
      bufs[i].flags = DSPQUEUE_BUFFER_FLAG_DEREF;
    }
    if (nb >= 2) {
      bufs[1].flags |= DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER |
                       DSPQUEUE_BUFFER_FLAG_INVALIDATE_RECIPIENT;
    }
    err = dspqueue_write(d->q, 0, nb, bufs, rlen, (const uint8_t *)&resp,
                         DSPQUEUE_TIMEOUT_NONE);
    if (err != AEE_SUCCESS) {
      FARF(ERROR, "dspq: write failed: 0x%08x", (unsigned)err);
      ++d->bad;
      break;
    }
    ++d->served;
    spin_until = d->spin_us ? dspq_now_us() + d->spin_us : 0;
  }
  qurt_thread_exit(0); /* QURT_EOK; the host stub has no qurt_error.h */
}

/** @brief Joins the thread, closes the queue, frees; NULL-safe. */
static int dspq_teardown(nntr_hvx_session *s, uint32_t res[4]) {
  struct nntr_hvx_dspq *d = s->dspq;
  if (d == NULL) {
    return AEE_EBADSTATE;
  }
  d->stop = 1;
  int status;
  qurt_thread_join(d->tid, &status);
  const int err = dspqueue_close(d->q);
  if (res != NULL) {
    res[0] = d->served;
    res[1] = d->bad;
    res[2] = d->empty;
    res[3] = d->spin_us;
  }
  for (uint32_t i = 0; i < HTP_DSPQ_CONV_CACHE_SLOTS; ++i)
    free(d->conv_cache[i].data);
  free(d->stack);
  free(d);
  s->dspq = NULL;
  return err;
}

int nntr_hvx_dspq_start(remote_handle64 handle, uint64 queue_id,
                        uint32 spin_us) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  if (!dspqueue_import || !dspqueue_close || !dspqueue_write ||
      !dspqueue_read || !dspqueue_read_noblock) {
    return AEE_EUNSUPPORTED;
  }
  if (s == NULL || s->dspq != NULL) {
    return AEE_EBADSTATE;
  }
  /* [#178] a lite session (no HMX) takes a queue for OP_TOKEN (S2 in #132
     Part B E3); its OP_MOE packets get the MoE entry's AEE_EUNSUPPORTED */
  if (nntr_hvx_moe_stage_count() != HTP_DSPQ_STAGES) {
    FARF(ERROR, "dspq: stage slots %u, wire header says %u",
         (unsigned)nntr_hvx_moe_stage_count(), (unsigned)HTP_DSPQ_STAGES);
    return AEE_EINVALIDFORMAT;
  }
  struct nntr_hvx_dspq *d =
    (struct nntr_hvx_dspq *)calloc(1, sizeof(struct nntr_hvx_dspq));
  if (d == NULL) {
    return AEE_ENOMEMORY;
  }
  d->s = s;
  d->spin_us = spin_us;
  // No packet callback: blocking reads are refused once one is set.
  int err = dspqueue_import(queue_id, NULL, NULL, NULL, &d->q);
  if (err != AEE_SUCCESS) {
    FARF(ERROR, "dspq: dspqueue_import failed: 0x%08x", (unsigned)err);
    free(d);
    return err;
  }
  d->stack = malloc(DSPQ_STACK);
  if (d->stack == NULL) {
    dspqueue_close(d->q);
    free(d);
    return AEE_ENOMEMORY;
  }
  qurt_thread_attr_t attr;
  qurt_thread_attr_init(&attr);
  qurt_thread_attr_set_name(&attr, "nntr_dspq");
  qurt_thread_attr_set_stack_addr(&attr, d->stack);
  qurt_thread_attr_set_stack_size(&attr, DSPQ_STACK);
  // The caller's own priority: the one lane 0 runs at under FastRPC and
  // the one the worker pool was created with (hvx_worker_pool.c).
  qurt_thread_attr_set_priority(
    &attr, (unsigned short)qurt_thread_get_priority(qurt_thread_get_id()));
  err = qurt_thread_create(&d->tid, &attr, dspq_thread, d);
  if (err != 0) { /* QURT_EOK */
    FARF(ERROR, "dspq: qurt_thread_create failed: %d", err);
    dspqueue_close(d->q);
    free(d->stack);
    free(d);
    return AEE_EQURTTHREADCREATE;
  }
  s->dspq = d;
  return AEE_SUCCESS;
}

int nntr_hvx_dspq_stop(remote_handle64 handle, uint32 *res, int resLen) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  if (s == NULL || res == NULL || resLen < 4) {
    return AEE_EINVALIDFORMAT;
  }
  return dspq_teardown(s, res);
}

void nntr_hvx_dspq_shutdown(nntr_hvx_session *s) {
  if (s != NULL && s->dspq != NULL) {
    FARF(HIGH, "dspq: close() stops a queue the ARM side left open");
    dspq_teardown(s, NULL);
  }
}
