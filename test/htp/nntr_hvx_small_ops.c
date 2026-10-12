// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   nntr_hvx_small_ops.c
 * @date   27 Sep 2026
 * @brief  DSP-side test entries for the M=1 small ops (hvx_m1_ops_f32.c):
 *         RMSNorm / per-head norm, RoPE at head_dim 64, conv1d + gate, the
 *         MoE router (#132)
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 *
 * Each entry validates the FastRPC lengths the kernel cannot see, then
 * runs on the caller's buffers with the session's worker pool. RoPE with
 * half output uses temporary f32 storage; cast-only needs no scratch.
 * A bad shape is AEE_EINVALIDFORMAT; AEE_EBADPARM stays the stale-skel
 * symptom (rule 3), so the gtest can tell the two apart.
 */

#include <AEEStdErr.h>
#include <HAP_farf.h>
#include <HAP_perf.h>
#include <remote.h>
#include <stdlib.h>
#include <string.h>

#include "nntr_hvx.h"
#include "nntr_hvx_session.h"

#include "hvx_convert.h"
#include "hvx_m1_ops_f32.h"
#include "hvx_rmsnorm_rows_f32.h"
#include "hvx_rope_rows_f32.h"
#include "hvx_swiglu_det.h"
#include "hvx_worker_pool.h"
#include "m1_ops_det.h"

typedef struct {
  const float *x, *aux;
  void *out;
  uint32_t count, op;
  float scale;
} math_ctx;

/** Independent 64-element blocks: keep decode inline and split prefill
 * across the session's existing HVX workers. No reductions change order. */
static void math_worker(uint32_t threads, uint32_t thread, void *opaque) {
  const math_ctx *c = (const math_ctx *)opaque;
  const uint32_t blocks = (c->count + 63u) / 64u;
  const uint32_t lo = (uint32_t)((uint64_t)blocks * thread / threads) * 2u;
  const uint32_t hi =
    (uint32_t)((uint64_t)blocks * (thread + 1u) / threads) * 2u;
  const uint32_t end = hi < c->count / 32u ? hi : c->count / 32u;
  const HVX_UVector *x = (const HVX_UVector *)c->x;
  const HVX_UVector *a = (const HVX_UVector *)c->aux;
  HVX_UVector *y = (HVX_UVector *)c->out;
  if (c->op == 6u) {
    for (uint32_t i = lo; i < end; i += 2u) {
      const HVX_Vector second = i + 1u < end ? x[i + 1u] : x[i];
      // vcvt interleaves the two source vectors; deal restores row order.
      const HVX_Vector h = Q6_Vh_vdeal_Vh(Q6_Vhf_vcvt_VsfVsf(x[i], second));
      if (i + 1u < end)
        y[i / 2u] = h;
      else
        memcpy((uint16_t *)c->out + i * 32u, &h, 64u);
    }
  } else if (c->op == 1u) {
    for (uint32_t i = lo; i < end; ++i)
      y[i] = Q6_Vsf_vadd_VsfVsf(x[i], a[i]);
  } else if (c->op == 2u) {
    for (uint32_t i = lo; i < end; ++i)
      y[i] = hvx_swiglu_det_sf(x[i], a[i]);
  } else if (c->op == 3u) {
    for (uint32_t i = lo; i < end; ++i)
      y[i] = hvx_geglu_det_sf(x[i], a[i]);
  } else {
    const HVX_Vector scale = hvx_splat_sf(c->scale);
    for (uint32_t i = lo; i < end; ++i)
      y[i] = Q6_Vsf_vmpy_VsfVsf(x[i], scale);
  }
}

static void math_run(hvx_worker_pool *pool, uint32_t op, uint32_t count,
                     const float *x, const float *aux, void *out, float scale) {
  math_ctx c = {x, aux, out, count, op, scale};
  hvx_worker_pool_run(pool, math_worker, &c, (count + 4095u) / 4096u);
}

int nntr_hvx_inference_math_f32(remote_handle64 handle, uint32 op, uint32 M,
                                uint32 N, uint32 chunk, float eps, float scale,
                                const float *x, int xLen, const float *aux,
                                int auxLen, const float *param, int paramLen,
                                uint8 *out, int outLen) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  const uint64_t count = (uint64_t)M * N;
  const int half = op == 5u || op == 6u;
  if (!s || !M || !N || N % 32u || op > 7u ||
      count > 0x7FFFFFFFu / (op == 7u ? 8u : 4u) || (uint64_t)xLen != count ||
      (uint64_t)outLen != count * (half       ? 2u
                                   : op == 7u ? 8u
                                              : 4u))
    return AEE_EINVALIDFORMAT;
  if ((((op >= 1u && op <= 3u) || op == 7u) && (uint64_t)auxLen != count) ||
      ((op == 0u || op == 7u) &&
       (!chunk || N % chunk || chunk % 32u ||
        (paramLen != 0 && (uint32)paramLen != chunk))) ||
      ((op == 4u || op == 5u) &&
       (!chunk || N % chunk || chunk % 64u ||
        (uint64_t)paramLen != (uint64_t)M * 2u * chunk)))
    return AEE_EINVALIDFORMAT;
  if (op == 7u) {
    float *sum = (float *)out;
    math_run(s->quant_pool, 1u, (uint32_t)count, x, aux, sum, 1.0f);
    const int rc =
      hvx_rmsnorm_rows_f32(sum, sum + count, M, N, chunk,
                           paramLen ? param : NULL, eps, s->quant_pool);
    return rc ? AEE_EINVALIDFORMAT : AEE_SUCCESS;
  }
  if (op == 6u) {
    math_run(s->quant_pool, 6u, (uint32_t)count, x, NULL, out, 1.0f);
    return AEE_SUCCESS;
  }
  float *y =
    half ? (float *)memalign(128, (size_t)count * sizeof(float)) : (float *)out;
  if (!y)
    return AEE_ENOMEMORY;
  int rc = 0;
  if (op == 0u) {
    rc = hvx_rmsnorm_rows_f32(x, y, M, N, chunk, paramLen ? param : NULL, eps,
                              s->quant_pool);
    if (!rc && scale != 1.0f)
      math_run(s->quant_pool, 0u, (uint32_t)count, y, NULL, y, scale);
  } else if (op >= 1u && op <= 3u) {
    math_run(s->quant_pool, op, (uint32_t)count, x, aux, y, 1.0f);
  } else {
    if (y != x)
      memcpy(y, x, (size_t)count * sizeof(float));
    rc = hvx_rope_rows_f32(y, M, N, chunk, param, s->quant_pool);
  }
  if (half) {
    if (!rc)
      math_run(s->quant_pool, 6u, (uint32_t)count, y, NULL, out, 1.0f);
    free(y);
  }
  return rc ? AEE_EINVALIDFORMAT : AEE_SUCCESS;
}

int nntr_hvx_add_rmsnorm_f32(remote_handle64 handle, uint32 M, uint32 N,
                             float eps, const float *x, int xLen,
                             const float *residual, int residualLen,
                             const float *gamma, int gammaLen, float *sum,
                             int sumLen, float *norm, int normLen,
                             uint32 *elapsed_us, int elapsedLen) {
  const uint64_t count = (uint64_t)M * N;
  if (!handle || !M || !N || N % 32u || count > 0x7FFFFFFFu / 4u ||
      (uint64_t)xLen != count || (uint64_t)residualLen != count ||
      (uint64_t)sumLen != count || (uint64_t)normLen != count ||
      (gammaLen != 0 && (uint32_t)gammaLen != N) || elapsedLen != 1 ||
      sum == norm)
    return AEE_EINVALIDFORMAT;
  const uint64_t start =
    HAP_perf_qtimer_count_to_us(HAP_perf_get_qtimer_count());
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  math_run(s->quant_pool, 1u, (uint32_t)count, x, residual, sum, 1.0f);
  const int rc = hvx_rmsnorm_rows_f32(
    sum, norm, M, N, N, gammaLen ? gamma : NULL, eps, s->quant_pool);
  elapsed_us[0] =
    (uint32_t)(HAP_perf_qtimer_count_to_us(HAP_perf_get_qtimer_count()) -
               start);
  return rc ? AEE_EINVALIDFORMAT : AEE_SUCCESS;
}

int nntr_hvx_inference_math_timed_f32(remote_handle64 handle, uint32 op,
                                      uint32 M, uint32 N, uint32 chunk,
                                      float eps, float scale, const float *x,
                                      int xLen, const float *aux, int auxLen,
                                      const float *param, int paramLen,
                                      uint8 *out, int outLen,
                                      uint32 *elapsed_us, int elapsedLen) {
  if (elapsedLen != 1)
    return AEE_EINVALIDFORMAT;
  const uint64_t begin =
    HAP_perf_qtimer_count_to_us(HAP_perf_get_qtimer_count());
  const int rc =
    nntr_hvx_inference_math_f32(handle, op, M, N, chunk, eps, scale, x, xLen,
                                aux, auxLen, param, paramLen, out, outLen);
  elapsed_us[0] =
    (uint32)(HAP_perf_qtimer_count_to_us(HAP_perf_get_qtimer_count()) - begin);
  return rc;
}

typedef struct {
  uint32_t M, E, top_k, n_sel;
  const float *logits, *bias;
  uint32_t *sel;
  float *weight;
} sigmoid_ctx;

typedef struct {
  const float *logits, *bias;
  float *sig, *score;
  uint32_t E;
} sigmoid_expert_ctx;

static void sigmoid_expert_worker(uint32_t threads, uint32_t thread,
                                  void *opaque) {
  const sigmoid_expert_ctx *c = (const sigmoid_expert_ctx *)opaque;
  const uint32_t lo = (uint32_t)((uint64_t)c->E * thread / threads);
  const uint32_t hi = (uint32_t)((uint64_t)c->E * (thread + 1u) / threads);
  for (uint32_t e = lo; e < hi; ++e)
    m1_router_cpu_sigmoid(c->logits[e], c->bias[e], &c->sig[e], &c->score[e]);
}

static void sigmoid_select(const float *sig, const float *score, uint32_t E,
                           uint32_t top_k, uint32_t n_sel, uint32_t *sel,
                           float *weight) {
  uint8 taken[128] = {0};
  for (uint32_t k = 0; k < n_sel; ++k) {
    uint32_t best = E;
    for (uint32_t e = 0; e < E; ++e)
      if (!taken[e] && (best == E || score[e] > score[best]))
        best = e;
    taken[best] = 1;
    sel[k] = best;
  }
  float sum = 0.0f;
  for (uint32_t k = 0; k < top_k; ++k)
    sum += sig[sel[k]];
  const float inv = 1.0f / (sum + 1e-6f);
  for (uint32_t k = 0; k < top_k; ++k)
    weight[k] = sig[sel[k]] * inv;
}

/** Exact prefill sigmoid. The common positive-normal denominator allows
 *  the existing specialized reciprocal; exceptional inputs retain the
 *  general IEEE division, including subnormal results and canonical NaN. */
static inline void prefill_router_sigmoid(float logit, float bias, float *sig,
                                          float *score) {
  const float ex =
    m1_expf_bionic_det(m1_det_float(m1_det_bits(logit) ^ 0x80000000u));
  const float denominator = m1_det_add(ex, 1.0f);
  const uint32_t bits = m1_det_bits(denominator);
  *sig = bits >= 0x3f800000u && bits < 0x7e800000u
           ? m1_recip_rn_det(denominator)
           : cpu_det_div_rn(1.0f, denominator);
  *score = m1_det_add(*sig, bias);
}

static void sigmoid_worker(uint32_t threads, uint32_t thread, void *opaque) {
  const sigmoid_ctx *c = (const sigmoid_ctx *)opaque;
  const uint32_t E = c->E, top_k = c->top_k, n_sel = c->n_sel;
  const float *logits = c->logits, *bias = c->bias;
  uint32_t *sel = c->sel;
  float *weight = c->weight;
  const uint32_t lo = (uint32_t)((uint64_t)c->M * thread / threads);
  const uint32_t hi = (uint32_t)((uint64_t)c->M * (thread + 1u) / threads);
  for (uint32 r = lo; r < hi; ++r) {
    float sig[128], score[128];
    for (uint32 e = 0; e < E; ++e)
      prefill_router_sigmoid(logits[(size_t)r * E + e], bias[e], &sig[e],
                             &score[e]);
    sigmoid_select(sig, score, E, top_k, n_sel, sel + (size_t)r * n_sel,
                   weight + (size_t)r * top_k);
  }
}

int nntr_hvx_sigmoid_topk_f32(remote_handle64 handle, uint32 M, uint32 E,
                              uint32 top_k, uint32 n_sel, const float *logits,
                              int logitsLen, const float *bias, int biasLen,
                              uint32 *sel, int selLen, float *weight,
                              int weightLen) {
  if (!handle || !M || !E || E > 128u || !top_k || top_k > n_sel || n_sel > E ||
      (uint32)biasLen != E || (uint64_t)logitsLen != (uint64_t)M * E ||
      (uint64_t)selLen != (uint64_t)M * n_sel ||
      (uint64_t)weightLen != (uint64_t)M * top_k)
    return AEE_EINVALIDFORMAT;
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  if (M == 1 && E >= 64u) {
    // Each exact sigmoid is independent; retain the original ordered top-k
    // and normalization to preserve ties and every output bit.
    float sig[128], score[128];
    sigmoid_expert_ctx ec = {logits, bias, sig, score, E};
    hvx_worker_pool_run(s->quant_pool, sigmoid_expert_worker, &ec, E / 16u);
    sigmoid_select(sig, score, E, top_k, n_sel, sel, weight);
    return AEE_SUCCESS;
  }
  sigmoid_ctx c = {M, E, top_k, n_sel, logits, bias, sel, weight};
  // Exact scalar sigmoid/top-k within a row; independent rows run in parallel.
  hvx_worker_pool_run(s->quant_pool, sigmoid_worker, &c, (M + 7u) / 8u);
  return AEE_SUCCESS;
}

int nntr_hvx_router_sigmoid_f32(remote_handle64 handle, uint32 M, uint32 K,
                                uint32 E, float eps, const float *gamma,
                                int gammaLen, uint32 w_handle, const float *w,
                                int wLen, const float *x, int xLen,
                                uint32 top_k, uint32 n_sel, const float *bias,
                                int biasLen, float *logits, int logitsLen,
                                uint32 *sel, int selLen, float *weight,
                                int weightLen) {
  if (!handle || !M || !E || E > 128u || !top_k || top_k > n_sel || n_sel > E ||
      (uint32)biasLen != E || (uint64_t)logitsLen != (uint64_t)M * E ||
      (uint64_t)selLen != (uint64_t)M * n_sel ||
      (uint64_t)weightLen != (uint64_t)M * top_k)
    return AEE_EINVALIDFORMAT;
  const int rc =
    w_handle == UINT32_MAX
      ? nntr_hvx_router_logits_f32(handle, M, K, E, eps, gamma, gammaLen, w,
                                   wLen, x, xLen, 0, 0, NULL, 0, logits,
                                   logitsLen, NULL, 0, NULL, 0)
      : nntr_hvx_router_logits_u8i8_f32(handle, M, K, E, eps, gamma, gammaLen,
                                        w_handle, x, xLen, 0, 0, NULL, 0,
                                        logits, logitsLen, NULL, 0, NULL, 0);
  if (rc != AEE_SUCCESS)
    return rc;
  return nntr_hvx_sigmoid_topk_f32(handle, M, E, top_k, n_sel, logits,
                                   logitsLen, bias, biasLen, sel, selLen,
                                   weight, weightLen);
}

/** @brief f32 lanes per HVX vector. */
#define LANES 32u
/** @brief RoPE head dimension: two vectors per head. */
#define HEAD_DIM 64u

int nntr_hvx_rmsnorm_det_f32(remote_handle64 handle, uint32 chunk, float eps,
                             const float *x, int xLen, const float *gamma,
                             int gammaLen, float *y, int yLen, float *row_scale,
                             int rsLen) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  if (!s) {
    return AEE_EBADPARM;
  }
  if (xLen <= 0 || chunk == 0u || chunk % LANES != 0u ||
      (chunk & (chunk - 1u)) != 0u || (uint32)xLen % chunk != 0u ||
      (uint32)gammaLen != chunk || yLen != xLen ||
      (uint32)rsLen != (uint32)xLen / chunk) {
    FARF(ERROR,
         "rmsnorm_det_f32: bad shape (chunk=%u x=%d gamma=%d y=%d rs=%d)",
         (unsigned)chunk, xLen, gammaLen, yLen, rsLen);
    return AEE_EINVALIDFORMAT;
  }
  hvx_rmsnorm_f32(x, gamma, y, (uint32_t)xLen, chunk, eps, row_scale);
  return AEE_SUCCESS;
}

int nntr_hvx_rmsnorm_add_f32(remote_handle64 handle, uint32 M, uint32 n,
                             float eps, float scale, const float *gamma,
                             int gammaLen, const float *x, int xLen, float *out,
                             int outLen) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  if (!s) {
    return AEE_EBADPARM;
  }
  const uint64_t rows = (uint64_t)M * n;
  if (M == 0u || n == 0u || n % LANES != 0u || rows > 0x7FFFFFFFu ||
      (uint64_t)outLen != rows ||
      ((uint64_t)xLen != rows && (uint64_t)xLen != 2u * rows) ||
      (gammaLen != 0 && (uint32)gammaLen != n)) {
    FARF(ERROR, "rmsnorm_add_f32: bad shape (M=%u n=%u gamma=%d x=%d out=%d)",
         (unsigned)M, (unsigned)n, gammaLen, xLen, outLen);
    return AEE_EINVALIDFORMAT;
  }
  const float *x2 = (uint64_t)xLen == 2u * rows ? x + rows : NULL;
  if (hvx_rmsnorm_add_f32(out, x, x2, M, n, gammaLen ? gamma : NULL, eps, scale,
                          s->quant_pool) != 0) {
    return AEE_EINVALIDFORMAT;
  }
  return AEE_SUCCESS;
}

int nntr_hvx_rope64_det_f32(remote_handle64 handle, uint32 n_q, const float *cs,
                            int csLen, const float *qk, int qkLen, float *y,
                            int yLen) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  if (!s) {
    return AEE_EBADPARM;
  }
  if ((uint32)csLen != HEAD_DIM || qkLen <= 0 ||
      (uint32)qkLen % HEAD_DIM != 0u || yLen != qkLen ||
      n_q > (uint32)qkLen / HEAD_DIM) {
    FARF(ERROR, "rope64_det_f32: bad shape (n_q=%u cs=%d qk=%d y=%d)",
         (unsigned)n_q, csLen, qkLen, yLen);
    return AEE_EINVALIDFORMAT;
  }
  memcpy(y, qk, (size_t)qkLen * sizeof(float));
  const uint32_t n_k = (uint32_t)qkLen / HEAD_DIM - n_q;
  hvx_rope64_f32(y, n_q, y + (size_t)n_q * HEAD_DIM, n_k, cs);
  return AEE_SUCCESS;
}

int nntr_hvx_conv_gate_m1_f32(remote_handle64 handle, const float *abc,
                              int abcLen, const float *conv_w, int wLen,
                              const float *state_in, int sLen, float *y,
                              int yLen, float *state_out, int soLen) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  if (!s) {
    return AEE_EBADPARM;
  }
  const uint32_t C = (yLen > 0) ? (uint32_t)yLen : 0u;
  if (C == 0u || C % LANES != 0u || (uint32)abcLen != 3u * C ||
      (uint32)wLen != 3u * C || (uint32)sLen != 2u * C ||
      (uint32)soLen != 3u * C) {
    FARF(ERROR,
         "conv_gate_m1_f32: bad shape (abc=%d w=%d state=%d y=%d "
         "state_out=%d)",
         abcLen, wLen, sLen, yLen, soLen);
    return AEE_EINVALIDFORMAT;
  }
  memcpy(state_out, state_in, (size_t)sLen * sizeof(float));
  hvx_conv_gate_m1_f32(abc, state_out, conv_w, y, C);
  return AEE_SUCCESS;
}

int nntr_hvx_router_topk_det_f32(remote_handle64 handle, uint32 top_k,
                                 const float *x, int xLen, const float *w,
                                 int wLen, const float *bias, int biasLen,
                                 float *logits, int logitsLen, uint32 *sel,
                                 int selLen, float *weight, int weightLen) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  if (!s) {
    return AEE_EBADPARM;
  }
  const uint32_t K = (xLen > 0) ? (uint32_t)xLen : 0u;
  const uint32_t E = (biasLen > 0) ? (uint32_t)biasLen : 0u;
  if (K == 0u || K % 4u != 0u || E == 0u || E > LANES || top_k == 0u ||
      top_k > E || (uint32)wLen != K * E || (uint32)logitsLen != E ||
      (uint32)selLen != top_k || (uint32)weightLen != top_k) {
    FARF(ERROR,
         "router_topk_det_f32: bad shape (top_k=%u x=%d w=%d bias=%d "
         "logits=%d sel=%d weight=%d)",
         (unsigned)top_k, xLen, wLen, biasLen, logitsLen, selLen, weightLen);
    return AEE_EINVALIDFORMAT;
  }
  /* the kernel reads one 32-lane vector per weight row */
  float *w32 = (float *)calloc((size_t)K * LANES, sizeof(float));
  if (w32 == NULL) {
    return AEE_ENOMEMORY;
  }
  for (uint32_t k = 0; k < K; ++k) {
    memcpy(w32 + (size_t)k * LANES, w + (size_t)k * E, E * sizeof(float));
  }
  hvx_router_topk_f32(x, w32, bias, K, E, top_k, logits, (uint32_t *)sel,
                      weight, s->quant_pool);
  free(w32);
  return AEE_SUCCESS;
}
