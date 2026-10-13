// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Haehun Yang <haehun.yang@ax.samsung.com>
 *
 * @file   hvx_kv_quant.c
 * @date   28 Sep 2026
 * @brief  HVX quantization of one appended KV row (per head) for hexkl_kv_q
 * @see    https://github.com/nntrainer/nntrainer
 * @author Haehun Yang <haehun.yang@ax.samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include "hvx_kv_quant.h"

#include <string.h>

#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>

#include "hvx_attn_softmax_f16.h"
#include "hvx_convert.h"
#include "hvx_tile_f16.h"

/** @brief Largest head_dim (512): 16 f32 vectors. */
#define MAX_VECS 16u

/**
 * @brief 64 fp16 (natural order) -> two f32 vectors in natural order: the
 *        widening multiply by 1 splits by parity, the word shuffle puts the
 *        halves back in order.
 */
static inline void widen64(HVX_Vector hf, HVX_Vector *lo, HVX_Vector *hi) {
  const HVX_Vector one_hf = Q6_Vh_vsplat_R((int)HVX_ATTN_HF_ONE);
  const HVX_VectorPair w = Q6_Wqf32_vmpy_VhfVhf(hf, one_hf);
  const HVX_VectorPair nat = Q6_W_vshuff_VVR(
    Q6_Vsf_equals_Vqf32(Q6_V_hi_W(w)), Q6_Vsf_equals_Vqf32(Q6_V_lo_W(w)), -4);
  *lo = Q6_V_lo_W(nat);
  *hi = Q6_V_hi_W(nat);
}

/** @brief hd fp16 -> hd/32 f32 vectors, natural order. */
static inline void load_row_f32(const uint16_t *x_hf, uint32_t hd,
                                HVX_Vector *v) {
  for (uint32_t i = 0; i < hd / 64u; ++i) {
    widen64(hvx_tile_load_u(x_hf + 64u * i), &v[2u * i], &v[2u * i + 1u]);
  }
  if (hd % 64u) {
    // 32 trailing values: the half vector's upper lanes are zero.
    // A 64-byte memcpy can become a 128-byte HVX load, crossing the end
    // of a FastRPC mapping. Volatile halfword loads keep this tail bounded.
    uint16_t tail[64] = {0};
    const volatile uint16_t *src = x_hf + (hd & ~63u);
    for (uint32_t i = 0; i < 32u; ++i)
      tail[i] = src[i];
    const HVX_Vector h = hvx_tile_load_u(tail);
    HVX_Vector hi;
    widen64(h, &v[hd / 32u - 1u], &hi);
  }
}

/** @brief Lane 0 of max over the given vectors of |x|. */
static inline float absmax_of(const HVX_Vector *v, uint32_t n) {
  const HVX_Vector mask = Q6_V_vsplat_R(0x7FFFFFFF);
  HVX_Vector m = Q6_V_vand_VV(v[0], mask);
  for (uint32_t i = 1; i < n; ++i) {
    m = Q6_Vsf_vmax_VsfVsf(m, Q6_V_vand_VV(v[i], mask));
  }
  return hvx_attn_lane0_f32(hvx_attn_max32_sf(m));
}

/** @brief rne(x * inv) clamped to [-fq, fq], as int32 lanes. */
static inline HVX_Vector quant_vec(HVX_Vector x, HVX_Vector inv, HVX_Vector lo,
                                   HVX_Vector hi) {
  HVX_Vector q = hvx_sf_to_w_rne(Q6_Vsf_vmpy_VsfVsf(x, inv));
  q = Q6_Vw_vmax_VwVw(q, lo);
  return Q6_Vw_vmin_VwVw(q, hi);
}

/** @brief hd int32 lanes (hd/32 vectors) -> hd bytes at @a q. */
static inline void pack_bytes(const HVX_Vector *w, uint32_t hd, int8_t *q) {
  uint32_t i = 0;
  for (; i + 4u <= hd / 32u; i += 4u) {
    const HVX_Vector h01 = Q6_Vh_vpack_VwVw_sat(w[i + 1u], w[i]);
    const HVX_Vector h23 = Q6_Vh_vpack_VwVw_sat(w[i + 3u], w[i + 2u]);
    hvx_tile_store_u(q + 32u * i, Q6_Vb_vpack_VhVh_sat(h23, h01));
  }
  for (; i < hd / 32u; ++i) {
    const HVX_Vector h = Q6_Vh_vpack_VwVw_sat(w[i], w[i]);
    const HVX_Vector b = Q6_Vb_vpack_VhVh_sat(h, h);
    uint8_t tmp[128];
    hvx_tile_store_u(tmp, b);
    memcpy(q + 32u * i, tmp, 32u);
  }
}

/* Keep the IEEE product boundary before round-to-nearest-even. Otherwise
 * the Hexagon combiner can feed a wider qf32 product into the magic add. */
static inline HVX_Vector rounded_product(HVX_Vector x, HVX_Vector inv) {
  HVX_Vector product = Q6_Vsf_vmpy_VsfVsf(x, inv);
#if defined(__hexagon__)
  __asm__ volatile("" : "+v"(product));
#endif
  return product;
}

typedef struct {
  const uint16_t *k, *v;
  const float *q;
  uint8_t *k8, *v8, *q8;
  float *kv_scales, *q_scales;
  int32_t *q_zp;
  uint32_t kv_heads, q_heads, hd;
  int invalid;
} attn_quant_ctx;

static inline __attribute__((always_inline)) int
quant_attention_head_impl(const void *input, uint32_t hd, int query,
                          uint8_t *output, float *scale, int32_t *zp) {
  HVX_Vector x[8], words[8];
  const uint32_t n = hd / 32u;
  if (query) {
    for (uint32_t i = 0; i < n; ++i)
      x[i] = hvx_tile_load_u((const float *)input + 32u * i);
  } else {
    load_row_f32(input, hd, x);
  }
  const HVX_Vector mask = Q6_V_vsplat_R(0x7FFFFFFF);
  HVX_Vector lo = Q6_V_vzero(), hi = Q6_V_vzero(), abs = Q6_V_vzero();
  for (uint32_t i = 0; i < n; ++i) {
    abs = Q6_Vw_vmax_VwVw(abs, Q6_V_vand_VV(x[i], mask));
    if (query) {
      lo = Q6_Vsf_vmin_VsfVsf(lo, x[i]);
      hi = Q6_Vsf_vmax_VsfVsf(hi, x[i]);
    }
  }
  for (int rot = 64; rot >= 4; rot >>= 1) {
    abs = Q6_Vw_vmax_VwVw(abs, Q6_V_vror_VR(abs, rot));
    if (query) {
      lo = Q6_Vsf_vmin_VsfVsf(lo, Q6_V_vror_VR(lo, rot));
      hi = Q6_Vsf_vmax_VsfVsf(hi, Q6_V_vror_VR(hi, rot));
    }
  }
  if ((uint32_t)Q6_R_vextract_VR(abs, 0) >= 0x7F800000u)
    return -1;
  float inv;
  int32_t zero = 0;
  if (query) {
    const float a = hvx_attn_lane0_f32(lo), b = hvx_attn_lane0_f32(hi);
    *scale = a == b ? 1.0f : (b - a) / 255.0f;
    uint32_t scale_bits;
    memcpy(&scale_bits, scale, sizeof(scale_bits));
    if (!(*scale > 0.0f) || (scale_bits & 0x7F800000u) == 0x7F800000u)
      return -1;
    zero = Q6_R_vextract_VR(hvx_sf_to_w_rne(hvx_splat_sf(-a / *scale)), 0);
    zero = zero < 0 ? 0 : zero > 255 ? 255 : zero;
    *zp = zero;
    inv = 1.0f / *scale;
  } else {
    const float peak = hvx_attn_lane0_f32(abs);
    *scale = peak == 0.0f ? 1.0f : peak / 127.0f;
    inv = peak == 0.0f ? 0.0f : 127.0f / peak;
  }
  const HVX_Vector vinv = hvx_splat_sf(inv);
  const HVX_Vector vz = Q6_V_vsplat_R(zero);
  const HVX_Vector vmin = Q6_V_vsplat_R(query ? 0 : -127);
  const HVX_Vector vmax = Q6_V_vsplat_R(query ? 255 : 127);
  for (uint32_t i = 0; i < n; ++i) {
    HVX_Vector w = hvx_sf_to_w_rne(rounded_product(x[i], vinv));
    w = Q6_Vw_vadd_VwVw(w, vz);
    words[i] = Q6_Vw_vmin_VwVw(Q6_Vw_vmax_VwVw(w, vmin), vmax);
  }
  for (uint32_t i = 0; i < n; i += 4u) {
    const HVX_Vector a = words[i];
    const HVX_Vector b = i + 1u < n ? words[i + 1u] : a;
    const HVX_Vector c = i + 2u < n ? words[i + 2u] : a;
    const HVX_Vector d = i + 3u < n ? words[i + 3u] : a;
    const HVX_Vector h0 = Q6_Vh_vpack_VwVw_sat(b, a);
    const HVX_Vector h1 = Q6_Vh_vpack_VwVw_sat(d, c);
    const HVX_Vector bytes =
      query ? Q6_Vub_vpack_VhVh_sat(h1, h0) : Q6_Vb_vpack_VhVh_sat(h1, h0);
    if (i + 4u <= n) {
      hvx_tile_store_u(output + 32u * i, bytes);
    } else {
      uint8_t tail[128];
      hvx_tile_store_u(tail, bytes);
      memcpy(output + 32u * i, tail, (n - i) * 32u);
    }
  }
  return 0;
}

/* The common 64-channel LFM2 heads use constant bounds, so the compiler
 * keeps their two vectors in registers instead of materializing the generic
 * eight-vector arrays on every head. */
static int quant_query64(const float *input, uint8_t *output, float *scale,
                         int32_t *zp) {
  return quant_attention_head_impl(input, 64u, 1, output, scale, zp);
}

static int quant_kv64(const uint16_t *input, uint8_t *output, float *scale) {
  return quant_attention_head_impl(input, 64u, 0, output, scale, NULL);
}

static int quant_attention_head(const void *input, uint32_t hd, int query,
                                uint8_t *output, float *scale, int32_t *zp) {
  return quant_attention_head_impl(input, hd, query, output, scale, zp);
}

static void attn_quant_worker(uint32_t n_threads, uint32_t tid, void *opaque) {
  attn_quant_ctx *c = opaque;
  /* Partition K/V and Q separately: query quantization costs more per head.
   * Assigning contiguous parts of K,V,Q together leaves some workers idle. */
  const uint32_t kv_lo = (uint32_t)((uint64_t)c->kv_heads * tid / n_threads);
  const uint32_t kv_hi =
    (uint32_t)((uint64_t)c->kv_heads * (tid + 1u) / n_threads);
  const uint32_t q_lo = (uint32_t)((uint64_t)c->q_heads * tid / n_threads);
  const uint32_t q_hi =
    (uint32_t)((uint64_t)c->q_heads * (tid + 1u) / n_threads);
  for (uint32_t h = kv_lo; h < kv_hi; ++h) {
    const size_t offset = (size_t)h * c->hd;
    const int rk =
      c->hd == 64u
        ? quant_kv64(c->k + offset, c->k8 + offset, c->kv_scales + 2u * h)
        : quant_attention_head(c->k + offset, c->hd, 0, c->k8 + offset,
                               c->kv_scales + 2u * h, NULL);
    const int rv =
      c->hd == 64u
        ? quant_kv64(c->v + offset, c->v8 + offset, c->kv_scales + 2u * h + 1u)
        : quant_attention_head(c->v + offset, c->hd, 0, c->v8 + offset,
                               c->kv_scales + 2u * h + 1u, NULL);
    if (rk || rv) {
      __atomic_store_n(&c->invalid, 1, __ATOMIC_RELAXED);
      return;
    }
  }
  for (uint32_t h = q_lo; h < q_hi; ++h) {
    const size_t offset = (size_t)h * c->hd;
    const int rc =
      c->hd == 64u
        ? quant_query64(c->q + offset, c->q8 + offset, c->q_scales + h,
                        c->q_zp + h)
        : quant_attention_head(c->q + offset, c->hd, 1, c->q8 + offset,
                               c->q_scales + h, c->q_zp + h);
    if (rc) {
      __atomic_store_n(&c->invalid, 1, __ATOMIC_RELAXED);
      return;
    }
  }
}

int hvx_attn_quant_rows_q8(const uint16_t *k, const uint16_t *v, const float *q,
                           uint32_t kv_heads, uint32_t q_heads, uint32_t hd,
                           uint8_t *k8, uint8_t *v8, float *kv_scales,
                           uint8_t *q8, float *q_scales, int32_t *q_zp,
                           hvx_worker_pool *pool) {
  const uint64_t heads = (uint64_t)kv_heads * 2u + q_heads;
  if (!q || !q8 || !q_scales || !q_zp || !q_heads || !hd || hd % 32u ||
      hd > 256u || heads > UINT32_MAX ||
      (kv_heads && (!k || !v || !k8 || !v8 || !kv_scales)))
    return -1;
  attn_quant_ctx c = {k,        v,    q,        k8,      v8, q8, kv_scales,
                      q_scales, q_zp, kv_heads, q_heads, hd, 0};
  hvx_worker_pool_run(pool, attn_quant_worker, &c, (uint32_t)heads);
  return __atomic_load_n(&c.invalid, __ATOMIC_RELAXED) ? -1 : 0;
}

void hvx_kv_quant_k_row(const uint16_t *x_hf, uint32_t hd, int32_t qmax,
                        int8_t *q, float *scale, int32_t *colsum) {
  HVX_Vector v[MAX_VECS];
  const uint32_t n = hd / 32u;
  load_row_f32(x_hf, hd, v);
  const float amax = absmax_of(v, n);
  if (amax == 0.0f) {
    memset(q, 0, hd);
    *scale = 1.0f;
    *colsum = 0;
    return;
  }
  const float fq = (float)qmax;
  const HVX_Vector inv = hvx_splat_sf(fq / amax);
  const HVX_Vector lo = Q6_V_vsplat_R(-qmax), hi = Q6_V_vsplat_R(qmax);
  HVX_Vector w[MAX_VECS];
  HVX_Vector sum = Q6_V_vzero();
  for (uint32_t i = 0; i < n; ++i) {
    w[i] = quant_vec(v[i], inv, lo, hi);
    sum = Q6_Vw_vadd_VwVw(sum, w[i]);
  }
  for (int rot = 4; rot <= 64; rot <<= 1) {
    sum = Q6_Vw_vadd_VwVw(sum, Q6_V_vror_VR(sum, rot));
  }
  pack_bytes(w, hd, q);
  *scale = amax / fq;
  *colsum = (int32_t)hvx_attn_word0(sum);
}

void hvx_kv_quant_v_row(const uint16_t *x_hf, uint32_t hd, int32_t qmax,
                        int8_t *q, float *scale) {
  HVX_Vector v[MAX_VECS];
  const uint32_t n = hd / 32u;
  load_row_f32(x_hf, hd, v);
  const float amax = absmax_of(v, n);
  if (amax == 0.0f) {
    memset(q, 0, hd);
    *scale = 1.0f;
    return;
  }
  const float fq = (float)qmax;
  const HVX_Vector inv = hvx_splat_sf(fq / amax);
  const HVX_Vector lo = Q6_V_vsplat_R(-qmax), hi = Q6_V_vsplat_R(qmax);
  HVX_Vector w[MAX_VECS];
  for (uint32_t g = 0; g < n; ++g) {
    w[g] = quant_vec(v[g], inv, lo, hi);
  }
  pack_bytes(w, hd, q);
  *scale = amax / fq;
}

void hvx_kv_quant_k_row_fixed(const uint16_t *x_hf, uint32_t hd, int32_t qmax,
                              float inv_s, int8_t *q, int32_t *colsum) {
  HVX_Vector v[MAX_VECS];
  const uint32_t n = hd / 32u;
  load_row_f32(x_hf, hd, v);
  const HVX_Vector inv = hvx_splat_sf(inv_s);
  const HVX_Vector lo = Q6_V_vsplat_R(-qmax), hi = Q6_V_vsplat_R(qmax);
  HVX_Vector w[MAX_VECS];
  HVX_Vector sum = Q6_V_vzero();
  for (uint32_t i = 0; i < n; ++i) {
    w[i] = quant_vec(v[i], inv, lo, hi);
    sum = Q6_Vw_vadd_VwVw(sum, w[i]);
  }
  for (int rot = 4; rot <= 64; rot <<= 1) {
    sum = Q6_Vw_vadd_VwVw(sum, Q6_V_vror_VR(sum, rot));
  }
  pack_bytes(w, hd, q);
  *colsum = (int32_t)hvx_attn_word0(sum);
}

void hvx_kv_quant_v_row_fixed(const uint16_t *x_hf, uint32_t hd, int32_t qmax,
                              const float *inv, int8_t *q) {
  HVX_Vector v[MAX_VECS];
  const uint32_t n = hd / 32u;
  load_row_f32(x_hf, hd, v);
  const HVX_Vector lo = Q6_V_vsplat_R(-qmax), hi = Q6_V_vsplat_R(qmax);
  HVX_Vector w[MAX_VECS];
  for (uint32_t g = 0; g < n; ++g) {
    const HVX_Vector iv = *(const HVX_UVector *)(inv + 32u * g);
    w[g] = quant_vec(v[g], iv, lo, hi);
  }
  pack_bytes(w, hd, q);
}
