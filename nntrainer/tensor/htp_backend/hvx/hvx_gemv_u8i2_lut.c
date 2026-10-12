// SPDX-License-Identifier: Apache-2.0
/**
 * @file hvx_gemv_u8i2_lut.c
 * @brief Pair-LUT GEMV and a register-only WH expansion comparison.
 */
#include "hvx_gemv_u8i2_lut.h"

#include <string.h>

#if defined(__hexagon__)
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#endif

/* V81's nomatch lookup replaces index masking/table-select instructions.
 * Keep the portable intrinsic path for older Hexagon targets. */
#ifndef NNTR_LUT_SINGLE_UNROLL
#define NNTR_LUT_SINGLE_UNROLL 4
#endif
#ifndef NNTR_LUT_NOMATCH
#define NNTR_LUT_NOMATCH 1
#endif
#if defined(__hexagon__) && __HEXAGON_ARCH__ >= 81 && NNTR_LUT_NOMATCH
#define NNTR_LUT_USE_NOMATCH 1
#else
#define NNTR_LUT_USE_NOMATCH 0
#endif

static size_t wh_code_offset(uint32_t k, uint32_t n, uint32_t N,
                             uint32_t *shift) {
  const uint32_t r = k % 32u, c = n % 32u;
  const uint32_t slot =
    (r / 8u) * 256u + c * 8u + (r % 4u) * 2u + (r / 4u) % 2u;
  const uint32_t old = slot / 2u;
  *shift = 2u * (slot % 2u) + ((old / 128u) % 2u) * 4u;
  return ((size_t)(k / 32u) * (N / 32u) + n / 32u) * 256u +
         (old / 256u) * 128u + old % 128u;
}

static int repack(const uint8_t *src, uint32_t K, uint32_t N, uint8_t *dst,
                  int inverse) {
  if (!src || !dst || src == dst || !K || !N || K % 32u || N % 128u ||
      (size_t)K > SIZE_MAX / N)
    return -1;
  memset(dst, 0, (size_t)K * N / 4u);
  for (uint32_t nt = 0; nt < N / 128u; ++nt) {
    for (uint32_t kt = 0; kt < K / 32u; ++kt) {
      for (uint32_t t = 0; t < 8u; ++t) {
        for (uint32_t j = 0; j < 128u; ++j) {
          const uint32_t n = nt * 128u + (j % 4u) * 32u + j / 4u;
          const size_t to = (((size_t)nt * (K / 32u) + kt) * 8u + t) * 128u + j;
          for (uint32_t h = 0; h < 4u; ++h) {
            const uint32_t k = kt * 32u + 2u * t + (h % 2u) + (h / 2u) * 16u;
            uint32_t shift;
            const size_t from = wh_code_offset(k, n, N, &shift);
            if (inverse)
              dst[from] |= ((src[to] >> (2u * h)) & 3u) << shift;
            else
              dst[to] |= ((src[from] >> shift) & 3u) << (2u * h);
          }
        }
      }
    }
  }
  return 0;
}

int hvx_i2_lut_pack_wh(const uint8_t *wh, uint32_t K, uint32_t N,
                       uint8_t *packed) {
  return repack(wh, K, N, packed, 0);
}

int hvx_i2_lut_unpack_wh(const uint8_t *packed, uint32_t K, uint32_t N,
                         uint8_t *wh) {
  return repack(packed, K, N, wh, 1);
}

#if defined(__hexagon__)
static inline HVX_Vector code_pair(HVX_Vector low, HVX_Vector high, int shift) {
  const HVX_Vector mask = Q6_V_vsplat_R(0x00000303);
  low = Q6_V_vand_VV(Q6_Vuw_vlsr_VuwR(low, shift), mask);
  high = Q6_V_vand_VV(Q6_Vuw_vlsr_VuwR(high, shift), mask);
  const HVX_Vector pair = Q6_V_vor_VV(low, Q6_Vw_vasl_VwR(high, 4));
  return Q6_V_vor_VV(pair, Q6_Vuw_vlsr_VuwR(pair, 6));
}
#endif

void hvx_i2_lut_repack(const uint8_t *src, uint32_t N, uint32_t src_stride,
                       uint8_t *dst, uint32_t dst_k_stride,
                       uint32_t dst_n_stride, uint32_t kt_begin,
                       uint32_t kt_end) {
  for (uint32_t kt = kt_begin; kt < kt_end; ++kt) {
    for (uint32_t nt = 0; nt < N / 128u; ++nt) {
      const uint8_t *s = src + (size_t)kt * src_stride + nt * 1024u;
      uint8_t *d = dst + (size_t)kt * dst_k_stride + (size_t)nt * dst_n_stride;
#if defined(__hexagon__)
      const HVX_Vector l0 = *(const HVX_Vector *)(s + 0u);
      const HVX_Vector h0 = *(const HVX_Vector *)(s + 128u);
      const HVX_Vector l1 = *(const HVX_Vector *)(s + 256u);
      const HVX_Vector h1 = *(const HVX_Vector *)(s + 384u);
      const HVX_Vector l2 = *(const HVX_Vector *)(s + 512u);
      const HVX_Vector h2 = *(const HVX_Vector *)(s + 640u);
      const HVX_Vector l3 = *(const HVX_Vector *)(s + 768u);
      const HVX_Vector h3 = *(const HVX_Vector *)(s + 896u);
      for (uint32_t t = 0; t < 8u; ++t) {
        const int shift = (int)((t & 1u) * 16u + (t / 2u) * 2u);
        const HVX_Vector b0 = code_pair(l0, h0, shift);
        const HVX_Vector b1 = code_pair(l1, h1, shift);
        const HVX_Vector b2 = code_pair(l2, h2, shift);
        const HVX_Vector b3 = code_pair(l3, h3, shift);
        const HVX_Vector h02 = Q6_Vh_vshuffe_VhVh(b2, b0);
        const HVX_Vector h13 = Q6_Vh_vshuffe_VhVh(b3, b1);
        *(HVX_Vector *)(d + t * 128u) = Q6_Vb_vshuffe_VbVb(h13, h02);
      }
#else
      /* Read the complete tile first, also for the permitted row overlap. */
      uint8_t tile[1024];
      memcpy(tile, s, sizeof(tile));
      for (uint32_t t = 0; t < 8u; ++t) {
        for (uint32_t j = 0; j < 128u; ++j) {
          const uint32_t n = (j % 4u) * 32u + j / 4u;
          uint8_t value = 0;
          for (uint32_t h = 0; h < 4u; ++h) {
            uint32_t shift;
            const uint32_t k = 2u * t + h % 2u + (h / 2u) * 16u;
            const size_t off = wh_code_offset(k, n, 128u, &shift);
            value |= ((tile[off] >> shift) & 3u) << (2u * h);
          }
          d[t * 128u + j] = value;
        }
      }
#endif
    }
  }
}

/* Inverse of the four-column interleave and 2-bit pairing. All eight
 * source vectors are consumed before the first store, so a tile may alias. */
void hvx_i2_lut_unrepack_row(uint8_t *row, uint32_t bytes) {
  for (uint32_t off = 0; off < bytes; off += 1024u) {
    uint8_t *p = row + off;
#if defined(__hexagon__)
    HVX_Vector x[8];
    for (uint32_t t = 0; t < 8u; ++t)
      x[t] = *(const HVX_Vector *)(p + t * 128u);
    const HVX_Vector mask = Q6_V_vsplat_R(0x03030303);
    for (uint32_t h = 0; h < 2u; ++h) {
      HVX_Vector v[4];
#pragma unroll
      for (uint32_t r = 0; r < 4u; ++r) {
        const uint32_t shift = h * 4u + (r % 2u) * 2u;
        const uint32_t t = r / 2u;
        HVX_Vector a = Q6_V_vand_VV(Q6_Vuw_vlsr_VuwR(x[t], shift), mask);
#pragma unroll
        for (uint32_t b = 1; b < 4u; ++b) {
          const HVX_Vector c =
            Q6_V_vand_VV(Q6_Vuw_vlsr_VuwR(x[t + b * 2u], shift), mask);
          a = Q6_V_vor_VV(a, Q6_Vw_vasl_VwR(c, b * 2u));
        }
        v[r] = a;
      }
      /* Transpose each 4x4 byte square: output vectors are N32 tiles,
       * their four bytes per column are the four consecutive K rows. */
      const HVX_VectorPair a = Q6_W_vshuff_VVR(v[1], v[0], 1);
      const HVX_VectorPair b = Q6_W_vshuff_VVR(v[3], v[2], 1);
      const HVX_VectorPair c = Q6_W_vshuff_VVR(Q6_V_lo_W(b), Q6_V_lo_W(a), 2);
      const HVX_VectorPair d = Q6_W_vshuff_VVR(Q6_V_hi_W(b), Q6_V_hi_W(a), 2);
      *(HVX_Vector *)(p + h * 128u) = Q6_V_lo_W(c);
      *(HVX_Vector *)(p + 256u + h * 128u) = Q6_V_lo_W(d);
      *(HVX_Vector *)(p + 512u + h * 128u) = Q6_V_hi_W(c);
      *(HVX_Vector *)(p + 768u + h * 128u) = Q6_V_hi_W(d);
    }
#else
    uint8_t tile[1024];
    memcpy(tile, p, sizeof(tile));
    (void)hvx_i2_lut_unpack_wh(tile, 32u, 128u, p);
#endif
  }
}

static inline void lut_prepare(const uint8_t *act_ah, const int8_t *palette,
                               uint32_t kt_begin, uint32_t kt_end,
                               int16_t *tables, int center) {
#if defined(__hexagon__)
  int16_t p0[64] __attribute__((aligned(128)));
  int16_t p1[64] __attribute__((aligned(128)));
  for (uint32_t j = 0; j < 64u; ++j) {
    const uint32_t idx = (j / 2u) % 16u;
    p0[j] = palette[idx & 3u];
    p1[j] = palette[idx >> 2u];
  }
  const HVX_Vector vp0 = *(const HVX_Vector *)p0;
  const HVX_Vector vp1 = *(const HVX_Vector *)p1;
  const HVX_VectorPred lower = Q6_Q_vsetq_R(64);
#endif
  for (uint32_t kt = kt_begin; kt < kt_end; ++kt) {
    const uint8_t *a = act_ah + (size_t)kt * 2048u;
    for (uint32_t t = 0; t < 8u; ++t) {
      int16_t *tab = tables + ((size_t)kt * 8u + t) * 64u;
#if defined(__hexagon__)
      const HVX_Vector a0 =
        Q6_V_vmux_QVV(lower, Q6_Vh_vsplat_R(((int)a[2u * t] - center)),
                      Q6_Vh_vsplat_R(((int)a[16u + 2u * t] - center)));
      const HVX_Vector a1 =
        Q6_V_vmux_QVV(lower, Q6_Vh_vsplat_R(((int)a[2u * t + 1u] - center)),
                      Q6_Vh_vsplat_R(((int)a[17u + 2u * t] - center)));
      *(HVX_Vector *)tab =
        Q6_Vh_vmpyiacc_VhVhVh(Q6_Vh_vmpyi_VhVh(a0, vp0), a1, vp1);
#else
      for (uint32_t h = 0; h < 2u; ++h) {
        for (uint32_t i = 0; i < 16u; ++i) {
          const int16_t v =
            ((int)a[h * 16u + 2u * t] - center) * palette[i & 3u] +
            ((int)a[h * 16u + 2u * t + 1u] - center) * palette[i >> 2u];
          tab[h * 32u + 2u * i] = v;
          tab[h * 32u + 2u * i + 1u] = v;
        }
      }
#endif
    }
  }
}

void hvx_i2_lut_prepare(const uint8_t *act_ah, const int8_t *palette,
                        uint32_t kt_begin, uint32_t kt_end, int16_t *tables) {
  lut_prepare(act_ah, palette, kt_begin, kt_end, tables, 0);
}

void hvx_i2_lut_prepare_centered(const uint8_t *act_ah, const int8_t *palette,
                                 uint32_t kt_begin, uint32_t kt_end,
                                 int16_t *tables) {
  lut_prepare(act_ah, palette, kt_begin, kt_end, tables, 127);
}

void hvx_gemv_u8i2_lut_panel(const uint8_t *packed, uint32_t K, uint32_t panel,
                             const int16_t *tables, int32_t *out) {
  hvx_gemv_u8i2_lut_strided(packed + (size_t)panel * (K / 32u) * 1024u, K / 32u,
                            1024u, tables, out);
}

void hvx_gemv_u8i2_lut_strided(const uint8_t *src, uint32_t ktiles,
                               uint32_t k_stride, const int16_t *tables,
                               int32_t *out) {
#if defined(__hexagon__)
#if !NNTR_LUT_USE_NOMATCH
  const HVX_Vector mask = Q6_V_vsplat_R(0x0f0f0f0f);
  const HVX_Vector select = Q6_V_vsplat_R(0x10101010);
#endif
  HVX_VectorPair even = Q6_W_vcombine_VV(Q6_V_vzero(), Q6_V_vzero());
  HVX_VectorPair odd = even;
  for (uint32_t kt = 0; kt < ktiles; ++kt) {
    HVX_Vector le = Q6_V_vzero(), lo = le, he = le, ho = le;
#if NNTR_LUT_USE_NOMATCH
#pragma clang loop unroll_count(NNTR_LUT_SINGLE_UNROLL)
#endif
    for (uint32_t t = 0; t < 8u; ++t) {
      const HVX_Vector codes =
        *(const HVX_Vector *)(src + (size_t)kt * k_stride + t * 128u);
      const HVX_Vector tab =
        *(const HVX_Vector *)(tables + ((size_t)kt * 8u + t) * 64u);
#if NNTR_LUT_USE_NOMATCH
      const HVX_VectorPair l = Q6_Wh_vlut16_VbVhR_nomatch(codes, tab, 0);
      const HVX_VectorPair h =
        Q6_Wh_vlut16_VbVhR_nomatch(Q6_Vuh_vlsr_VuhR(codes, 4), tab, 1);
#else
      const HVX_Vector il = Q6_V_vand_VV(codes, mask);
      const HVX_Vector ih =
        Q6_V_vor_VV(Q6_V_vand_VV(Q6_Vuh_vlsr_VuhR(codes, 4), mask), select);
      const HVX_VectorPair l = Q6_Wh_vlut16_VbVhI(il, tab, 0);
      const HVX_VectorPair h = Q6_Wh_vlut16_VbVhI(ih, tab, 1);
#endif
      le = Q6_Vh_vadd_VhVh(le, Q6_V_lo_W(l));
      lo = Q6_Vh_vadd_VhVh(lo, Q6_V_hi_W(l));
      he = Q6_Vh_vadd_VhVh(he, Q6_V_lo_W(h));
      ho = Q6_Vh_vadd_VhVh(ho, Q6_V_hi_W(h));
    }
    even = Q6_Ww_vaddacc_WwVhVh(even, le, he);
    odd = Q6_Ww_vaddacc_WwVhVh(odd, lo, ho);
  }
  *(HVX_Vector *)(out + 0u) = Q6_V_lo_W(even);
  *(HVX_Vector *)(out + 32u) = Q6_V_lo_W(odd);
  *(HVX_Vector *)(out + 64u) = Q6_V_hi_W(even);
  *(HVX_Vector *)(out + 96u) = Q6_V_hi_W(odd);
#else
  memset(out, 0, 128u * sizeof(*out));
  for (uint32_t kt = 0; kt < ktiles; ++kt) {
    for (uint32_t t = 0; t < 8u; ++t) {
      const int16_t *tab = tables + ((size_t)kt * 8u + t) * 64u;
      for (uint32_t j = 0; j < 128u; ++j) {
        const uint8_t c = src[(size_t)kt * k_stride + t * 128u + j];
        out[(j % 4u) * 32u + j / 4u] +=
          tab[2u * (c & 15u)] + tab[32u + 2u * (c >> 4u)];
      }
    }
  }
#endif
}

/** Measured V81 register schedule. The optimum is not monotonic in the
 * unroll count: bounded and fully unrolled versions trade spills for ILP. */
#ifndef NNTR_LUT_PAIR_UNROLL
#define NNTR_LUT_PAIR_UNROLL 8
#endif
void hvx_gemv_u8i2_lut_pair(const uint8_t *src0, const uint8_t *src1,
                            uint32_t ktiles, uint32_t k_stride,
                            const int16_t *tables, int32_t *out0,
                            int32_t *out1) {
#if defined(__hexagon__)
#if !NNTR_LUT_USE_NOMATCH
  const HVX_Vector mask = Q6_V_vsplat_R(0x0f0f0f0f);
  const HVX_Vector select = Q6_V_vsplat_R(0x10101010);
#endif
  const HVX_Vector zero = Q6_V_vzero();
  HVX_VectorPair even0 = Q6_W_vcombine_VV(zero, zero), odd0 = even0;
  HVX_VectorPair even1 = even0, odd1 = even0;
  for (uint32_t kt = 0; kt < ktiles; ++kt) {
    HVX_Vector le0 = zero, lo0 = zero, he0 = zero, ho0 = zero;
    HVX_Vector le1 = zero, lo1 = zero, he1 = zero, ho1 = zero;
#pragma clang loop unroll_count(NNTR_LUT_PAIR_UNROLL)
    for (uint32_t t = 0; t < 8u; ++t) {
      const HVX_Vector tab =
        *(const HVX_Vector *)(tables + ((size_t)kt * 8u + t) * 64u);
      {
        const HVX_Vector codes =
          *(const HVX_Vector *)(src0 + (size_t)kt * k_stride + t * 128u);
#if NNTR_LUT_USE_NOMATCH
        const HVX_VectorPair l = Q6_Wh_vlut16_VbVhR_nomatch(codes, tab, 0);
        const HVX_VectorPair h =
          Q6_Wh_vlut16_VbVhR_nomatch(Q6_Vuh_vlsr_VuhR(codes, 4), tab, 1);
#else
        const HVX_Vector il = Q6_V_vand_VV(codes, mask);
        const HVX_Vector ih =
          Q6_V_vor_VV(Q6_V_vand_VV(Q6_Vuh_vlsr_VuhR(codes, 4), mask), select);
        const HVX_VectorPair l = Q6_Wh_vlut16_VbVhI(il, tab, 0);
        const HVX_VectorPair h = Q6_Wh_vlut16_VbVhI(ih, tab, 1);
#endif
        le0 = Q6_Vh_vadd_VhVh(le0, Q6_V_lo_W(l));
        lo0 = Q6_Vh_vadd_VhVh(lo0, Q6_V_hi_W(l));
        he0 = Q6_Vh_vadd_VhVh(he0, Q6_V_lo_W(h));
        ho0 = Q6_Vh_vadd_VhVh(ho0, Q6_V_hi_W(h));
      }
      {
        const HVX_Vector codes =
          *(const HVX_Vector *)(src1 + (size_t)kt * k_stride + t * 128u);
#if NNTR_LUT_USE_NOMATCH
        const HVX_VectorPair l = Q6_Wh_vlut16_VbVhR_nomatch(codes, tab, 0);
        const HVX_VectorPair h =
          Q6_Wh_vlut16_VbVhR_nomatch(Q6_Vuh_vlsr_VuhR(codes, 4), tab, 1);
#else
        const HVX_Vector il = Q6_V_vand_VV(codes, mask);
        const HVX_Vector ih =
          Q6_V_vor_VV(Q6_V_vand_VV(Q6_Vuh_vlsr_VuhR(codes, 4), mask), select);
        const HVX_VectorPair l = Q6_Wh_vlut16_VbVhI(il, tab, 0);
        const HVX_VectorPair h = Q6_Wh_vlut16_VbVhI(ih, tab, 1);
#endif
        le1 = Q6_Vh_vadd_VhVh(le1, Q6_V_lo_W(l));
        lo1 = Q6_Vh_vadd_VhVh(lo1, Q6_V_hi_W(l));
        he1 = Q6_Vh_vadd_VhVh(he1, Q6_V_lo_W(h));
        ho1 = Q6_Vh_vadd_VhVh(ho1, Q6_V_hi_W(h));
      }
    }
    even0 = Q6_Ww_vaddacc_WwVhVh(even0, le0, he0);
    odd0 = Q6_Ww_vaddacc_WwVhVh(odd0, lo0, ho0);
    even1 = Q6_Ww_vaddacc_WwVhVh(even1, le1, he1);
    odd1 = Q6_Ww_vaddacc_WwVhVh(odd1, lo1, ho1);
  }
  *(HVX_Vector *)(out0 + 0u) = Q6_V_lo_W(even0);
  *(HVX_Vector *)(out0 + 32u) = Q6_V_lo_W(odd0);
  *(HVX_Vector *)(out0 + 64u) = Q6_V_hi_W(even0);
  *(HVX_Vector *)(out0 + 96u) = Q6_V_hi_W(odd0);
  *(HVX_Vector *)(out1 + 0u) = Q6_V_lo_W(even1);
  *(HVX_Vector *)(out1 + 32u) = Q6_V_lo_W(odd1);
  *(HVX_Vector *)(out1 + 64u) = Q6_V_hi_W(even1);
  *(HVX_Vector *)(out1 + 96u) = Q6_V_hi_W(odd1);
#else
  hvx_gemv_u8i2_lut_strided(src0, ktiles, k_stride, tables, out0);
  hvx_gemv_u8i2_lut_strided(src1, ktiles, k_stride, tables, out1);
#endif
}

void hvx_i2_lut_prepare_dense(const uint8_t *act_ah, const int8_t *palette,
                              uint32_t kt_begin, uint32_t kt_end,
                              int16_t *tables) {
#if defined(__hexagon__)
  int16_t p0[64] __attribute__((aligned(128)));
  int16_t p1[64] __attribute__((aligned(128)));
  for (uint32_t j = 0; j < 64u; ++j) {
    p0[j] = palette[(j / 2u) & 3u];
    p1[j] = palette[((j / 2u) % 16u) >> 2u];
  }
  const HVX_Vector vp0 = *(const HVX_Vector *)p0;
  const HVX_Vector vp1 = *(const HVX_Vector *)p1;
  const HVX_VectorPred lower = Q6_Q_vsetq_R(64);
#endif
  for (uint32_t kt = kt_begin; kt < kt_end; ++kt) {
    const uint8_t *a = act_ah + (size_t)kt * 2048u;
    for (uint32_t g = 0; g < 4u; ++g) {
      int16_t *tab = tables + ((size_t)kt * 4u + g) * 64u;
#if defined(__hexagon__)
      const uint32_t low0 = a[4u * g] | ((uint32_t)a[4u * g + 2u] << 16u);
      const uint32_t high0 =
        a[16u + 4u * g] | ((uint32_t)a[18u + 4u * g] << 16u);
      const uint32_t low1 = a[4u * g + 1u] | ((uint32_t)a[4u * g + 3u] << 16u);
      const uint32_t high1 =
        a[17u + 4u * g] | ((uint32_t)a[19u + 4u * g] << 16u);
      const HVX_Vector a0 =
        Q6_V_vmux_QVV(lower, Q6_V_vsplat_R(low0), Q6_V_vsplat_R(high0));
      const HVX_Vector a1 =
        Q6_V_vmux_QVV(lower, Q6_V_vsplat_R(low1), Q6_V_vsplat_R(high1));
      *(HVX_Vector *)tab =
        Q6_Vh_vmpyiacc_VhVhVh(Q6_Vh_vmpyi_VhVh(a0, vp0), a1, vp1);
#else
      for (uint32_t h = 0; h < 2u; ++h)
        for (uint32_t t = 0; t < 2u; ++t)
          for (uint32_t i = 0; i < 16u; ++i)
            tab[h * 32u + 2u * i + t] =
              a[h * 16u + 4u * g + 2u * t] * palette[i & 3u] +
              a[h * 16u + 4u * g + 2u * t + 1u] * palette[i >> 2u];
#endif
    }
  }
}
#ifndef NNTR_LUT_DENSE_UNROLL
#define NNTR_LUT_DENSE_UNROLL 1
#endif
void hvx_gemv_u8i2_lut_dense_pair(const uint8_t *src0, const uint8_t *src1,
                                  uint32_t ktiles, uint32_t k_stride,
                                  const int16_t *tables, int32_t *out0,
                                  int32_t *out1) {
#if defined(__hexagon__)
#if !NNTR_LUT_USE_NOMATCH
  const HVX_Vector mask = Q6_V_vsplat_R(0x0f0f0f0f);

#endif
  const HVX_Vector zero = Q6_V_vzero();
  HVX_VectorPair even0 = Q6_W_vcombine_VV(zero, zero), odd0 = even0;
  HVX_VectorPair even1 = even0, odd1 = even0;
  for (uint32_t kt = 0; kt < ktiles; ++kt) {
    HVX_Vector le0 = zero, lo0 = zero, he0 = zero, ho0 = zero;
    HVX_Vector le1 = zero, lo1 = zero, he1 = zero, ho1 = zero;
#pragma clang loop unroll_count(NNTR_LUT_DENSE_UNROLL)
    for (uint32_t t = 0; t < 8u; ++t) {
      const uint32_t half = t % 2u;
      const HVX_Vector tab =
        *(const HVX_Vector *)(tables + ((size_t)kt * 4u + t / 2u) * 64u);
      {
        const HVX_Vector codes =
          *(const HVX_Vector *)(src0 + (size_t)kt * k_stride + t * 128u);
#if NNTR_LUT_USE_NOMATCH
        const HVX_VectorPair l =
          Q6_Wh_vlut16_VbVhR_nomatch(codes, tab, 2u * half);
        const HVX_VectorPair h = Q6_Wh_vlut16_VbVhR_nomatch(
          Q6_Vuh_vlsr_VuhR(codes, 4), tab, 2u * half + 1u);
#else
        const HVX_Vector il = Q6_V_vor_VV(Q6_V_vand_VV(codes, mask),
                                          Q6_V_vsplat_R(half * 0x20202020));
        const HVX_Vector ih =
          Q6_V_vor_VV(Q6_V_vand_VV(Q6_Vuh_vlsr_VuhR(codes, 4), mask),
                      Q6_V_vsplat_R(half * 0x20202020 + 0x10101010));
        const HVX_VectorPair l = Q6_Wh_vlut16_VbVhR(il, tab, 2u * half);
        const HVX_VectorPair h = Q6_Wh_vlut16_VbVhR(ih, tab, 2u * half + 1u);
#endif
        le0 = Q6_Vh_vadd_VhVh(le0, Q6_V_lo_W(l));
        lo0 = Q6_Vh_vadd_VhVh(lo0, Q6_V_hi_W(l));
        he0 = Q6_Vh_vadd_VhVh(he0, Q6_V_lo_W(h));
        ho0 = Q6_Vh_vadd_VhVh(ho0, Q6_V_hi_W(h));
      }
      {
        const HVX_Vector codes =
          *(const HVX_Vector *)(src1 + (size_t)kt * k_stride + t * 128u);
#if NNTR_LUT_USE_NOMATCH
        const HVX_VectorPair l =
          Q6_Wh_vlut16_VbVhR_nomatch(codes, tab, 2u * half);
        const HVX_VectorPair h = Q6_Wh_vlut16_VbVhR_nomatch(
          Q6_Vuh_vlsr_VuhR(codes, 4), tab, 2u * half + 1u);
#else
        const HVX_Vector il = Q6_V_vor_VV(Q6_V_vand_VV(codes, mask),
                                          Q6_V_vsplat_R(half * 0x20202020));
        const HVX_Vector ih =
          Q6_V_vor_VV(Q6_V_vand_VV(Q6_Vuh_vlsr_VuhR(codes, 4), mask),
                      Q6_V_vsplat_R(half * 0x20202020 + 0x10101010));
        const HVX_VectorPair l = Q6_Wh_vlut16_VbVhR(il, tab, 2u * half);
        const HVX_VectorPair h = Q6_Wh_vlut16_VbVhR(ih, tab, 2u * half + 1u);
#endif
        le1 = Q6_Vh_vadd_VhVh(le1, Q6_V_lo_W(l));
        lo1 = Q6_Vh_vadd_VhVh(lo1, Q6_V_hi_W(l));
        he1 = Q6_Vh_vadd_VhVh(he1, Q6_V_lo_W(h));
        ho1 = Q6_Vh_vadd_VhVh(ho1, Q6_V_hi_W(h));
      }
    }
    even0 = Q6_Ww_vaddacc_WwVhVh(even0, le0, he0);
    odd0 = Q6_Ww_vaddacc_WwVhVh(odd0, lo0, ho0);
    even1 = Q6_Ww_vaddacc_WwVhVh(even1, le1, he1);
    odd1 = Q6_Ww_vaddacc_WwVhVh(odd1, lo1, ho1);
  }
  *(HVX_Vector *)(out0 + 0u) = Q6_V_lo_W(even0);
  *(HVX_Vector *)(out0 + 32u) = Q6_V_lo_W(odd0);
  *(HVX_Vector *)(out0 + 64u) = Q6_V_hi_W(even0);
  *(HVX_Vector *)(out0 + 96u) = Q6_V_hi_W(odd0);
  *(HVX_Vector *)(out1 + 0u) = Q6_V_lo_W(even1);
  *(HVX_Vector *)(out1 + 32u) = Q6_V_lo_W(odd1);
  *(HVX_Vector *)(out1 + 64u) = Q6_V_hi_W(even1);
  *(HVX_Vector *)(out1 + 96u) = Q6_V_hi_W(odd1);
#else
  for (uint32_t panel = 0; panel < 2u; ++panel) {
    const uint8_t *src = panel ? src1 : src0;
    int32_t *out = panel ? out1 : out0;
    memset(out, 0, 128u * sizeof(*out));
    for (uint32_t kt = 0; kt < ktiles; ++kt)
      for (uint32_t t = 0; t < 8u; ++t) {
        const int16_t *tab = tables + ((size_t)kt * 4u + t / 2u) * 64u + t % 2u;
        for (uint32_t j = 0; j < 128u; ++j) {
          const uint8_t c = src[(size_t)kt * k_stride + t * 128u + j];
          out[(j % 4u) * 32u + j / 4u] +=
            tab[2u * (c & 15u)] + tab[32u + 2u * (c >> 4u)];
        }
      }
  }
#endif
}

/* With a-127 in [-127,128], 32 signed-int4 products sum to
 * [-32768,32512]. Combining both LUT halves in int16 is therefore exact.
 * The caller must use zp-127 in its dequantization, or restore 127*colsum. */
#ifndef NNTR_LUT_CENTER_UNROLL
#define NNTR_LUT_CENTER_UNROLL 2
#endif
void hvx_gemv_u8i2_lut_pair_centered(const uint8_t *src0, const uint8_t *src1,
                                     uint32_t ktiles, uint32_t k_stride,
                                     const int16_t *tables, int32_t *out0,
                                     int32_t *out1) {
#if defined(__hexagon__)
#if !NNTR_LUT_USE_NOMATCH
  const HVX_Vector mask = Q6_V_vsplat_R(0x0f0f0f0f);
  const HVX_Vector select = Q6_V_vsplat_R(0x10101010);
#endif
  const HVX_Vector zero = Q6_V_vzero();
  HVX_VectorPair even0 = Q6_W_vcombine_VV(zero, zero), odd0 = even0;
  HVX_VectorPair even1 = even0, odd1 = even0;
  for (uint32_t kt = 0; kt < ktiles; ++kt) {
    HVX_VectorPair sum0 = Q6_W_vcombine_VV(zero, zero), sum1 = sum0;
#pragma clang loop unroll_count(NNTR_LUT_CENTER_UNROLL)
    for (uint32_t t = 0; t < 8u; ++t) {
      const HVX_Vector tab =
        *(const HVX_Vector *)(tables + ((size_t)kt * 8u + t) * 64u);
      {
        const HVX_Vector codes =
          *(const HVX_Vector *)(src0 + (size_t)kt * k_stride + t * 128u);
#if NNTR_LUT_USE_NOMATCH
        const HVX_VectorPair l = Q6_Wh_vlut16_VbVhR_nomatch(codes, tab, 0);
        const HVX_VectorPair h =
          Q6_Wh_vlut16_VbVhR_nomatch(Q6_Vuh_vlsr_VuhR(codes, 4), tab, 1);
#else
        const HVX_Vector il = Q6_V_vand_VV(codes, mask);
        const HVX_Vector ih =
          Q6_V_vor_VV(Q6_V_vand_VV(Q6_Vuh_vlsr_VuhR(codes, 4), mask), select);
        const HVX_VectorPair l = Q6_Wh_vlut16_VbVhI(il, tab, 0);
        const HVX_VectorPair h = Q6_Wh_vlut16_VbVhI(ih, tab, 1);
#endif
        sum0 = Q6_Wh_vadd_WhWh(sum0, l);
        sum0 = Q6_Wh_vadd_WhWh(sum0, h);
      }
      {
        const HVX_Vector codes =
          *(const HVX_Vector *)(src1 + (size_t)kt * k_stride + t * 128u);
#if NNTR_LUT_USE_NOMATCH
        const HVX_VectorPair l = Q6_Wh_vlut16_VbVhR_nomatch(codes, tab, 0);
        const HVX_VectorPair h =
          Q6_Wh_vlut16_VbVhR_nomatch(Q6_Vuh_vlsr_VuhR(codes, 4), tab, 1);
#else
        const HVX_Vector il = Q6_V_vand_VV(codes, mask);
        const HVX_Vector ih =
          Q6_V_vor_VV(Q6_V_vand_VV(Q6_Vuh_vlsr_VuhR(codes, 4), mask), select);
        const HVX_VectorPair l = Q6_Wh_vlut16_VbVhI(il, tab, 0);
        const HVX_VectorPair h = Q6_Wh_vlut16_VbVhI(ih, tab, 1);
#endif
        sum1 = Q6_Wh_vadd_WhWh(sum1, l);
        sum1 = Q6_Wh_vadd_WhWh(sum1, h);
      }
    }
    even0 = Q6_Ww_vaddacc_WwVhVh(even0, Q6_V_lo_W(sum0), zero);
    odd0 = Q6_Ww_vaddacc_WwVhVh(odd0, Q6_V_hi_W(sum0), zero);
    even1 = Q6_Ww_vaddacc_WwVhVh(even1, Q6_V_lo_W(sum1), zero);
    odd1 = Q6_Ww_vaddacc_WwVhVh(odd1, Q6_V_hi_W(sum1), zero);
  }
  *(HVX_Vector *)(out0 + 0u) = Q6_V_lo_W(even0);
  *(HVX_Vector *)(out0 + 32u) = Q6_V_lo_W(odd0);
  *(HVX_Vector *)(out0 + 64u) = Q6_V_hi_W(even0);
  *(HVX_Vector *)(out0 + 96u) = Q6_V_hi_W(odd0);
  *(HVX_Vector *)(out1 + 0u) = Q6_V_lo_W(even1);
  *(HVX_Vector *)(out1 + 32u) = Q6_V_lo_W(odd1);
  *(HVX_Vector *)(out1 + 64u) = Q6_V_hi_W(even1);
  *(HVX_Vector *)(out1 + 96u) = Q6_V_hi_W(odd1);
#else
  hvx_gemv_u8i2_lut_strided(src0, ktiles, k_stride, tables, out0);
  hvx_gemv_u8i2_lut_strided(src1, ktiles, k_stride, tables, out1);
#endif
}

void hvx_i2_lut_restore(int32_t *out, const int32_t *colsum, uint32_t N) {
#if defined(__hexagon__)
  for (uint32_t n = 0; n < N; n += 32u) {
    const HVX_Vector cs = *(const HVX_UVector *)(colsum + n);
    const HVX_Vector correction = Q6_Vw_vsub_VwVw(Q6_Vw_vasl_VwR(cs, 7), cs);
    *(HVX_Vector *)(out + n) =
      Q6_Vw_vadd_VwVw(*(HVX_Vector *)(out + n), correction);
  }
#else
  for (uint32_t n = 0; n < N; ++n)
    out[n] = (int32_t)((int64_t)out[n] + 127 * (int64_t)colsum[n]);
#endif
}

void hvx_gemv_u8i2_wh_fused_col(const uint8_t *act_ah, uint32_t k_tiles,
                                const uint8_t *wh, uint32_t n_col, uint32_t nt,
                                const uint8_t *table, int32_t *out) {
  hvx_gemv_u8i2_wh_fused_strided(act_ah, k_tiles, wh, n_col * 256u, nt, table,
                                 out);
}

void hvx_gemv_u8i2_wh_fused_strided(const uint8_t *act_ah, uint32_t k_tiles,
                                    const uint8_t *wh, uint32_t k_stride,
                                    uint32_t nt, const uint8_t *table,
                                    int32_t *out) {
#if defined(__hexagon__)
  const HVX_Vector tab = *(const HVX_Vector *)table;
  const HVX_Vector m0f = Q6_V_vsplat_R(0x0f0f0f0f);
  const HVX_Vector mf0 = Q6_V_vsplat_R((int)0xf0f0f0f0u);
  HVX_Vector acc = Q6_V_vzero();
  for (uint32_t kt = 0; kt < k_tiles; ++kt) {
    const uint8_t *tile = wh + (size_t)kt * k_stride + nt * 256u;
    const uint32_t *a = (const uint32_t *)(act_ah + (size_t)kt * 2048u);
    for (uint32_t p = 0; p < 2u; ++p) {
      const HVX_Vector codes = *(const HVX_Vector *)(tile + p * 128u);
      const HVX_Vector v0 =
        Q6_Vb_vlut32_VbVbI(Q6_V_vand_VV(codes, m0f), tab, 0);
      const HVX_Vector v1 = Q6_Vb_vlut32_VbVbI(
        Q6_V_vand_VV(Q6_Vuh_vlsr_VuhR(codes, 4), m0f), tab, 0);
      acc = Q6_Vw_vrmpyacc_VwVubVb(acc, Q6_V_vsplat_R((int)a[4u * p]),
                                   Q6_V_vand_VV(Q6_Vh_vasl_VhR(v0, 4), mf0));
      acc = Q6_Vw_vrmpyacc_VwVubVb(acc, Q6_V_vsplat_R((int)a[4u * p + 1u]),
                                   Q6_V_vand_VV(v0, mf0));
      acc = Q6_Vw_vrmpyacc_VwVubVb(acc, Q6_V_vsplat_R((int)a[4u * p + 2u]),
                                   Q6_V_vand_VV(Q6_Vh_vasl_VhR(v1, 4), mf0));
      acc = Q6_Vw_vrmpyacc_VwVubVb(acc, Q6_V_vsplat_R((int)a[4u * p + 3u]),
                                   Q6_V_vand_VV(v1, mf0));
    }
  }
  *(HVX_Vector *)out = Q6_Vw_vasr_VwR(acc, 4);
#else
  memset(out, 0, 32u * sizeof(*out));
  for (uint32_t k = 0; k < k_tiles * 32u; ++k) {
    for (uint32_t c = 0; c < 32u; ++c) {
      uint32_t shift;
      const size_t off = (size_t)(k / 32u) * k_stride + nt * 256u +
                         wh_code_offset(k % 32u, c, 32u, &shift);
      const uint32_t code = (wh[off] >> shift) & 3u;
      const uint8_t nib = table[2u * code] & 15u;
      const int w = nib > 7u ? (int)nib - 16 : nib;
      out[c] += act_ah[(size_t)(k / 32u) * 2048u + k % 32u] * w;
    }
  }
#endif
}
