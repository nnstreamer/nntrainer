// SPDX-License-Identifier: Apache-2.0
/**
 * @file moe_epilogue_host_check.c
 * @brief Exact ordered DN dequant/scatter with alias, tail, and FP edge
 * coverage.
 * @author NNTrainer contributors
 * @bug See the repository issue tracker for known issues.
 */

#include "hvx_dequant_i32.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define DECL(fn)                                                               \
  void fn(const int32_t *, uint32_t, uint32_t, const float *, const int32_t *, \
          const int32_t *, const float *, const float *, float *, uint32_t,    \
          int)
static DECL(reference_tile);
static uint32_t state = 849917;
static uint32_t next(void) { return state = 1664525u * state + 1013904223u; }

/** Independent C scalar expressions may select a different quiet-NaN payload
 * from the lane model, especially with sanitizer instrumentation. Require exact
 * bits for finite values, infinities and signed zero. Permit NaN classification
 * only at a written output lane; all other storage must remain byte exact. */
static int same_output(const void *reference, const void *actual, size_t bytes,
                       const float *output, uint32_t count, uint32_t stride,
                       const uint32_t *rows) {
  const uint8_t *a = reference, *b = actual;
  for (size_t i = 0; i < bytes; i += sizeof(float)) {
    uint32_t expected, got;
    memcpy(&expected, a + i, sizeof expected);
    memcpy(&got, b + i, sizeof got);
    if (expected == got)
      continue;
    if ((expected & 0x7fffffffu) <= 0x7f800000u ||
        (got & 0x7fffffffu) <= 0x7f800000u)
      return 0;
    int written = 0;
    for (uint32_t r = 0; r < count; ++r) {
      const uintptr_t begin =
        (uintptr_t)(output + (size_t)(rows ? rows[r] : r) * stride);
      const uintptr_t address = (uintptr_t)(a + i);
      written |= address >= begin && address - begin < 32u * sizeof(float);
    }
    if (!written)
      return 0;
  }
  return 1;
}

static unsigned cases;
static void run(unsigned m, unsigned stride, unsigned offset, unsigned layout,
                int acc, unsigned alias, unsigned edge) {
  const size_t bytes = ((size_t)80 * stride * 4 + 16384 + 127) & ~(size_t)127;
  uint8_t *a = aligned_alloc(128, bytes), *b = aligned_alloc(128, bytes),
          *ta = aligned_alloc(128, bytes), *tb = aligned_alloc(128, bytes);
  if (!a || !b || !ta || !tb)
    abort();
  float as[64], ws[32], bias[32];
  int32_t zp[64], cs[32];
  for (size_t j = 0; j < bytes / 4; ++j) {
    float f = ((int)(next() % 1001) - 500) * .03125f;
    memcpy(a + j * 4, &f, 4);
  }
  memcpy(b, a, bytes);
  memset(ta, 0, bytes);
  memset(tb, 0, bytes);
  unsigned row_stride = layout == 0 ? 0 : layout == 1 ? 32 : 35;
  const uint32_t raw_edges[] = {0,          1,          0x7fffffff, 0x80000000,
                                0xffffffff, 0x01020304, 0xf0f1f2f3};
  for (unsigned r = 0; r < 64; ++r)
    for (unsigned n = 0; n < 32; ++n) {
      uint32_t z = edge ? raw_edges[(r + n) % 7] : next();
      if (!row_stride) {
        for (unsigned p = 0; p < 4; ++p)
          ta[p * 2048 + r * 32 + n] = (z >> (p * 8));
      } else
        memcpy(ta + (r * row_stride + n) * 4, &z, 4);
    }
  memcpy(tb, ta, bytes);
  const uint32_t float_edges[] = {0,          0x80000000, 1,
                                  0x00800001, 0x3f800000, 0xbf800000,
                                  0x7f800000, 0xff800000, 0x7fc01234};
  for (unsigned r = 0; r < 64; ++r) {
    as[r] = (r % 7 + 1) * 1e-5f;
    zp[r] = (int)r - 31;
    if (edge)
      memcpy(as + r, float_edges + r % 9, 4);
  }
  for (unsigned n = 0; n < 32; ++n) {
    ws[n] = (n % 5 + 1) * 1e-4f;
    bias[n] = (n % 3 - 1.0f) * .03f;
    cs[n] = (int)n * 257 - 4000;
    if (edge) {
      memcpy(ws + n, float_edges + (n + 3) % 9, 4);
      memcpy(bias + n, float_edges + (n + 5) % 9, 4);
    }
  }
  float *oa = (float *)a + offset, *ob = (float *)b + offset;
  const float *sa = as, *sb = as;
  const int32_t *za = zp, *zb = zp;
  if (alias == 1 || alias == 2) {
    oa = (float *)ta + (alias == 2 ? 32 : 0);
    ob = (float *)tb + (alias == 2 ? 32 : 0);
  }
  if (alias == 3) {
    sa = oa;
    sb = ob;
    memcpy(oa, as, sizeof as);
    memcpy(ob, as, sizeof as);
  }
  if (alias == 4) {
    za = (int32_t *)oa;
    zb = (int32_t *)ob;
    memcpy(oa, zp, sizeof zp);
    memcpy(ob, zp, sizeof zp);
  }
  uint8_t *fallback = NULL;
  if (!alias && !row_stride) {
    fallback = aligned_alloc(128, bytes + 128u);
    if (!fallback)
      abort();
    memcpy(fallback + sizeof(float), b, bytes);
    hvx_dequant_acc_tile_to_f32((int32_t *)tb, row_stride, m, sb, zb, cs, ws,
                                bias, (float *)fallback + 1u + offset, stride,
                                acc);
  }
  reference_tile((int32_t *)ta, row_stride, m, sa, za, cs, ws, bias, oa, stride,
                 acc);
  hvx_dequant_acc_tile_to_f32((int32_t *)tb, row_stride, m, sb, zb, cs, ws,
                              bias, ob, stride, acc);
  if ((fallback && memcmp(b, fallback + sizeof(float), bytes)) ||
      !same_output(a, b, bytes, oa, m, stride, NULL) ||
      !same_output(ta, tb, bytes, oa, m, stride, NULL)) {
    for (size_t i = 0; i < bytes / 4; ++i) {
      uint32_t want, got;
      memcpy(&want, a + i * 4, 4);
      memcpy(&got, b + i * 4, 4);
      if (want != got) {
        fprintf(stderr,
                "first output difference word=%zu expected=%08x actual=%08x\n",
                i, want, got);
        break;
      }
    }
    fprintf(stderr,
            "FAIL rows=%u stride=%u off=%u layout=%u acc=%d alias=%u edge=%u\n",
            m, stride, offset, layout, acc, alias, edge);
    exit(1);
  }
  free(a);
  free(b);
  free(ta);
  free(tb);
  free(fallback);
  ++cases;
}
static int check_dequant(void) {
  const unsigned rows[] = {0, 1, 2, 3, 4, 5, 7, 31, 32, 33, 60, 61, 62, 63, 64},
                 strides[] = {32, 64, 113, 2048};
  for (unsigned r = 0; r < 15; ++r)
    for (unsigned s = 0; s < 4; ++s)
      for (unsigned off = 0; off < 2; ++off)
        for (unsigned layout = 0; layout < 3; ++layout)
          for (unsigned acc = 0; acc < 2; ++acc)
            for (unsigned edge = 0; edge < 2; ++edge)
              run(rows[r], strides[s], off, layout, acc, 0, edge);
  for (unsigned m = 0; m <= 64; ++m)
    for (unsigned alias = 1; alias < 5; ++alias)
      for (unsigned acc = 0; acc < 2; ++acc)
        run(m, 32, 0, 0, acc, alias, 0);
  printf("DEQUANT_FOUR_HOST_PASS cases=%u "
         "all_row_counts_0_to_64_planar_and_rowmajor_aligned_unaligned_"
         "accumulation_tile_and_parameter_alias_fp_edges=PASS "
         "same_HVX_fallback_bits=EXACT scalar_NaN=classification\n",
         cases);
  return 0;
}

#include "hvx_scale_add_f32.h"
static void reference_scatter(float *, const float *, const uint32_t *,
                              const float *, uint32_t, uint32_t);
static int check_scatter(void) {
  const unsigned strides[] = {32, 64, 128, 2048, 113}, counts[] = {0, 1, 3, 64};
  const uint32_t edges[] = {0,          0x80000000, 0x7f800000, 0xff800000,
                            0x7fc01234, 0x00000001, 0x00800001};
  unsigned cases = 0, seed = 28497;
  for (unsigned st = 0; st < 5; ++st)
    for (unsigned ct = 0; ct < 4; ++ct)
      for (unsigned off = 0; off < 2; ++off)
        for (unsigned alias = 0; alias < 4; ++alias)
          for (unsigned routing = 0; routing < 3; ++routing) {
            unsigned n = strides[st], size = (80 * n + 256) * 4, src_off = off,
                     out_off = alias == 2   ? 32
                               : alias == 3 ? 4
                                            : off;
            float *a = aligned_alloc(128, (size + 127) & ~127),
                  *b = aligned_alloc(128, (size + 127) & ~127),
                  *x = aligned_alloc(128, (size + 127) & ~127);
            for (unsigned j = 0; j < size / 4; ++j) {
              seed = 1664525u * seed + 1013904223u;
              float z = ((int)(seed % 251) - 125) * .03125f;
              a[j] = b[j] = x[j] = z;
            }
            uint32_t rows[64];
            float weights[64];
            for (unsigned j = 0; j < 64; ++j) {
              rows[j] = routing == 0 ? j : routing == 1 ? (63 - j) : j % 7;
              weights[j] = (j % 9 - 4.0f) * .2f;
            }
            for (unsigned j = 0; j < 7; ++j) {
              memcpy(a + src_off + j, edges + j, 4);
              memcpy(b + src_off + j, edges + j, 4);
              memcpy(x + src_off + j, edges + j, 4);
            }
            const float *xa = alias ? a + src_off : x + src_off,
                        *xb = alias ? b + src_off : x + src_off;
            float *fallback_base =
              aligned_alloc(128, ((size + 127u) & ~127u) + 128u);
            if (!fallback_base)
              abort();
            float *fallback = fallback_base + 1;
            memcpy(fallback, b, size);
            const float *xf = alias ? fallback + src_off : x + src_off;
            for (unsigned pass = 0; pass < 2; ++pass) {
              reference_scatter(a + out_off, xa, rows, weights, counts[ct], n);
              hvx_scale_add_scatter_tile_f32(b + out_off, xb, rows, weights,
                                             counts[ct], n);
              hvx_scale_add_scatter_tile_f32(fallback + out_off, xf, rows,
                                             weights, counts[ct], n);
              if (memcmp(b, fallback, size) ||
                  !same_output(a, b, size, a + out_off, counts[ct], n, rows)) {
                printf("SCATTER_FAIL stride=%u rows=%u offset=%u alias=%u "
                       "routing=%u pass=%u\n",
                       n, counts[ct], off, alias, routing, pass);
                return 1;
              }
            }
            free(a);
            free(b);
            free(x);
            free(fallback_base);
            cases++;
          }
  printf("SCATTER_ALIGNED_HOST_PASS cases=%u "
         "alias_exact_partial_disjoint_duplicates_reversed_rows_fp_edges=PASS "
         "same_HVX_fallback_bits=EXACT scalar_NaN=classification\n",
         cases);
  return 0;
}

/** Independent byte extraction and scalar rounded operations. Four planar
 * rows are consumed before writing a group, matching the public alias contract;
 * row parameters are still read immediately before each output row. */
static void reference_tile(const int32_t *tile, uint32_t row_stride,
                           uint32_t count, const float *scale,
                           const int32_t *zp, const int32_t *colsum,
                           const float *wscale, const float *bias, float *out,
                           uint32_t stride, int accumulate) {
  float columns[32], weights[32], offsets[32];
  for (unsigned c = 0; c < 32; ++c) {
    columns[c] = (float)colsum[c];
    weights[c] = wscale[c];
    offsets[c] = bias[c];
  }
  for (unsigned first = 0; first < count; first += 4) {
    int32_t raw[4][32];
    const unsigned n = count - first < 4 ? count - first : 4;
    for (unsigned r = 0; r < n; ++r) {
      for (unsigned c = 0; c < 32; ++c) {
        if (row_stride) {
          raw[r][c] = tile[(size_t)(first + r) * row_stride + c];
        } else {
          uint32_t bits = 0;
          const uint8_t *bytes = (const uint8_t *)tile;
          for (unsigned p = 0; p < 4; ++p)
            bits |= (uint32_t)bytes[p * 2048u + (first + r) * 32u + c]
                    << (8u * p);
          memcpy(&raw[r][c], &bits, sizeof bits);
        }
      }
    }
    for (unsigned r = 0; r < n; ++r) {
      const float s = scale[first + r], z = (float)zp[first + r];
      float result[32];
      for (unsigned c = 0; c < 32; ++c) {
        volatile float correction = z * columns[c];
        volatile float value = (float)raw[r][c] - correction;
        value = value * s;
        value = value * weights[c];
        value = value + offsets[c];
        if (accumulate)
          value = out[(size_t)(first + r) * stride + c] + value;
        result[c] = value;
      }
      memcpy(out + (size_t)(first + r) * stride, result, sizeof result);
    }
  }
}

/** Row contributions are observed in order. Snapshot each whole source and
 * destination vector before writing, including partial source/output aliases.
 */
static void reference_scatter(float *out, const float *src,
                              const uint32_t *rows, const float *weights,
                              uint32_t count, uint32_t stride) {
  for (unsigned r = 0; r < count; ++r) {
    float input[32], previous[32], result[32];
    float *dst = out + (size_t)rows[r] * stride;
    memcpy(input, src + (size_t)r * stride, sizeof input);
    memcpy(previous, dst, sizeof previous);
    for (unsigned c = 0; c < 32; ++c) {
      volatile float weighted = input[c] * weights[r];
      volatile float value = previous[c] + weighted;
      result[c] = value;
    }
    memcpy(dst, result, sizeof result);
  }
}
int main(void) { return check_dequant() || check_scatter(); }
