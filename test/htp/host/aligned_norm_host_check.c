// SPDX-License-Identifier: Apache-2.0
/** @file aligned_norm_host_check.c
 * @brief Exact graph publication and shared AH packing with real HVX emulation.
 * @author NNTrainer contributors
 * @bug See the repository issue tracker for known issues.
 */
#include "hvx_quant_u8.h"
#include "hvx_rmsnorm_rows_f32.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hvx_alignment_host_alloc.h"

static unsigned misaligned;
static void *test_alloc(size_t size) {
  return hvx_alignment_host_alloc(size, misaligned == 1u ? 4u : 0u);
}
static void test_free(void *p) { hvx_alignment_host_free(p); }

static uint32_t workers = 7u;
void hvx_worker_pool_run(hvx_worker_pool *pool, hvx_worker_pool_func f,
                         void *ctx, uint32_t units) {
  (void)pool;
  const uint32_t n = units < workers ? units : workers;
  for (uint32_t i = 0; i < n; ++i)
    f(n, i, ctx);
}
static void same(const char *what, const void *a, const void *b, size_t bytes,
                 uint32_t m, uint32_t k, uint32_t mode) {
  if (memcmp(a, b, bytes)) {
    fprintf(stderr, "FAIL %s M=%u K=%u mode=%u workers=%u\n", what, m, k, mode,
            workers);
    exit(1);
  }
}
int main(int argc, char **argv) {
  misaligned = argc > 1 ? (unsigned)atoi(argv[1]) : 0u;
  if (argc > 2 || misaligned > 1u)
    return 2;
  const uint32_t ms[] = {2, 3, 4, 5, 63, 64, 65, 129, 1024};
  const uint32_t ks[] = {128, 256, 2048};
  uint32_t seed = 8123, checked = 0;
  for (uint32_t mi = 0; mi < sizeof ms / sizeof ms[0]; ++mi)
    for (uint32_t ki = 0; ki < sizeof ks / sizeof ks[0]; ++ki) {
      const uint32_t m = ms[mi], k = ks[ki], pad = (m + 63u) & ~63u;
      const size_t n = (size_t)m * k, bytes = n * sizeof(float);
      float *orig = test_alloc(bytes), *resid = test_alloc(bytes),
            *x = test_alloc(bytes), *r = test_alloc(bytes),
            *sum = test_alloc(bytes), *norm = test_alloc(bytes),
            *want_sum = test_alloc(bytes), *want_norm = test_alloc(bytes),
            *scratch = test_alloc(bytes),
            *gamma = test_alloc(k * sizeof(float));
      float *scale = test_alloc(pad * sizeof(float)),
            *want_scale = test_alloc(pad * sizeof(float));
      int32_t *zp = test_alloc(pad * sizeof(int32_t)),
              *want_zp = test_alloc(pad * sizeof(int32_t));
      uint8_t *ah = test_alloc((size_t)pad * k),
              *want_ah = test_alloc((size_t)pad * k);
      if (!orig || !resid || !x || !r || !sum || !norm || !want_sum ||
          !want_norm || !scratch || !gamma || !scale || !want_scale || !zp ||
          !want_zp || !ah || !want_ah)
        return 2;
      for (size_t i = 0; i < n; ++i) {
        seed = seed * 1664525u + 1013904223u;
        orig[i] = (int32_t)(seed >> 8u) * (1.f / 8388608.f) - 1.f;
        seed = seed * 1664525u + 1013904223u;
        resid[i] = (int32_t)(seed >> 8u) * (1.f / 8388608.f) - 1.f;
        /* Cancellation, zero, and scales far apart expose intermediate f32
         * rounding changes that smooth random inputs alone can miss. */
        if (i % 29u == 0u)
          resid[i] = -orig[i];
        if (i % 37u == 0u) {
          orig[i] = 32768.f;
          resid[i] = 0.001953125f;
        }
      }
      for (uint32_t i = 0; i < k; ++i)
        gamma[i] = 0.25f + (float)(i % 53u) * 0.03125f;
      for (uint32_t add = 0; add < 2u; ++add)
        for (uint32_t gm = 0; gm < 2u; ++gm) {
          const float *g = gm ? gamma : NULL;
          if (add) {
            if (hvx_add_rmsnorm_rows_f32(orig, resid, want_sum, want_norm, m, k,
                                         k, g, 1.e-5f, NULL))
              return 3;
          } else if (hvx_rmsnorm_rows_f32(orig, want_norm, m, k, k, g, 1.e-5f,
                                          NULL)) {
            return 3;
          }
          if (hvx_quant_params_pack_u8_ah(want_norm, m, pad, k, NULL, 0.f, NULL,
                                          want_scale, want_zp, want_ah, NULL))
            return 4;
          for (uint32_t alias = 0; alias < (add ? 7u : 2u); ++alias) {
            workers = checked % 2u ? 7u : 1u;
            memcpy(x, orig, bytes);
            memcpy(r, resid, bytes);
            memset(norm, 0xa7, bytes);
            memset(sum, 0xa7, bytes);
            memset(ah, 0xa7, (size_t)pad * k);
            float *s = sum, *y = norm;
            if (add) {
              if (alias == 1u || alias == 5u)
                s = x;
              if (alias == 2u || alias == 6u)
                s = r;
              if (alias == 3u || alias == 6u)
                y = x;
              if (alias == 4u || alias == 5u)
                y = r;
            } else if (alias) {
              y = x;
            }
            if (hvx_quant_norm_params_pack_u8_ah(
                  x, add ? r : NULL, add ? s : NULL, y, m, pad, k, g, 1.e-5f,
                  scratch, scale, zp, ah, NULL))
              return 5;
            const uint32_t mode = 100u * add + 10u * gm + alias;
            same("norm", y, want_norm, bytes, m, k, mode);
            if (add)
              same("sum", s, want_sum, bytes, m, k, mode);
            same("scale", scale, want_scale, pad * sizeof(float), m, k, mode);
            same("zero", zp, want_zp, pad * sizeof(int32_t), m, k, mode);
            same("AH", ah, want_ah, (size_t)pad * k, m, k, mode);
            ++checked;
          }
        }
      test_free(orig);
      test_free(resid);
      test_free(x);
      test_free(r);
      test_free(sum);
      test_free(norm);
      test_free(want_sum);
      test_free(want_norm);
      test_free(scratch);
      test_free(gamma);
      test_free(scale);
      test_free(want_scale);
      test_free(zp);
      test_free(want_zp);
      test_free(ah);
      test_free(want_ah);
    }
  printf("ALIGNED NORM PACK PASS: %u cases, mode=%u, exact "
         "sum/norm/scale/zero/AH\n",
         checked, misaligned);
  return 0;
}
