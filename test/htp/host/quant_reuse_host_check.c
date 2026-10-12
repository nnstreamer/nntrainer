// SPDX-License-Identifier: Apache-2.0
/* Run the real HVX quantizer and transpose against mapped AH packing. */
#include "hvx_quant_u8.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
void hvx_worker_pool_run(hvx_worker_pool *pool, hvx_worker_pool_func f, void *c,
                         uint32_t units) {
  (void)pool;
  uint32_t n = units < 5u ? units : 5u;
  for (uint32_t i = 0; i < n; ++i)
    f(n, i, c);
}
int main(void) {
  const uint32_t ms[] = {1, 3, 5, 31, 33, 63, 65, 129};
  const uint32_t ks[] = {128, 256, 384, 2048};
  uint32_t seed = 19, checked = 0;
  for (uint32_t mi = 0; mi < 8; ++mi)
    for (uint32_t ki = 0; ki < 4; ++ki) {
      const uint32_t m = ms[mi], k = ks[ki], slots = 128,
                     pad = (m + 63u) / 64u * 64u;
      float *x = malloc((size_t)m * k * sizeof(float)),
            *scale = malloc(pad * sizeof(float)),
            *rs = malloc(pad * sizeof(float));
      int32_t *zp = malloc(pad * sizeof(int32_t)),
              *rz = malloc(pad * sizeof(int32_t));
      uint8_t *q = malloc((size_t)m * k), *ref = malloc((size_t)slots * k),
              *got = malloc((size_t)slots * k);
      float ss[128];
      int32_t zz[128];
      uint32_t map[128];
      for (uint32_t r = 0; r < m; ++r)
        for (uint32_t c = 0; c < k; ++c) {
          seed = seed * 1664525u + 1013904223u;
          x[(size_t)r * k + c] =
            r == 0   ? 0.0f
            : r == 1 ? 3.125f
            : r == 2 ? -2.75f
                     : ((int32_t)(seed % 200001u) - 100000) * 0.00731f;
        }
      hvx_quant_rows_u8_params(x, m, pad, k, rs, rz, NULL);
      memset(scale, 0x99, pad * sizeof(float));
      memset(zp, 0x99, pad * sizeof(int32_t));
      hvx_quant_rows_u8_params_rowmajor(x, m, pad, k, scale, zp, q, NULL);
      if (memcmp(rs, scale, pad * sizeof(float)) ||
          memcmp(rz, zp, pad * sizeof(int32_t))) {
        printf("PARAMS FAIL m=%u k=%u\n", m, k);
        return 1;
      }
      for (uint32_t r = 0; r < slots; ++r) {
        map[r] = (r * 37u + 11u) % m;
        ss[r] = scale[map[r]];
        zz[r] = zp[map[r]];
      }
      if (hvx_quant_pack_u8_ah_mapped(x, map, slots, slots, k, ss, zz, ref,
                                      NULL))
        return 2;
      memset(got, 0x9b, (size_t)slots * k);
      for (uint32_t r = 0; r < slots; r += 16u)
        hvx_gather_rowmajor_u8_ah_rows(q, map, r, r + 16u, k, got);
      if (memcmp(ref, got, (size_t)slots * k)) {
        printf("QUANT REUSE FAIL m=%u k=%u\n", m, k);
        return 1;
      }
      for (uint32_t r = 0; r < m; ++r)
        for (uint32_t c = 0; c < k; ++c)
          q[(size_t)r * k + c] = (uint8_t)((r * 19u + c * 31u) ^ (c >> 5));
      for (uint32_t r = 0; r < slots; r += 16u)
        hvx_gather_rowmajor_u8_ah_rows(q, map, r, r + 16u, k, got);
      for (uint32_t r = 0; r < slots; ++r)
        for (uint32_t c = 0; c < k; ++c) {
          size_t off = (size_t)(r / 64u) * 64u * k + (c / 32u) * 2048u +
                       (r % 64u) * 32u + c % 32u;
          if (got[off] != q[(size_t)map[r] * k + c]) {
            printf("TRANSPOSE FAIL m=%u k=%u\n", m, k);
            return 1;
          }
        }
      free(x);
      free(scale);
      free(zp);
      free(rs);
      free(rz);
      free(q);
      free(ref);
      free(got);
      ++checked;
    }
  printf("Q8 ROW REUSE OK: %u shapes, exact AH bytes and independent transpose "
         "reference\n",
         checked);
  return 0;
}
