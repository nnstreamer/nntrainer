// SPDX-License-Identifier: Apache-2.0
/**
 * @file dequant_planes_host_check.c
 * @brief Compare planar HMX accumulator epilogues against int32 tiles using
 *        the real HVX source on lane emulation, including partial row groups.
 */
#include "hvx_dequant_i32.h"
#include "hvx_scale_add_f32.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

void hvx_worker_pool_run(hvx_worker_pool *pool, hvx_worker_pool_func func,
                         void *ctx, uint32_t n_units) {
  const uint32_t n = n_units < 3u ? n_units : 3u;
  (void)pool;
  for (uint32_t i = 0; i < n; ++i)
    func(n, i, ctx);
}

/* Scalar byte extraction is independent of the HVX shuffle implementation.
   Padded rows contain distinct data to catch over-read or wrong plane pitch. */
static void make_planes(const int32_t *src, uint8_t *dst) {
  for (uint32_t i = 0; i < 2048; ++i) {
    uint32_t bits;
    memcpy(&bits, src + i, sizeof bits);
    for (uint32_t p = 0; p < 4; ++p)
      dst[p * 2048u + i] = (uint8_t)(bits >> (8u * p));
  }
}

/* The reference materializes dequantized values, then applies independent
   scalar multiply/add in routing order. Repeated destinations deliberately
   expose preloading an output row before an earlier contribution is stored. */
static int check_weighted_scatter(void) {
  enum { ROWS = 71, OUT_STRIDE = 113, COL0 = 13 };
  static int32_t tile[64 * 32] __attribute__((aligned(128)));
  static int32_t padded[64 * 64] __attribute__((aligned(128)));
  static uint8_t planes[8192] __attribute__((aligned(128)));
  static float ref[ROWS * OUT_STRIDE], got[ROWS * OUT_STRIDE];
  static float materialized[64 * 32];
  static float staged[64 * OUT_STRIDE];
  float as[64], ws[32], bias[32], weights[64];
  int32_t az[64], cs[32];
  uint32_t rows[64], seed = 0x83a149b5u;
  const uint32_t counts[] = {0, 1, 3, 4, 5, 31, 32, 33, 63, 64};
  const float weight_values[] = {0.0f, -0.0f,  0.125f,     -0.75f,
                                 1.3f, -2.25f, 0.33333334f};
  for (uint32_t r = 0; r < 64; ++r) {
    as[r] = (1u + r % 7u) * 0.00001f;
    az[r] = (int32_t)(r % 13u) - 6;
    for (uint32_t c = 0; c < 64; ++c)
      padded[r * 64 + c] = 0x456789ab;
    for (uint32_t c = 0; c < 32; ++c) {
      seed = seed * 1664525u + 1013904223u;
      memcpy(tile + r * 32 + c, &seed, sizeof seed);
      padded[r * 64 + c] = tile[r * 32 + c];
    }
  }
  make_planes(tile, planes);
  for (uint32_t c = 0; c < 32; ++c) {
    ws[c] = (1u + c % 5u) * 0.0001f;
    cs[c] = (int32_t)(c * 117u) - 2000;
    bias[c] = ((int32_t)(c % 9u) - 4) * 0.125f;
  }
  unsigned checked = 0;
  for (uint32_t ci = 0; ci < sizeof counts / sizeof counts[0]; ++ci) {
    for (uint32_t layout = 0; layout < 3; ++layout) {
      const int32_t *input = layout == 0   ? (const int32_t *)planes
                             : layout == 1 ? tile
                                           : padded;
      const uint32_t stride = layout == 0 ? 0 : layout == 1 ? 32 : 64;
      for (uint32_t route = 0; route < 3; ++route) {
        for (uint32_t i = 0; i < ROWS * OUT_STRIDE; ++i)
          ref[i] = got[i] = ((int32_t)(i % 101u) - 50) * 0.03125f;
        for (uint32_t pass = 0; pass < 2; ++pass) {
          for (uint32_t r = 0; r < 64; ++r) {
            rows[r] = route == 0   ? r
                      : route == 1 ? (r * 37u + 11u) % ROWS
                                   : r % 3u;
            weights[r] = weight_values[(r + 3u * pass) % 7u];
          }
          hvx_dequant_acc_tile_to_f32(tile, 32, counts[ci], as, az, cs, ws,
                                      bias, materialized, 32, 0);
          for (uint32_t r = 0; r < counts[ci]; ++r)
            for (uint32_t c = 0; c < 32; ++c) {
              const float contribution = materialized[r * 32 + c] * weights[r];
              ref[rows[r] * OUT_STRIDE + COL0 + c] += contribution;
            }
          hvx_dequant_acc_tile_to_f32(input, stride, counts[ci], as, az, cs, ws,
                                      bias, staged + COL0, OUT_STRIDE, 0);
          hvx_scale_add_scatter_tile_f32(got + COL0, staged + COL0, rows,
                                         weights, counts[ci], OUT_STRIDE);
          if (memcmp(ref, got, sizeof ref)) {
            printf(
              "DEQUANT SCATTER FAILED rows=%u layout=%u route=%u pass=%u\n",
              counts[ci], layout, route, pass);
            return 1;
          }
        }
        ++checked;
      }
    }
  }
  printf("DEQUANT SCATTER OK: %u cases, two contributions each\n", checked);
  return 0;
}

int main(void) {
  enum { PAIRS = 4, INTER = 192, STRIDE = INTER + 32 };
  static int32_t tiles[2 * PAIRS][2048] __attribute__((aligned(128)));
  static uint8_t planes[2 * PAIRS][8192] __attribute__((aligned(128)));
  static float ref[64 * STRIDE], got[64 * STRIDE];
  float as[64], ws[2 * INTER], bias[2 * INTER];
  int32_t az[64], cs[2 * INTER];
  uint32_t seed = 0xabcdef01u;
  const uint32_t edge[] = {0u,          1u,          0xffffffffu, 0x80000000u,
                           0x7fffffffu, 0x01020304u, 0x80ff7f00u};
  for (uint32_t t = 0; t < 2 * PAIRS; ++t) {
    for (uint32_t i = 0; i < 2048; ++i) {
      seed = seed * 1664525u + 1013904223u;
      uint32_t bits = i < sizeof edge / sizeof edge[0] ? edge[i] : seed;
      memcpy(&tiles[t][i], &bits, sizeof bits);
    }
    make_planes(tiles[t], planes[t]);
  }
  for (uint32_t r = 0; r < 64; ++r) {
    as[r] = 0.00001f * (1u + r % 3u);
    az[r] = (int32_t)(r % 11u) - 5;
  }
  for (uint32_t c = 0; c < 2 * INTER; ++c) {
    cs[c] = (int32_t)(c * 113u) - 20000;
    ws[c] = 0.0001f * (1u + c % 7u);
    bias[c] = ((int32_t)(c % 9u) - 4) * 0.125f;
  }
  const uint32_t counts[] = {0, 1, 3, 4, 5, 31, 32, 33, 63, 64};
  unsigned checked = 0;
  for (uint32_t m = 0; m < sizeof counts / sizeof counts[0]; ++m) {
    for (uint32_t mode = 0; mode < 4; ++mode) {
      for (uint32_t i = 0; i < 64 * STRIDE; ++i)
        ref[i] = got[i] = 123.25f;
      if (mode < 2) {
        hvx_dequant_acc_tile_to_f32(tiles[0], 32, counts[m], as, az, cs, ws,
                                    bias, ref + 32, STRIDE, mode);
        hvx_dequant_acc_tile_to_f32((const int32_t *)planes[0], 0, counts[m],
                                    as, az, cs, ws, bias, got + 32, STRIDE,
                                    mode);
      } else {
        hvx_dequant_swiglu_acc_tiles_to_f32(
          (const uint8_t *)tiles, 8192, PAIRS, 1, 32, counts[m], as, az, cs, ws,
          bias, INTER, mode - 2, ref, STRIDE, NULL);
        hvx_dequant_swiglu_acc_tiles_to_f32(
          (const uint8_t *)planes, 8192, PAIRS, 1, 0, counts[m], as, az, cs, ws,
          bias, INTER, mode - 2, got, STRIDE, NULL);
      }
      if (memcmp(ref, got, sizeof ref)) {
        printf("DEQUANT PLANES FAILED rows=%u mode=%u\n", counts[m], mode);
        return 1;
      }
      ++checked;
    }
  }
  printf("DEQUANT PLANES OK: %u cases (store, accumulate, SwiGLU, GeGLU)\n",
         checked);
  return check_weighted_scatter();
}
