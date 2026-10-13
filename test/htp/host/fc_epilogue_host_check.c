// SPDX-License-Identifier: Apache-2.0
/**
 * @file fc_epilogue_host_check.c
 * @brief Compare fused FC row postprocessing with separate HVX passes.
 * @author NNTrainer contributors
 * @bug See the repository issue tracker for known issues.
 */
#include "hvx_dequant_i32.h"
#include "hvx_rmsnorm_rows_f32.h"
#include "hvx_rope_rows_f32.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>

enum { NT = 6, N = 256, LD = 288, HD = 64 };
static unsigned callbacks[64];
static float gamma[HD], cs[64 * 2 * HD];
static int do_norm, do_rope;
void hvx_worker_pool_run(hvx_worker_pool *pool, hvx_worker_pool_func f,
                         void *ctx, uint32_t units) {
  (void)pool;
  const uint32_t n = units < 3u ? units : 3u;
  for (uint32_t i = 0; i < n; ++i)
    f(n, i, ctx);
}
static void finish(void *ctx, float *row, uint32_t m) {
  (void)ctx;
  ++callbacks[m];
  if (do_norm)
    hvx_rmsnorm_row_chunks_f32(row, row, gamma, NT * 32, HD, 1e-6f);
  if (do_rope)
    hvx_rope_row_f32(row, NT * 32, HD, cs + (size_t)m * 2 * HD);
}
int main(void) {
  static int32_t tiles[NT][64 * 32] __attribute__((aligned(128)));
  static uint8_t planes[NT][8192] __attribute__((aligned(128)));
  static int32_t padded[NT][64 * 64] __attribute__((aligned(128)));
  static float ref[64 * LD], got[64 * LD], as[64], ws[N], bs[N];
  static int32_t az[64], sums[N];
  const uint32_t rows[] = {1, 3, 4, 5, 31, 32, 33, 63, 64};
  unsigned seed = 42, checked = 0;
  for (unsigned j = 0; j < NT; ++j)
    for (unsigned i = 0; i < 64 * 32; ++i) {
      seed = seed * 1664525u + 1013904223u;
      memcpy(&tiles[j][i], &seed, 4);
      padded[j][(i / 32) * 64 + i % 32] = tiles[j][i];
      for (unsigned b = 0; b < 4; ++b)
        planes[j][b * 2048 + i] = seed >> (8 * b);
    }
  for (unsigned i = 0; i < 64; ++i) {
    as[i] = (i % 7 + 1) * .00001f;
    az[i] = (int32_t)(i % 19) - 9;
    gamma[i] = .75f + .01f * i;
  }
  for (unsigned i = 0; i < N; ++i) {
    ws[i] = (i % 3 + 1) * .0001f;
    bs[i] = ((int)(i % 17) - 8) * .125f;
    sums[i] = (int32_t)i * 101 - 3000;
  }
  for (unsigned i = 0; i < 64 * 2 * HD; ++i)
    cs[i] = ((int)(i % 23) - 11) * .0625f;
  for (unsigned ci = 0; ci < sizeof rows / sizeof rows[0]; ++ci)
    for (unsigned layout = 0; layout < 3; ++layout)
      for (unsigned mode = 0; mode < 4; ++mode) {
        const uint32_t m = rows[ci];
        for (unsigned i = 0; i < 64 * LD; ++i)
          ref[i] = got[i] = -123.5f;
        memset(callbacks, 0, sizeof callbacks);
        do_norm = mode & 1;
        do_rope = mode & 2;
        hvx_dq_tiles_job t = {.tiles_base = (const uint8_t *)tiles,
                              .tile_stride = 8192,
                              .nt0 = 2,
                              .row_stride = 32,
                              .m_count = m,
                              .act_scale = as,
                              .act_zp = az,
                              .colsum_w = sums,
                              .w_scale = ws,
                              .bias = bs,
                              .dst_a = ref,
                              .split = N,
                              .dst_stride = LD,
                              .n_tiles = NT};
        hvx_dq_tiles_worker(1, 0, &t);
        if (do_norm)
          hvx_rmsnorm_rows_ld_f32(ref + 64, ref + 64, m, NT * 32, LD, HD, gamma,
                                  1e-6f, NULL);
        if (do_rope)
          hvx_rope_rows_ld_f32(ref + 64, m, NT * 32, LD, HD, cs, NULL);
        t.tiles_base = layout == 0   ? (const uint8_t *)planes
                       : layout == 1 ? (const uint8_t *)tiles
                                     : (const uint8_t *)padded;
        t.row_stride = layout == 0 ? 0 : layout == 1 ? 32 : 64;
        t.tile_stride = layout == 2 ? 16384 : 8192;
        t.dst_a = got;
        hvx_dq_rows_job jb = {.t = t, .row_done = finish};
        for (unsigned worker = 0; worker < 7; ++worker)
          hvx_dq_tiles_rows_worker(7, worker, &jb);
        if (memcmp(ref, got, sizeof ref)) {
          printf("FC EPILOGUE FAIL: rows=%u layout=%u mode=%u\n", m, layout,
                 mode);
          return 1;
        }
        for (unsigned i = 0; i < 64; ++i)
          if (callbacks[i] != (i < m))
            return 2;
        ++checked;
      }
  printf("FC EPILOGUE OK: %u bit-exact cases, planar/int32/padded, partial "
         "rows and guards\n",
         checked);
  return 0;
}
