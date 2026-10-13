// SPDX-License-Identifier: Apache-2.0
/**
 * @file kv_i8_tile_host_check.c
 * @date 13 October 2026
 * @brief Compare full-tile INT8 append against the actual scalar WH writer.
 * Host vectors are scalar stand-ins; device parity remains required.
 */
#include "hvx_worker_pool.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Expose the actual private worker without adding a production test API. */
#include "../../../nntrainer/tensor/htp_backend/hmx/hexkl_kv_q.c"

static int same(const hexkl_kv_q *a, const hexkl_kv_q *b) {
  const size_t rows = (size_t)a->max_rows * a->n_head_kv;
  const size_t values = rows * a->head_dim;
  return !memcmp(a->kt4, b->kt4, values) && !memcmp(a->v4, b->v4, values) &&
         !memcmp(a->s_k, b->s_k, rows * sizeof(float)) &&
         !memcmp(a->s_v, b->s_v, (rows + HEXKL_KV_Q_SV_PAD) * sizeof(float)) &&
         !memcmp(a->colsum_k, b->colsum_k, rows * sizeof(int32_t)) &&
         !memcmp(a->kt, b->kt, values) && !memcmp(a->v, b->v, values);
}

static void populate(int8_t *k, int8_t *v, float *scales, size_t rows,
                     uint32_t hd, uint32_t salt) {
  for (size_t i = 0; i < rows * hd; ++i) {
    k[i] = (int8_t)((i * 53u + i / hd + salt) % 256u - 128);
    v[i] = (int8_t)((i * 97u + 7u + salt) % 256u - 128);
  }
  for (size_t i = 0; i < rows * 2u; ++i)
    scales[i] = (float)(1u + (i + salt) % 17u) * 0.000125f;
}

static int run(hvx_worker_pool *pool, uint32_t hd, uint32_t heads,
               uint32_t row0, uint32_t rows, int alternate) {
  const uint32_t cap = (row0 + rows + 63u) & ~31u;
  hexkl_kv_q_table table = {0};
  uint32_t baseline, candidate;
  if (hexkl_kv_q_register(&table, HEXKL_KV_Q8, cap, heads, hd, &baseline) ||
      hexkl_kv_q_register(&table, HEXKL_KV_Q8, cap, heads, hd, &candidate))
    return 1;
  const size_t src_rows = (size_t)cap * heads;
  int8_t *k = malloc(src_rows * hd), *v = malloc(src_rows * hd);
  float *scales = malloc(src_rows * 2u * sizeof(float));
  if (!k || !v || !scales)
    return 1;

  /* A different but still bijective WH permutation must retain scatter. */
  for (uint32_t r = 0; r < 32u; ++r)
    for (uint32_t d = 0; d < 32u; ++d) {
      const uint16_t pos = (uint16_t)((r / 4u) * 128u + 4u * d + r % 4u);
      g_wh_i8_pos[r * 32u + d] = alternate ? 1023u - pos : pos;
    }
  int bad = wh_i8_segments_match(g_wh_i8_pos) != !alternate;
  const int direct =
    wh_i8_segments_match(g_wh_i8_pos) ? WH_I8_SEGMENTS : WH_I8_SCATTER;

  /* Start with nonzero cache contents so partial overwrites cannot hide
     accidental stores to a neighbour or the untouched end of a tile. */
  populate(k, v, scales, src_rows, hd, 11u);
  append_i8_ctx seed = {
    &table.slots[baseline], 0u, cap, 0u, cap / 32u, k, v, scales,
    WH_I8_SCATTER};
  append_i8_worker(1u, 0u, &seed);
  seed.kv = &table.slots[candidate];
  append_i8_worker(1u, 0u, &seed);
  populate(k, v, scales, src_rows, hd, 137u);
  const uint32_t first = row0 / 32u;
  const uint32_t tiles = rows ? (row0 + rows + 31u) / 32u - first : 0u;
  append_i8_ctx reference = {
    &table.slots[baseline], row0, rows, first, tiles, k, v, scales,
    WH_I8_SCATTER};
  append_i8_ctx tested = reference;
  tested.kv = &table.slots[candidate];
  tested.direct = direct;
  if (rows) {
    append_i8_worker(1u, 0u, &reference);
    hvx_worker_pool_run(pool, append_i8_worker, &tested, heads * tiles);
  }
  bad |= !same(reference.kv, tested.kv);

  if (rows) {
    const float invalid[] = {0.0f, -1.0f, INFINITY, NAN};
    const size_t last = (size_t)rows * heads * 2u - 1u;
    const float saved = scales[last];
    k[0] ^= 0x55;
    v[0] ^= 0x33;
    for (size_t i = 0; i < sizeof(invalid) / sizeof(invalid[0]); ++i) {
      scales[last] = invalid[i];
      bad |= hexkl_kv_q_append_i8_pool(&table, candidate, row0, rows, k, v,
                                       scales, NULL, pool) != (int)AEE_EBADPARM;
      bad |= !same(reference.kv, tested.kv);
    }
    scales[last] = saved;
  }
  if (bad)
    fprintf(stderr, "FAIL hd=%u heads=%u row0=%u rows=%u alternate=%d\n", hd,
            heads, row0, rows, alternate);
  free(k);
  free(v);
  free(scales);
  hexkl_kv_q_release(&table, baseline);
  hexkl_kv_q_release(&table, candidate);
  return bad != 0;
}

int main(void) {
  const uint32_t dims[] = {32, 64, 96, 128, 256, 512};
  const uint32_t offsets[] = {0, 1, 31, 33};
  const uint32_t rows[] = {0, 1, 31, 32, 33, 64, 65, 1024};
  const uint32_t heads[] = {1, 8};
  unsigned cases = 0;
  for (uint32_t workers = 3; workers <= 7; workers += 4) {
    hvx_worker_pool *pool = hvx_worker_pool_create(workers);
    if (!pool)
      return 1;
    for (unsigned alt = 0; alt < 2; ++alt)
      for (unsigned d = 0; d < sizeof(dims) / sizeof(dims[0]); ++d)
        for (unsigned h = 0; h < sizeof(heads) / sizeof(heads[0]); ++h)
          for (unsigned o = 0; o < sizeof(offsets) / sizeof(offsets[0]); ++o)
            for (unsigned r = 0; r < sizeof(rows) / sizeof(rows[0]); ++r) {
              if (run(pool, dims[d], heads[h], offsets[o], rows[r], alt))
                return 1;
              ++cases;
            }
    hvx_worker_pool_destroy(pool);
  }
  printf("KV I8 TILE: ALL %u CASES PASS (masters, metadata, WH, partial "
         "overwrite, alternate layout, invalid metadata)\n",
         cases);
  return 0;
}
