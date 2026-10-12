// SPDX-License-Identifier: Apache-2.0
/**
 * @file kv_i8_pool_host_check.c
 * @brief Exercise the actual private INT8 append worker with the real pool
 *        and pthread QuRT stand-ins. HMX tile writes require device tests.
 * @author NNTrainer contributors
 * @bug See the repository issue tracker for known issues.
 */
#include "hvx_worker_pool.h"
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/** Including the implementation exposes the same private worker/partition
   used on DSP without adding a production test hook. Host public dispatch
   remains serial; the pool below exercises its actual worker concurrently. */
#include "../../../nntrainer/tensor/htp_backend/hmx/hexkl_kv_q.c"

static _Atomic uint32_t lanes;
static void counted_worker(uint32_t n, uint32_t i, void *ctx) {
  atomic_fetch_or(&lanes, 1u << i);
  append_i8_worker(n, i, ctx);
}

static int same(const hexkl_kv_q *a, const hexkl_kv_q *b) {
  const size_t rows = (size_t)a->max_rows * a->n_head_kv;
  const size_t values = rows * a->head_dim;
  return !memcmp(a->kt4, b->kt4, values) && !memcmp(a->v4, b->v4, values) &&
         !memcmp(a->s_k, b->s_k, rows * sizeof(float)) &&
         !memcmp(a->s_v, b->s_v, (rows + HEXKL_KV_Q_SV_PAD) * sizeof(float)) &&
         !memcmp(a->colsum_k, b->colsum_k, rows * sizeof(int32_t)) &&
         !memcmp(a->kt, b->kt, values) && !memcmp(a->v, b->v, values);
}

static int run(hvx_worker_pool *pool, uint32_t workers, uint32_t hd,
               uint32_t row0, uint32_t n_rows) {
  const uint32_t heads = 8, cap = 1088;
  hexkl_kv_q_table table = {0};
  uint32_t hs, hp, hw;
  if (hexkl_kv_q_register(&table, HEXKL_KV_Q8, cap, heads, hd, &hs) ||
      hexkl_kv_q_register(&table, HEXKL_KV_Q8, cap, heads, hd, &hp) ||
      hexkl_kv_q_register(&table, HEXKL_KV_Q8, cap, heads, hd, &hw))
    return 1;
  const size_t src_rows = (size_t)(n_rows ? n_rows : 1u) * heads;
  int8_t *k = malloc(src_rows * hd), *v = malloc(src_rows * hd);
  float *scales = malloc(src_rows * 2 * sizeof(float));
  if (!k || !v || !scales)
    return 1;
  for (size_t i = 0; i < src_rows * hd; ++i) {
    k[i] = (int8_t)((i * 53u + i / hd) % 256u - 128);
    v[i] = (int8_t)((i * 97u + 7u) % 256u - 128);
  }
  for (size_t i = 0; i < src_rows * 2; ++i)
    scales[i] = (float)(1u + i % 17u) * 0.000125f;
  int bad = hexkl_kv_q_append_i8(&table, hs, row0, n_rows, k, v, scales, NULL);
  bad |= hexkl_kv_q_append_i8_pool(&table, hp, row0, n_rows, k, v, scales, NULL,
                                   pool);
  const uint32_t c_lo = row0 / 32u;
  const uint32_t n_col = n_rows ? (row0 + n_rows + 31u) / 32u - c_lo : 0u;
  append_i8_ctx ctx = {&table.slots[hw], row0, n_rows, c_lo, n_col, k, v,
                       scales,           0};
  atomic_store(&lanes, 0u);
  if (n_rows)
    hvx_worker_pool_run(pool, counted_worker, &ctx, heads * n_col);
  const uint32_t want_lanes = n_rows ? (1u << (workers + 1u)) - 1u : 0u;
  bad |= atomic_load(&lanes) != want_lanes;
  bad |= !same(&table.slots[hs], &table.slots[hp]);
  bad |= !same(&table.slots[hs], &table.slots[hw]);
  /* An independent check of every written/untouched logical master value. */
  const hexkl_kv_q *kv = &table.slots[hw];
  for (uint32_t row = 0; row < cap; ++row) {
    for (uint32_t h = 0; h < heads; ++h) {
      const int active = row >= row0 && row - row0 < n_rows;
      const size_t src = active ? ((size_t)(row - row0) * heads + h) : 0;
      int32_t sum = 0;
      for (uint32_t d = 0; d < hd; ++d) {
        const int32_t kr = active ? k[src * hd + d] : 0;
        const int32_t vr = active ? v[src * hd + d] : 0;
        sum += kr;
        bad |= kv->kt4[hexkl_kv_q_kt4_index(kv, h, row, d)] != kr + 128;
        bad |= kv->v4[hexkl_kv_q_v4_index(kv, h, row, d)] != vr + 128;
      }
      bad |= kv->colsum_k[hexkl_kv_q_sk_index(kv, h, row)] != sum;
      bad |= kv->s_k[hexkl_kv_q_sk_index(kv, h, row)] !=
             (active ? scales[2u * src] : 1.0f);
      bad |= kv->s_v[hexkl_kv_q_sv_index(kv, h, row)] !=
             (active ? scales[2u * src + 1u] : 1.0f);
    }
  }
  if (n_rows) {
    const float invalid[] = {0.0f, -1.0f, INFINITY, NAN};
    const size_t last = (size_t)n_rows * heads * 2u - 1u;
    const float saved = scales[last];
    /* Corrupting the final scale must reject before even the first write. */
    for (size_t j = 0; j < sizeof invalid / sizeof invalid[0]; ++j) {
      scales[last] = invalid[j];
      k[0] ^= 0x55;
      v[0] ^= 0x33;
      bad |= hexkl_kv_q_append_i8_pool(&table, hw, row0, n_rows, k, v, scales,
                                       NULL, pool) != (int)AEE_EBADPARM;
      bad |= !same(&table.slots[hs], &table.slots[hw]);
      bad |= hexkl_kv_q_append_i8(&table, hp, row0, n_rows, k, v, scales,
                                  NULL) != (int)AEE_EBADPARM;
      bad |= !same(&table.slots[hs], &table.slots[hp]);
      k[0] ^= 0x55;
      v[0] ^= 0x33;
    }
    scales[last] = saved;
  }
  printf("KV I8 POOL hd=%u row0=%u M=%u lanes=%u: %s\n", hd, row0, n_rows,
         workers + 1u, bad ? "FAILED" : "OK");
  free(k);
  free(v);
  free(scales);
  hexkl_kv_q_release(&table, hs);
  hexkl_kv_q_release(&table, hp);
  hexkl_kv_q_release(&table, hw);
  return bad != 0;
}

int main(void) {
  const uint32_t rows[] = {0, 1, 63, 64, 65, 1024};
  const uint32_t offsets[] = {0, 1, 31, 33};
  for (uint32_t workers = 3; workers <= 7; workers += 4) {
    hvx_worker_pool *pool = hvx_worker_pool_create(workers);
    if (!pool)
      return 1;
    for (uint32_t hd = 64; hd <= 128; hd *= 2)
      for (size_t o = 0; o < sizeof offsets / sizeof offsets[0]; ++o)
        for (size_t m = 0; m < sizeof rows / sizeof rows[0]; ++m)
          if (run(pool, workers, hd, offsets[o], rows[m]))
            return 1;
    hvx_worker_pool_destroy(pool);
  }
  puts("KV I8 POOL: ALL 96 CASES PASS (host masters, actual worker pool)");
  return 0;
}
