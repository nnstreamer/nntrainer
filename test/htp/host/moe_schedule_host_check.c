// SPDX-License-Identifier: Apache-2.0
/**
 * @file moe_schedule_host_check.c
 * @brief Production expansion partition and threaded DN/requant scheduling.
 * @author NNTrainer contributors
 * @bug See the repository issue tracker for known issues.
 */

#include "hvx_dequant_i32.h"
#include "hvx_quant_u8.h"
#include "hvx_scale_add_f32.h"
#include <assert.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define HEXKL_HMX_INT8_BLOCK_N_ROW 64u
#define HEXKL_ACC_TILE_COLS 32u
#define ROUND_UP_U32(x, a) (((x) + (a)-1u) & ~((a)-1u))
#define HEXKL_PROBE_T0(t) ((void)(t))
#include "hvx_expand_i2i4.h"
#include "hvx_gemv_u8i2_lut.h"
void hvx_worker_pool_run(hvx_worker_pool *pool, hvx_worker_pool_func f,
                         void *ctx, uint32_t n) {
  (void)pool;
  (void)n;
  f(1, 0, ctx);
}
#include "moe_dn_under_test.inc"
static inline void moe_worker_probe_add(uint64_t t0) { (void)t0; }
struct launch {
  moe_dn_ctx *c;
  unsigned n, i;
};
static void *worker(void *v) {
  struct launch *a = v;
  moe_dn_worker(a->n, a->i, a->c);
  return NULL;
}
static unsigned seed = 548712u;
static unsigned next(void) { return seed = 1664525u * seed + 1013904223u; }
static int check_dn(void) {
  enum { N = 2048, I = 1536, M = 64, BR = 5 };
  uint8_t *tiles = aligned_alloc(128, 64 * 8192);
  float *as = malloc(M * 4), *ws = malloc(N * 4), *bias = malloc(N * 4),
        *weights = malloc(M * 4);
  int32_t *az = malloc(M * 4), *cs = malloc(N * 4);
  uint32_t rows[M], rows_b[BR];
  float weights_b[BR];
  float *tail = malloc(BR * N * 4), *basegate = malloc(M * I * 4),
        *baseout = aligned_alloc(128, M * N * 4);
  for (unsigned t = 0; t < 64; ++t)
    for (unsigned j = 0; j < 2048; ++j) {
      unsigned z = next();
      for (unsigned p = 0; p < 4; ++p)
        tiles[t * 8192 + p * 2048 + j] = (z >> (8 * p));
    }
  for (unsigned m = 0; m < M; ++m) {
    as[m] = (1 + m % 3) * 1e-5f;
    az[m] = m % 19;
    weights[m] = (m % 9 - 4.0f) * .125f;
    rows[m] = (m * 7) % 13;
  }
  for (unsigned j = 0; j < N; ++j) {
    ws[j] = (1 + j % 7) * 1e-4f;
    cs[j] = (int)(j % 419) - 200;
    bias[j] = (j % 3 - 1.0f) * .001f;
  }
  for (unsigned j = 0; j < BR; ++j) {
    rows_b[j] = j % 3;
    weights_b[j] = (j - 2.0f) * .3f;
  }
  for (unsigned j = 0; j < BR * N; ++j)
    tail[j] = ((int)(next() % 251) - 125) * .01f;
  for (unsigned j = 0; j < M * I; ++j)
    basegate[j] = ((int)(next() % 251) - 125) * .03f;
  for (unsigned j = 0; j < M * N; ++j)
    baseout[j] = ((int)(next() % 251) - 125) * .1f;
  const unsigned thread_counts[] = {1, 3, 4, 5, 6, 7, 8},
                 tile_counts[] = {0, 1, 7, 32, 64}, counts[] = {1, 13, 63, 64};
  unsigned cases = 0;
  for (unsigned tc = 0; tc < 7; ++tc)
    for (unsigned nt = 0; nt < 5; ++nt)
      for (unsigned rc = 0; rc < 4; ++rc)
        for (unsigned rq = 0; rq < 2; ++rq) {
          moe_dn_ctx a = {0}, c;
          a.tiles_base = tiles;
          a.tile_stride = 8192;
          a.nb = tile_counts[nt];
          a.row_stride = 0;
          a.m_count = counts[rc];
          a.N_out = N;
          a.act_scale = as;
          a.act_zp = az;
          a.colsum_w = cs;
          a.w_scale = ws;
          a.bias = bias;
          a.rows = rows;
          a.weights = weights;
          a.res_b = tail;
          a.rows_b = rows_b;
          a.weights_b = weights_b;
          a.n_rows_b = BR;
          a.inter = I;
          a.rq_m_blk = counts[rc];
          c = a;
          a.out = aligned_alloc(128, M * N * 4);
          c.out = aligned_alloc(128, M * N * 4);
          a.res = aligned_alloc(128, M * N * 4);
          c.res = aligned_alloc(128, M * N * 4);
          a.gate = rq ? malloc(M * I * 4) : NULL;
          c.gate = rq ? malloc(M * I * 4) : NULL;
          a.mid = malloc(M * I);
          c.mid = malloc(M * I);
          a.rq_scale = malloc(M * 4);
          c.rq_scale = malloc(M * 4);
          a.rq_zp = malloc(M * 4);
          c.rq_zp = malloc(M * 4);
          memcpy(a.out, baseout, M * N * 4);
          memcpy(c.out, baseout, M * N * 4);
          memset(a.res, 0xA5, M * N * 4);
          memset(c.res, 0xA5, M * N * 4);
          if (rq) {
            memcpy(a.gate, basegate, M * I * 4);
            memcpy(c.gate, basegate, M * I * 4);
          }
          memset(a.mid, 0xA5, M * I);
          memset(c.mid, 0xA5, M * I);
          memset(a.rq_scale, 0xA5, M * 4);
          memset(c.rq_scale, 0xA5, M * 4);
          memset(a.rq_zp, 0xA5, M * 4);
          memset(c.rq_zp, 0xA5, M * 4);
          moe_dn_worker(1, 0, &a);
          pthread_t th[8];
          struct launch launches[8];
          for (unsigned j = 0; j < thread_counts[tc]; ++j) {
            launches[j] = (struct launch){&c, thread_counts[tc], j};
            assert(!pthread_create(th + j, NULL, worker, launches + j));
          }
          for (unsigned j = 0; j < thread_counts[tc]; ++j)
            assert(!pthread_join(th[j], NULL));
          if (memcmp(a.out, c.out, M * N * 4) ||
              memcmp(a.res, c.res, M * N * 4) || memcmp(a.mid, c.mid, M * I) ||
              memcmp(a.rq_scale, c.rq_scale, M * 4) ||
              memcmp(a.rq_zp, c.rq_zp, M * 4) ||
              (rq && memcmp(a.gate, c.gate, M * I * 4))) {
            printf("FAIL threads=%u tiles=%u rows=%u rq=%u\n",
                   thread_counts[tc], tile_counts[nt], counts[rc], rq);
            return 1;
          }
          free(a.out);
          free(c.out);
          free(a.res);
          free(c.res);
          free(a.gate);
          free(c.gate);
          free(a.mid);
          free(c.mid);
          free(a.rq_scale);
          free(c.rq_scale);
          free(a.rq_zp);
          free(c.rq_zp);
          cases++;
        }
  free(tiles);
  free(as);
  free(ws);
  free(bias);
  free(weights);
  free(az);
  free(cs);
  free(tail);
  free(basegate);
  free(baseout);
  printf("DN_WORKSHARE_HOST_PASS cases=%u repeated_output_rows=13 "
         "tail_overlap=3 threads=1,3,4,5,6,7,8\n",
         cases);
  return 0;
}

#include "moe_expand_under_test.inc"
static int check_partition(void) {
  const unsigned nr[] = {1, 3, 7, 48, 64, 129}, nu[] = {1, 3, 7, 16, 32, 257},
                 nb[] = {1, 2}, sz[] = {128, 256, 384, 1024, 1536, 2048};
  unsigned cases = 0;
  for (unsigned r = 0; r < 6; ++r)
    for (unsigned u = 0; u < 6; ++u)
      for (unsigned z = 0; z < 2; ++z)
        for (unsigned q = 0; q < 6; ++q)
          for (unsigned prep = 0; prep < 2; ++prep) {
            if (prep && sz[q] % 1024u)
              continue;
            const unsigned rows = nr[r], units = nu[u], bases = nb[z],
                           bytes = sz[q], stride = 4 * bytes + 256,
                           total = rows * stride;
            uint8_t *buf = aligned_alloc(128, total), *ref = malloc(total);
            int8_t pal[] = {-7, -2, 1, 5};
            moe_expand_ctx c = {0};
            for (unsigned j = 0; j < total; ++j)
              buf[j] = (j * 33u + j / 256) % 256;
            memcpy(ref, buf, total);
            c.base[0] = buf;
            c.base[1] = buf + 2 * bytes + 128;
            c.nbase = bases;
            c.nrows = rows;
            c.stride = stride;
            c.src_bytes = bytes;
            c.prepared = prep;
            hvx_expand_i2i4_table(pal, c.lut);
            for (unsigned b = 0; b < bases; ++b)
              for (unsigned k = 0; k < rows; ++k) {
                uint8_t *p = ref + b * (2 * bytes + 128) + k * stride;
                if (prep)
                  hvx_i2_lut_unrepack_row(p + bytes, bytes);
                hvx_expand_i2i4_scalar(p + bytes, bytes, c.lut, p);
              }
            for (unsigned id = units; id--;)
              moe_expand_unit(units, id, &c);
            if (memcmp(buf, ref, total)) {
              printf("PARTITION_FAIL rows=%u units=%u bases=%u bytes=%u "
                     "prepared=%u\n",
                     rows, units, bases, bytes, prep);
              return 1;
            }
            free(buf);
            free(ref);
            cases++;
          }
  printf("EXPAND_PARTITION_HOST_PASS cases=%u "
         "empty_units_reverse_order_prepared_sideeffects=PASS\n",
         cases);
  return 0;
}

int main(void) { return check_partition() || check_dn(); }
