// SPDX-License-Identifier: Apache-2.0
/**
 * @file nntr_hvx_lut_probe.c
 * @brief Isolated pair-LUT/WH GEMV comparison; no model dispatch changes.
 */
#include <AEEStdErr.h>
#include <remote.h>
#include <stdlib.h>
#include <string.h>

#include "hexkl_dma_ring.h"
#include "hexkl_probe.h"
#include "hvx_expand_i2i4.h"
#include "hvx_gemm_u8i4_wh.h"
#include "hvx_gemv_u8i2_lut.h"
#include "hvx_worker_pool.h"
#include "nntr_hvx.h"
#include "nntr_hvx_session.h"

typedef struct {
  uint32_t K, N, mode;
  const uint8_t *act, *weight, *expand_table;
  const int8_t *pal;
  int16_t *tables;
  int32_t *out;
  uint32_t threads;
  uint8_t *work;
} lut_probe_ctx;

static void repack_worker(uint32_t threads, uint32_t id, void *v) {
  const lut_probe_ctx *c = v;
  const uint32_t kt = c->K / 32u;
  if ((c->mode == 4u || c->mode == 6u || c->mode == 8u || c->mode == 10u ||
       c->mode == 11u))
    hvx_i2_lut_repack(c->work + c->N * 8u, c->N, c->N * 16u, c->work,
                      c->N * 16u, 1024u, kt * id / threads,
                      kt * (id + 1u) / threads);
  else
    hvx_i2_lut_repack(c->weight, c->N, c->N * 8u, c->work, 1024u, kt * 1024u,
                      kt * id / threads, kt * (id + 1u) / threads);
}

static void prepare_worker(uint32_t threads, uint32_t id, void *v) {
  lut_probe_ctx *c = v;
  const uint32_t kt = c->K / 32u;
  void (*prepare)(const uint8_t *, const int8_t *, uint32_t, uint32_t,
                  int16_t *) = c->mode >= 9u   ? hvx_i2_lut_prepare_dense
                               : c->mode >= 7u ? hvx_i2_lut_prepare_centered
                                               : hvx_i2_lut_prepare;
  prepare(c->act, c->pal, kt * id / threads, kt * (id + 1u) / threads,
          c->tables);
}

static void gemv_worker(uint32_t threads, uint32_t id, void *v) {
  lut_probe_ctx *c = v;
  if (id == 0u)
    c->threads = threads;
  const uint32_t paired = c->mode >= 5u && c->N % 256u == 0u;
  const uint32_t units =
    c->N / (paired ? 256u : (c->mode == 1u || c->mode >= 3u ? 128u : 32u));
  for (uint32_t u = units * id / threads; u < units * (id + 1u) / threads;
       ++u) {
    if (paired) {
      const uint32_t panel_stride =
        (c->mode == 6u || c->mode == 8u || c->mode == 10u || c->mode == 11u)
          ? 1024u
          : c->K * 32u;
      void (*gemv)(const uint8_t *, const uint8_t *, uint32_t, uint32_t,
                   const int16_t *, int32_t *, int32_t *) =
        c->mode >= 9u   ? hvx_gemv_u8i2_lut_dense_pair
        : c->mode >= 7u ? hvx_gemv_u8i2_lut_pair_centered
                        : hvx_gemv_u8i2_lut_pair;
      gemv(c->weight + u * panel_stride, c->weight + (u + units) * panel_stride,
           c->K / 32u,
           (c->mode == 6u || c->mode == 8u || c->mode == 10u || c->mode == 11u)
             ? c->N * 16u
             : 1024u,
           c->tables, c->out + u * 128u, c->out + (u + units) * 128u);
    } else if (c->mode >= 9u) {
      const uint32_t stride =
        (c->mode == 10u || c->mode == 11u) ? c->N * 16u : 1024u;
      const uint8_t *src =
        c->weight +
        u * ((c->mode == 10u || c->mode == 11u) ? 1024u : c->K * 32u);
      hvx_gemv_u8i2_lut_dense_pair(src, src, c->K / 32u, stride, c->tables,
                                   c->out + u * 128u, c->out + u * 128u);
    } else if (c->mode == 4u || c->mode == 6u || c->mode == 8u ||
               c->mode == 10u || c->mode == 11u)
      hvx_gemv_u8i2_lut_strided(c->weight + u * 1024u, c->K / 32u, c->N * 16u,
                                c->tables, c->out + u * 128u);
    else if (c->mode == 1u || c->mode == 3u || c->mode == 5u || c->mode == 7u)
      hvx_gemv_u8i2_lut_panel(c->weight, c->K, u, c->tables, c->out + u * 128u);
    else if (c->mode == 2u)
      hvx_gemv_u8i2_wh_fused_col(c->act, c->K / 32u, c->weight, c->N / 32u, u,
                                 c->expand_table, c->out + u * 32u);
    else
      hvx_gemm_u8i4_wh_col_nopf(c->act, 1u, c->K / 32u, c->weight, c->N / 32u,
                                u, 1u, c->out + u * 32u);
  }
}

int nntr_hvx_lut_gemv_probe(remote_handle64 handle, uint32 K, uint32 N,
                            uint32 mode, uint32 workers, uint32 passes,
                            const uint8 *act, int actLen, const int8 *pal,
                            int palLen, const uint8 *weight, int weightLen,
                            int32 *out, int outLen, uint32 *stats,
                            int statsLen) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  const uint32_t dma = mode & 256u;
  mode &= ~256u;
  if (!s || !s->vtcm_base || !K || !N || K % 32u || N % 128u || K > 8192u ||
      N >= 8192u ||
      ((mode == 4u || mode == 6u || mode == 8u || mode == 10u || mode == 11u) &&
       N >= 4096u) ||
      mode > 11u || (!workers || workers > 8u) || !passes || passes > 1000u ||
      !act || !pal || !weight || !out || !stats || actLen != (int)K ||
      palLen != 4 || outLen != (int)N || statsLen < 6 ||
      weightLen != (int)((uint64_t)K * N / 4u))
    return AEE_EBADPARM;
  for (uint32_t i = 0; i < 4u; ++i)
    if (pal[i] < -8 || pal[i] > 7)
      return AEE_EBADPARM;
  const size_t bytes = (size_t)K * N / 4u;
  const size_t need = 3u * bytes + (size_t)K * 32u + N * 4u + 128u;
  if (need > s->config_off)
    return AEE_ENOMEMORY;
  uint8_t *compact = s->vtcm_base;
  uint8_t *work = compact + bytes;
  /* The production M=1 AH rows live in cacheable heap scratch. Scalar
   * activation-word loads from VTCM make vrmpy artificially much slower. */
  uint8_t *ah = memalign(128u, (size_t)K * 64u);
  if (!ah)
    return AEE_ENOMEMORY;
  int16_t *tables = (int16_t *)(work + 2u * bytes);
  int32_t *result = (int32_t *)((uint8_t *)tables + (size_t)K * 32u);
  uint8_t *expand = (uint8_t *)(result + N);
  hvx_expand_i2i4_table(pal, expand);
  memset(ah, 0, (size_t)K * 64u);
  for (uint32_t kt = 0; kt < K / 32u; ++kt)
    memcpy(ah + (size_t)kt * 2048u, act + kt * 32u, 32u);
  memcpy(compact, weight, bytes);
  if (mode == 4u || mode == 6u || mode == 8u || mode == 10u)
    for (uint32_t kt = 0; kt < K / 32u; ++kt)
      memcpy(work + (size_t)kt * N * 16u + N * 8u, weight + (size_t)kt * N * 8u,
             N * 8u);
  if (mode == 11u)
    for (uint32_t kt = 0; kt < K / 32u; ++kt)
      memcpy(work + (size_t)kt * N * 16u, weight + (size_t)kt * N * 8u, N * 8u);
  hvx_worker_pool *pool = workers == 1u ? NULL : s->quant_pool;
  lut_probe_ctx c = {K,   N,      mode,   ah, compact, expand,
                     pal, tables, result, 0u, work};
  memset(stats, 0, (size_t)statsLen * sizeof(*stats));
  /* Three unrecorded warmups. DMA here is serial whole-matrix transfer,
   * deliberately not a claim about the MoE's cross-expert overlap. */
  for (uint32_t pass = 0; pass < passes + 3u; ++pass) {
    const uint64_t t0 = hexkl_probe_now();
    if (dma) {
      const uint32_t idx = hexkl_dma_ring_next_idx();
      hexkl_dma_ring_push2d(
        mode == 11u ? work
        : (mode == 4u || mode == 6u || mode == 8u || mode == 10u)
          ? work + N * 8u
          : compact,
        weight,
        (mode == 4u || mode == 6u || mode == 8u || mode == 10u || mode == 11u)
          ? N * 16u
          : N * 8u,
        N * 8u, N * 8u, K / 32u, 0, 1);
      hexkl_dma_ring_wait(idx);
    }
    const uint64_t t1 = hexkl_probe_now();
    c.weight = compact;
    if (mode == 3u || mode == 4u || (mode == 6u || mode == 8u || mode == 10u))
      hvx_worker_pool_run(pool, repack_worker, &c, K / 32u);
    const uint64_t t2 = hexkl_probe_now();
    if (mode == 0u)
      hvx_expand_i2i4_pool(compact, (uint32_t)bytes, expand, work, pool);
    if (mode == 1u || mode >= 3u)
      hvx_worker_pool_run(pool, prepare_worker, &c, K / 32u);
    const uint64_t t3 = hexkl_probe_now();
    c.weight = mode == 0u || mode == 3u || mode == 4u || mode == 11u ||
                   (mode == 6u || mode == 8u || mode == 10u)
                 ? work
                 : compact;
    const uint32_t units = N / (mode >= 5u && N % 256u == 0u
                                  ? 256u
                                  : (mode == 1u || mode >= 3u ? 128u : 32u));
    hvx_worker_pool_run(pool, gemv_worker, &c,
                        units < workers ? units : workers);
    const uint64_t t4 = hexkl_probe_now();
    if (pass >= 3u) {
      stats[0] += (uint32_t)(t1 - t0);
      stats[1] += (uint32_t)(t2 - t1);
      stats[2] += (uint32_t)(t3 - t2);
      stats[3] += (uint32_t)(t4 - t3);
      stats[4] += (uint32_t)(t4 - t0);
    }
  }
  memcpy(out, result, N * sizeof(*out));
  stats[5] = c.threads;
  free(ah);
  return AEE_SUCCESS;
}
