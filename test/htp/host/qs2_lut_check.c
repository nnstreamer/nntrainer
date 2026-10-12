// SPDX-License-Identifier: Apache-2.0
/**
 * @file qs2_lut_check.c
 * @brief Independent dense reference, WH roundtrip and optional device probe.
 */
#include "hvx_gemv_u8i2_lut.h"
#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef NNTR_LUT_DEVICE
#include "nntr_hvx.h"
#include <AEEStdErr.h>
#include <remote.h>
static remote_handle64 session;
#endif

static uint32_t random_state = 12345u;
static uint32_t next_random(void) {
  random_state ^= random_state << 13;
  random_state ^= random_state >> 17;
  random_state ^= random_state << 5;
  return random_state;
}

static void *buffer(size_t bytes) {
  void *p = NULL;
  if (posix_memalign(&p, 128u, bytes) || !p)
    abort();
  memset(p, 0, bytes);
  return p;
}

static void check(uint32_t K, uint32_t N, const int8_t *pal, uint32_t pattern,
                  uint32_t passes) {
  const size_t bytes = (size_t)K * N / 4u;
  uint8_t *wh = buffer(bytes), *packed = buffer(bytes + 128u);
  uint8_t *back = buffer(bytes), *act = buffer(K);
  uint8_t *offline = buffer(bytes);
  uint8_t *rows = buffer(bytes * 2u + 128u);
  uint8_t *ah = buffer((size_t)K * 64u);
  int16_t *tables = buffer((size_t)K * 32u);
  int32_t *ref = buffer(N * 4u), *out = buffer(N * 4u);
  int32_t *sums = buffer(N * 4u);
  uint8_t *expand = buffer(128u);
  for (uint32_t i = 0; i < 16u; ++i)
    expand[2u * i] = (pal[i & 3u] & 15) | ((pal[i >> 2u] & 15) << 4);
  memset(packed + bytes, 0xa5, 128u);
  for (uint32_t k = 0; k < K; ++k) {
    act[k] = pattern == 0u ? (uint8_t)next_random() : pattern == 1u ? 255u : 0u;
    ah[(size_t)(k / 32u) * 2048u + k % 32u] = act[k];
    for (uint32_t n = 0; n < N; ++n) {
      const uint32_t code = pattern == 0u ? next_random() & 3u : (n / 32u) % 4u;
      /* Independent coordinate encoder for the EXISTING compact WH file.
       * Dense reference consumes the logical code before any packing. */
      const uint32_t r = k % 32u, c = n % 32u;
      const uint32_t old_byte = (r / 8u) * 128u + c * 4u + r % 4u;
      const size_t b = ((size_t)(k / 32u) * (N / 32u) + n / 32u) * 256u +
                       (old_byte / 256u) * 128u + old_byte % 128u;
      const uint32_t shift = 2u * ((r / 4u) % 2u) + 4u * ((r / 8u) % 2u);
      wh[b] |= code << shift;
      ref[n] += act[k] * pal[code];
      sums[n] += pal[code];
    }
  }
  assert(hvx_i2_lut_pack_wh(wh, K, N, packed) == 0);
  assert(hvx_i2_lut_unpack_wh(packed, K, N, back) == 0);
  assert(!memcmp(wh, back, bytes));
  hvx_i2_lut_repack(wh, N, N * 8u, back, 1024u, (K / 32u) * 1024u, 0, K / 32u);
  assert(!memcmp(packed, back, bytes));
  hvx_i2_lut_repack(wh, N, N * 8u, offline, N * 8u, 1024u, 0, K / 32u);
  memcpy(back, offline, bytes);
  for (uint32_t kt = 0; kt < K / 32u; ++kt)
    hvx_i2_lut_unrepack_row(back + (size_t)kt * N * 8u, N * 8u);
  assert(!memcmp(wh, back, bytes));
  for (uint32_t kt = 0; kt < K / 32u; ++kt)
    memcpy(rows + (size_t)kt * N * 16u + N * 8u, wh + (size_t)kt * N * 8u,
           N * 8u);
  memset(rows + bytes * 2u, 0xa5, 128u);
  hvx_i2_lut_repack(rows + N * 8u, N, N * 16u, rows, N * 16u, 1024u, 0,
                    K / 32u);
  for (uint32_t i = 0; i < 128u; ++i)
    assert(rows[bytes * 2u + i] == 0xa5);
  for (uint32_t i = 0; i < 128u; ++i)
    assert(packed[bytes + i] == 0xa5);
  hvx_i2_lut_prepare(ah, pal, 0, K / 32u, tables);
  for (uint32_t n = 0; n < N; n += 128u)
    hvx_gemv_u8i2_lut_panel(packed, K, n / 128u, tables, out + n);
  assert(!memcmp(out, ref, N * 4u));
  for (uint32_t n = 0; n < N; n += 128u)
    hvx_gemv_u8i2_lut_strided(rows + (n / 128u) * 1024u, K / 32u, N * 16u,
                              tables, out + n);
  assert(!memcmp(out, ref, N * 4u));
  for (uint32_t n = 0; n < N; n += 32u)
    hvx_gemv_u8i2_wh_fused_col(ah, K / 32u, wh, N / 32u, n / 32u, expand,
                               out + n);
  assert(!memcmp(out, ref, N * 4u));
  for (uint32_t n = 0; n + 256u <= N; n += 256u)
    hvx_gemv_u8i2_lut_pair(packed + (size_t)n * K / 4u,
                           packed + (size_t)(n + 128u) * K / 4u, K / 32u, 1024u,
                           tables, out + n, out + n + 128u);
  assert(!memcmp(out, ref, N * 4u));
  memset(tables, 0xa5, (size_t)K * 32u);
  hvx_i2_lut_prepare_dense(ah, pal, 0, K / 32u, tables);
  for (uint32_t n = 0; n < N; n += 256u) {
    const uint32_t other = n + 128u < N ? n + 128u : n;
    hvx_gemv_u8i2_lut_dense_pair(packed + (size_t)n * K / 4u,
                                 packed + (size_t)other * K / 4u, K / 32u,
                                 1024u, tables, out + n, out + other);
  }
  assert(!memcmp(out, ref, N * 4u));
  for (size_t i = (size_t)K * 16u; i < (size_t)K * 32u; ++i)
    assert(((uint8_t *)tables)[i] == 0xa5);
  hvx_i2_lut_prepare_centered(ah, pal, 0, K / 32u, tables);
  for (uint32_t n = 0; n < N; n += 256u) {
    const uint32_t other = n + 128u < N ? n + 128u : n;
    hvx_gemv_u8i2_lut_pair_centered(packed + (size_t)n * K / 4u,
                                    packed + (size_t)other * K / 4u, K / 32u,
                                    1024u, tables, out + n, out + other);
  }
  hvx_i2_lut_restore(out, sums, N);
  assert(!memcmp(out, ref, N * 4u));
#ifdef NNTR_LUT_DEVICE
  const uint32_t worker_counts[] = {1u, 4u, 6u, 8u};
  for (uint32_t wi = 0; wi < 4u; ++wi) {
    const uint32_t workers = worker_counts[wi];
    for (uint32_t dma = 0u; dma <= 256u; dma += 256u) {
      for (uint32_t mode = 0; mode < 12u; ++mode) {
        const uint32_t runs = passes;
        uint32_t stats[6];
        int rc = nntr_hvx_lut_gemv_probe(
          session, K, N, mode | dma, workers, runs, act, K, pal, 4,
          mode == 11u                                            ? offline
          : mode == 1u || mode == 5u || mode == 7u || mode == 9u ? packed
                                                                 : wh,
          (int)bytes, out, N, stats, 6);
        if (rc) {
          fprintf(stderr, "probe error: K=%u N=%u mode=%u rc=%#x\n", K, N, mode,
                  rc);
          abort();
        }
        for (uint32_t n = 0; n < N; ++n) {
          if (mode == 7u || mode == 8u)
            out[n] += 127 * sums[n];
          if (out[n] != ref[n]) {
            fprintf(stderr, "mismatch: K=%u N=%u mode=%u n=%u got=%d want=%d\n",
                    K, N, mode, n, out[n], ref[n]);
            abort();
          }
        }
        printf("LUT_GEMV K=%u N=%u pattern=%u palette=%d,%d,%d,%d mode=%u "
               "workers=%u dma=%u passes=%u dma_us=%.3f repack_us=%.3f "
               "prep_us=%.3f gemv_us=%.3f total_us=%.3f exact=1\n",
               K, N, pattern, pal[0], pal[1], pal[2], pal[3], mode, stats[5],
               !!dma, runs, (double)stats[0] / runs, (double)stats[1] / runs,
               (double)stats[2] / runs, (double)stats[3] / runs,
               (double)stats[4] / runs);
        fflush(stdout);
      }
    }
  }
#else
  (void)passes;
  printf("QS2 LUT exact: K=%u N=%u pattern=%u palette=%d,%d,%d,%d\n", K, N,
         pattern, pal[0], pal[1], pal[2], pal[3]);
#endif
  free(sums);
  free(wh);
  free(packed);
  free(back);
  free(offline);
  free(act);
  free(ah);
  free(tables);
  free(ref);
  free(out);
  free(expand);
  free(rows);
}

int main(int argc, char **argv) {
  const uint32_t passes = argc > 1 ? (uint32_t)strtoul(argv[1], NULL, 10) : 20u;
#ifdef NNTR_LUT_DEVICE
  struct remote_rpc_control_unsigned_module u = {CDSP_DOMAIN_ID, 1};
  int rc =
    remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &u, sizeof(u));
  assert(rc == AEE_SUCCESS);
  rc = nntr_hvx_open(nntr_hvx_URI "&_dom=cdsp", &session);
  assert(rc == AEE_SUCCESS);
#endif
  const int8_t palettes[][4] = {
    {-8, -3, 0, 7}, {-2, -1, 0, 1}, {-8, -8, 7, 7}, {7, 0, -3, -8}};
  uint8_t guard[128] = {0};
  assert(hvx_i2_lut_pack_wh(guard, 31, 128, guard + 64) == -1);
  assert(hvx_i2_lut_pack_wh(guard, 32, 32, guard + 64) == -1);
  assert(hvx_i2_lut_pack_wh(guard, 32, 128, guard) == -1);
  const int model_only = argc > 2 && !strcmp(argv[2], "model");
  if (!model_only)
    for (uint32_t p = 0; p < 4u; ++p)
      for (uint32_t pattern = 0; pattern < 3u; ++pattern)
        check(64u, 256u, palettes[p], pattern, passes);
  check(2048u, 3072u, palettes[0], 0u, passes);
  check(1536u, 2048u, palettes[0], 0u, passes);
  if (!model_only)
    check(4096u, 128u, palettes[2], 1u, passes);
#ifdef NNTR_LUT_DEVICE
  assert(nntr_hvx_close(session) == AEE_SUCCESS);
#endif
  puts("QS2 LUT: all layout, arithmetic and guard checks passed");
  return 0;
}
