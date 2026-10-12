/** Host harness for hexkl_mm_u8i4_layer_run (the FC / projection call) on
   the scalar stand-ins: several handles of different widths against one
   activation, as the attention q/k/v call makes it, checked row by row
   against the reference. What it checks is the batching added for the
   pooled epilogue -- that staged slot j carries n-tile nt0+j of the right
   handle and lands at its columns of the right output block, across row
   blocks, across handles, and through a partial last row block -- not the
   HMX's arithmetic, which the stubs define self-consistently.
 * @file fc_layer_host_check.c
 * @brief Regression checks for fc layer host check.
 * @author NNTrainer contributors
 * @bug See the repository issue tracker for known issues.
 */
#include "hexkl_mm_opts.h"
#include "hexkl_mm_u8i4_dma.h"
#include "hexkl_probe.h"
#include "hvx_scalar_stubs.h"
#include <AEEStdErr.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void reference(const W *const *ws, uint32_t n_h, const float *x,
                      uint32_t M, uint32_t K, float *want, float base) {
  uint8_t *xq = (uint8_t *)malloc(K);
  size_t off = 0;
  for (uint32_t i = 0; i < n_h; ++i) {
    for (uint32_t t = 0; t < M; ++t) {
      float xs;
      int32_t xz;
      quant_row(x + (size_t)t * K, K, xq, &xs, &xz);
      ref_mm(ws[i], xq, xs, xz, want + off + (size_t)t * ws[i]->N);
      if (base != 0.f) {
        for (uint32_t c = 0; c < ws[i]->N; ++c)
          want[off + (size_t)t * ws[i]->N + c] += base;
      }
    }
    off += (size_t)M * ws[i]->N;
  }
  free(xq);
}

static int compare(const char *what, const float *got, const float *want,
                   size_t n) {
  double worst = 0.0;
  size_t bad = 0;
  for (size_t i = 0; i < n; ++i) {
    double d = fabs((double)got[i] - (double)want[i]);
    double s = fabs((double)want[i]) + 1e-6;
    if (d / s > 1e-5)
      ++bad;
    if (d / s > worst)
      worst = d / s;
  }
  printf("%-18s: mismatches=%zu of %zu   worst_rel=%g\n", what, bad, n, worst);
  return bad != 0;
}

int main(void) {
  /** Three row blocks with a short last one; the widest handle is 33
     n-tiles, one more than a staging batch, so the first handle takes two
     batches per row block and the parity crosses handles unevenly. */
  const uint32_t M = 150, K = 64;
  const uint32_t N[3] = {1056, 512, 512};
  static uint8_t vtcm[8u << 20];
  W w[3];
  const W *ws[3];
  uint32_t handles[3];
  size_t out_n = 0;
  for (uint32_t i = 0; i < 3; ++i) {
    make_weight(i, K, N[i], &w[i]);
    ws[i] = &w[i];
    handles[i] = i;
    out_n += (size_t)M * N[i];
  }
  float *x = (float *)malloc(sizeof(float) * M * K);
  for (uint32_t i = 0; i < M * K; ++i)
    x[i] = rndf();
  float *got = (float *)malloc(sizeof(float) * out_n);
  float *want = (float *)malloc(sizeof(float) * out_n);
  int fail = 0;

  int rc = hexkl_mm_u8i4_layer_run(&g_tbl, vtcm, sizeof vtcm, sizeof vtcm, M, K,
                                   handles, 3, x, got, NULL);
  printf("run rc=%d\n", rc);
  if (rc)
    return 1;
  reference(ws, 3, x, M, K, want, 0.f);
  fail |= compare("three handles", got, want, out_n);

  /* One row: decode's shape, 63 padding rows never emitted. */
  rc = hexkl_mm_u8i4_layer_run(&g_tbl, vtcm, sizeof vtcm, sizeof vtcm, 1, K,
                               handles, 3, x, got, NULL);
  reference(ws, 3, x, 1, K, want, 0.f);
  fail |= (rc != 0) | compare("M=1", got, want, N[0] + N[1] + N[2]);

  // Mix an arena image, a flushed heap image and a dirty heap image.
  // Only the first two may bypass L2; DMA must preserve every output bit.
  g_tbl.slots[0].borrowed = 1;
  g_tbl.slots[1].clean = 1;
  for (uint32_t m = 1; m <= M; m += M - 1u) {
    const size_t count = (size_t)m * (N[0] + N[1] + N[2]);
    hexkl_mm_opts opts = {0};
    rc = hexkl_mm_u8i4_layer_run(&g_tbl, vtcm, sizeof vtcm, sizeof vtcm, m, K,
                                 handles, 3, x, want, &opts);
    fail |= rc != 0;
    opts.weight_src_bypass = 1;
    g_weight_bypass_bytes = g_invalid_bypass_bytes = 0;
    rc = hexkl_mm_u8i4_layer_run(&g_tbl, vtcm, sizeof vtcm, sizeof vtcm, m, K,
                                 handles, 3, x, got, &opts);
    const uint64_t expected = (uint64_t)K * (N[0] + N[1]) / 2u;
    const int same = memcmp(got, want, count * sizeof(float)) == 0;
    printf("FC M=%u bypass: identical=%d bytes=%llu expected=%llu bad=%llu\n",
           m, same, (unsigned long long)g_weight_bypass_bytes,
           (unsigned long long)expected,
           (unsigned long long)g_invalid_bypass_bytes);
    fail |= rc != 0 || !same || g_weight_bypass_bytes != expected ||
            g_invalid_bypass_bytes != 0;

    // Preserve accumulation and non-contiguous column-slice outputs.
    const uint32_t offsets[3] = {N[2], N[2] + N[0], 0};
    const uint32_t strides[3] = {N[0] + N[1] + N[2], N[0] + N[1] + N[2],
                                 N[0] + N[1] + N[2]};
    opts.out_off = offsets;
    opts.out_ld = strides;
    opts.accumulate = 1;
    for (size_t j = 0; j < count; ++j)
      got[j] = want[j] = 0.25f;
    rc = hexkl_mm_u8i4_layer_run(&g_tbl, vtcm, sizeof vtcm, sizeof vtcm, m, K,
                                 handles, 3, x, got, &opts);
    fail |= rc != 0;
    rc = hexkl_mm_u8i4_layer_run(&g_tbl, vtcm, sizeof vtcm, sizeof vtcm, m, K,
                                 handles, 3, x, want, &opts);
    fail |= rc != 0 || memcmp(got, want, count * sizeof(float)) != 0;
  }

  /** M=1 compact slabs: tail columns, handle changes, accumulation, and
     caller-supplied output offsets must match the HMX path exactly. */
  {
    uint8_t ah[64u * 64u] __attribute__((aligned(128)));
    const size_t count = N[0] + N[1] + N[2];
    const uint32_t offsets[] = {N[2], N[2] + N[0], 0};
    const uint32_t strides[] = {count, count, count};
    for (uint32_t chunk = 4; chunk <= 32; chunk *= 2) {
      for (int accumulate = 0; accumulate < 2; ++accumulate) {
        hexkl_mm_opts o = {.out_off = offsets,
                           .out_ld = strides,
                           .accumulate = accumulate,
                           .weight_src_bypass = 1};
        for (size_t i = 0; i < count; ++i)
          got[i] = want[i] = 0.25f;
        rc = hexkl_mm_u8i4_layer_run(&g_tbl, vtcm, sizeof vtcm, sizeof vtcm, 1,
                                     K, handles, 3, x, want, &o);
        fail |= rc != 0;
        o.m1_scratch = ah;
        o.m1_chunk_tiles = chunk;
        g_weight_bypass_bytes = g_invalid_bypass_bytes = 0;
        rc = hexkl_mm_u8i4_layer_run(&g_tbl, vtcm, sizeof vtcm, sizeof vtcm, 1,
                                     K, handles, 3, x, got, &o);
        fail |= rc != 0 || memcmp(got, want, count * sizeof(float)) != 0;
        fail |= g_invalid_bypass_bytes != 0 ||
                g_weight_bypass_bytes != (uint64_t)K * (N[0] + N[1]) / 2u;
        printf("HVX chunk=%u accumulate=%d rc=%d exact=%d\n", chunk, accumulate,
               rc, memcmp(got, want, count * sizeof(float)) == 0);
      }
    }
    hexkl_mm_opts o = {.m1_scratch = ah, .m1_chunk_tiles = 8};
    rc = hexkl_mm_u8i4_layer_run(&g_tbl, vtcm, 32u * K - 1u, sizeof vtcm, 1, K,
                                 handles, 3, x, got, &o);
    fail |= rc != AEE_ENOMEMORY || memcmp(got, want, count * sizeof(float));
    /* A one-tile grant forces slab shrinking; no exported result is lost. */
    rc = hexkl_mm_u8i4_layer_run(&g_tbl, vtcm, 32u * K, sizeof vtcm, 1, K,
                                 handles, 3, x, got, &o);
    reference(ws, 3, x, 1, K, want, 0.f);
    fail |= rc != 0 || compare("HVX tiny arena", got, want, count);
  }

  /* accumulate: the synchronous path, adding into what is there. */
  {
    hexkl_mm_opts o = {0};
    o.accumulate = 1;
    for (size_t i = 0; i < out_n; ++i)
      got[i] = 1.0f;
    rc = hexkl_mm_u8i4_layer_run(&g_tbl, vtcm, sizeof vtcm, sizeof vtcm, M, K,
                                 handles, 3, x, got, &o);
    reference(ws, 3, x, M, K, want, 1.0f);
    fail |= (rc != 0) | compare("accumulate", got, want, out_n);
  }

  /** An arena with room for exactly two staging tiles still runs (one
     tile per batch); one tile short of that is refused. */
  {
    const uint32_t k_tiles = K / 32u, m_pad = 192u;
    const uint32_t act = (m_pad / 64u) * k_tiles * 2048u;
    const uint32_t wb = k_tiles * (N[0] / 32u) * 512u;
    const uint32_t result_off = act + 2u * wb;
    rc = hexkl_mm_u8i4_layer_run(&g_tbl, vtcm, result_off + 2u * 8192u,
                                 result_off + 2u * 8192u, M, K, handles, 3, x,
                                 got, NULL);
    reference(ws, 3, x, M, K, want, 0.f);
    fail |= (rc != 0) | compare("2-tile arena", got, want, out_n);
    rc = hexkl_mm_u8i4_layer_run(&g_tbl, vtcm, result_off + 8192u,
                                 result_off + 8192u, M, K, handles, 3, x, got,
                                 NULL);
    printf("1-tile arena      : rc=%d (want %d)\n", rc, AEE_ENOMEMORY);
    fail |= (rc != AEE_ENOMEMORY);
  }

  printf(fail ? "\nFAIL\n" : "\nALL CHECKS PASS\n");
  return fail;
}
