/* Host harness for hexkl_conv_block_run: the kernel on the scalar stand-ins
   (hvx_scalar_stubs.c) against a row-at-a-time reference of the whole
   block. What it checks is the loop structure -- that a and c pair up at
   the same columns, that g's rows land where phase 2's two-row look-back
   expects them, that the second walk over the rows sees the right slot
   contents after the weight swap, that the state is g's last two rows --
   not the HMX's arithmetic, which the stubs define self-consistently. */
#include "hexkl_conv_block.h"
#include "hexkl_probe.h"
#include "hvx_scalar_stubs.h"
#include <AEEStdErr.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* The reference: one row of x at a time through in_proj's three slices,
   the conv over the previous rows' gate values, out_proj. */
static void ref_block(const W *wa, const W *wb, const W *wc, const W *wo,
                      const float *conv_w, const float *x, uint32_t M,
                      uint32_t K, uint32_t C, uint32_t N_out, float *out,
                      float *state) {
  uint8_t *xq = (uint8_t *)malloc(K);
  uint8_t *zq = (uint8_t *)malloc(C);
  float *a = (float *)malloc(sizeof(float) * C);
  float *b = (float *)malloc(sizeof(float) * C);
  float *c = (float *)malloc(sizeof(float) * C);
  float *history = (float *)malloc((size_t)(M + 2u) * C * sizeof(float));
  memcpy(history, state, 2u * C * sizeof(float));
  float *g = history + 2u * C;
  float *z = (float *)malloc(sizeof(float) * C);
  for (uint32_t t = 0; t < M; ++t) {
    float xs;
    int32_t xz;
    quant_row(x + (size_t)t * K, K, xq, &xs, &xz);
    ref_mm(wa, xq, xs, xz, a);
    ref_mm(wc, xq, xs, xz, c);
    for (uint32_t j = 0; j < C; ++j)
      g[(size_t)t * C + j] = a[j] * c[j];
  }
  for (uint32_t t = 0; t < M; ++t) {
    float xs, zs;
    int32_t xz, zz;
    quant_row(x + (size_t)t * K, K, xq, &xs, &xz);
    ref_mm(wb, xq, xs, xz, b);
    for (uint32_t j = 0; j < C; ++j) {
      const float x0 = g[(size_t)t * C + j];
      const float x1 = history[(size_t)(t + 1u) * C + j];
      const float x2 = history[(size_t)t * C + j];
      volatile float p0 = conv_w[j] * x0;
      volatile float p1 = conv_w[C + j] * x1;
      volatile float p2 = conv_w[2u * C + j] * x2;
      volatile float y = p0 + p1;
      y = y + p2;
      z[j] = b[j] * y;
    }
    quant_row(z, C, zq, &zs, &zz);
    ref_mm(wo, zq, zs, zz, out + (size_t)t * N_out);
  }
  memcpy(state, history + (size_t)M * C, 2u * C * sizeof(float));
  free(xq);
  free(zq);
  free(a);
  free(b);
  free(c);
  free(history);
  free(z);
}

static int compare(const char *what, const float *got, const float *want,
                   size_t n) {
  double worst = 0.0;
  size_t bad = 0;
  for (size_t i = 0; i < n; ++i) {
    double d = fabs((double)got[i] - (double)want[i]);
    double s = fabs((double)want[i]) + 1e-6;
    if (!isfinite(got[i]) || !isfinite(want[i]) || d / s > 1e-5)
      ++bad;
    if (d / s > worst)
      worst = d / s;
  }
  printf("%-18s: mismatches=%zu of %zu   worst_rel=%g\n", what, bad, n, worst);
  return bad != 0;
}

int main(void) {
  /* Shapes small enough to run in seconds and awkward on purpose: M is
     three blocks with a short last one; C is 17 tiles, one more than the
     16 pairs a staging buffer holds, so phase 1 has a full batch and a
     one-pair batch and the second staging buffer is used; N_out is 33
     tiles, two out_proj batches; K != C != N_out, so a slot sized for the
     wrong weight would show. */
  const uint32_t M = 150, K = 64, C = 544, N_out = 1056;
  static uint8_t vtcm[8u << 20];

  hexkl_conv_block_layout L;
  int rc = hexkl_conv_block_layout_get(K, C, N_out, sizeof vtcm, &L);
  printf("layout rc=%d total=%u (act %u A %u B %u z %u mid %u conv_w %u "
         "stage %u tiles %u)\n",
         rc, L.total, L.act_off, L.w_a_off, L.w_b_off, L.z_off, L.mid_off,
         L.conv_w_off, L.result_off, L.acc_tiles);
  if (rc)
    return 1;

  W wa, wb, wc, wo;
  make_weight(0, K, C, &wa);
  make_weight(1, K, C, &wb);
  make_weight(2, K, C, &wc);
  make_weight(3, C, N_out, &wo);
  g_tbl.slots[0].borrowed = 1;
  g_tbl.slots[2].clean = 1;
  g_tbl.slots[3].borrowed = 1;
  g_weight_bypass_bytes = g_invalid_bypass_bytes = 0;

  float *x = (float *)malloc(sizeof(float) * M * K);
  for (uint32_t i = 0; i < M * K; ++i)
    x[i] = rndf();
  float *conv_w = (float *)malloc(sizeof(float) * 3 * C);
  for (uint32_t i = 0; i < 3 * C; ++i)
    conv_w[i] = rndf();

  float *got = (float *)malloc(sizeof(float) * M * N_out);
  float *got_state = (float *)calloc(2u * C, sizeof(float));
  hexkl_moe_scratch scratch = {NULL, NULL, 0};
  rc = hexkl_conv_block_run(&g_tbl, vtcm, sizeof vtcm, sizeof vtcm, M, K, C,
                            N_out, 0, 1, 2, 3, conv_w, x, got, got_state, NULL,
                            &scratch, 1, 0u);
  printf("run rc=%d\n", rc);
  if (rc)
    return 1;

  float *want = (float *)malloc(sizeof(float) * M * N_out);
  float *want_state = (float *)calloc(2u * C, sizeof(float));
  ref_block(&wa, &wb, &wc, &wo, conv_w, x, M, K, C, N_out, want, want_state);
  int fail = compare("output", got, want, (size_t)M * N_out);
  fail |= compare("conv state", got_state, want_state, 2u * C);
  const uint64_t bypass_bytes = (uint64_t)K * C + (uint64_t)C * N_out / 2u;
  fail |= g_weight_bypass_bytes != bypass_bytes || g_invalid_bypass_bytes != 0;
  printf("conv weight bypass: bytes=%llu expected=%llu bad=%llu\n",
         (unsigned long long)g_weight_bypass_bytes,
         (unsigned long long)bypass_bytes,
         (unsigned long long)g_invalid_bypass_bytes);
  printf(fail ? "CONV BLOCK DIFFERS\n" : "CONV BLOCK MATCHES REFERENCE\n");

  /* Every weight pushed exactly once: a, b, c (K x C) and out (C x N_out).
     The DMA stub completes at push, so a missing push cannot show as a
     wrong result -- only this counter catches it. */
  {
    const uint64_t in_kb = ((K / 32u) * (C / 32u) * 512u) >> 10;
    const uint64_t out_kb = ((C / 32u) * (N_out / 32u) * 512u) >> 10;
    const uint64_t want_kb = 3u * in_kb + out_kb;
    printf("weight DMA        : %llu KB (want %llu)\n",
           (unsigned long long)hexkl_probe_us[HEXKL_PROBE_DMA_KB],
           (unsigned long long)want_kb);
    if (hexkl_probe_us[HEXKL_PROBE_DMA_KB] != want_kb)
      fail = 1;
    printf("blocks            : %llu (want 3)\n",
           (unsigned long long)hexkl_probe_us[HEXKL_PROBE_BLOCKS]);
    if (hexkl_probe_us[HEXKL_PROBE_BLOCKS] != 3u)
      fail = 1;
  }

  /* Edge: one row -- state row 0 is zero, and the block has no look-back. */
  {
    float *got1 = (float *)malloc(sizeof(float) * N_out);
    float *st1 = (float *)calloc(2u * C, sizeof(float));
    float *want1 = (float *)malloc(sizeof(float) * N_out);
    float *wst1 = (float *)calloc(2u * C, sizeof(float));
    int r = hexkl_conv_block_run(&g_tbl, vtcm, sizeof vtcm, sizeof vtcm, 1, K,
                                 C, N_out, 0, 1, 2, 3, conv_w, x, got1, st1,
                                 NULL, &scratch, 1, 0u);
    ref_block(&wa, &wb, &wc, &wo, conv_w, x, 1, K, C, N_out, want1, wst1);
    int ok = (r == 0) && !compare("M=1 output", got1, want1, N_out) &&
             !compare("M=1 state", st1, wst1, 2u * C);
    printf("one row           : %s\n", ok ? "ok" : "WRONG");
    fail |= !ok;
    free(got1);
    free(st1);
    free(want1);
    free(wst1);
  }
  /* Repeated decode and ragged prefill must retain both previous rows.
     Nonzero initial history catches kernels that silently reset on entry. */
  {
    const uint32_t chunks[] = {1, 1, 2, 33, 65, 48};
    float *split = (float *)malloc(sizeof(float) * M * N_out);
    float *split_state = (float *)malloc(sizeof(float) * 2 * C);
    for (uint32_t j = 0; j < 2u * C; ++j)
      split_state[j] = want_state[j] = 0.125f * rndf();
    ref_block(&wa, &wb, &wc, &wo, conv_w, x, M, K, C, N_out, want, want_state);
    uint32_t row = 0;
    for (size_t i = 0; i < sizeof(chunks) / sizeof(chunks[0]); ++i) {
      int r = hexkl_conv_block_run(
        &g_tbl, vtcm, sizeof vtcm, sizeof vtcm, chunks[i], K, C, N_out, 0, 1, 2,
        3, conv_w, x + (size_t)row * K, split + (size_t)row * N_out,
        split_state, NULL, &scratch, 1, 0u);
      fail |= r != 0;
      row += chunks[i];
    }
    fail |= compare("continued output", split, want, (size_t)M * N_out);
    fail |= compare("continued state", split_state, want_state, 2u * C);
    free(split);
    free(split_state);
  }
  /* Every supported chunk, including partial final slabs and repeated M=1
     state advancement. The unequal dimensions catch packed-WH stride bugs. */
  void *packed_alloc[4];
  uint8_t *packed[4];
  for (uint32_t i = 0; i < 4u; ++i) {
    const hexkl_weight_u8i4 *w = &g_tbl.slots[i];
    packed_alloc[i] = malloc((size_t)w->K * w->N / 2u + 511u);
    packed[i] =
      (uint8_t *)(((uintptr_t)packed_alloc[i] + 511u) & ~(uintptr_t)511u);
  }
  for (uint32_t variant = 0; variant < 32u; ++variant) {
    const uint32_t code = (variant % 16u) / 4u, out_code = variant % 4u;
    const uint32_t flags = HTP_CONV_FLAG_M1_HVX |
                           (code << HTP_CONV_CHUNK_SHIFT) |
                           (out_code << HTP_CONV_OUT_CHUNK_SHIFT);
    printf("HVX input/output tiles: %u/%u\n", htp_conv_chunk_tiles(flags),
           htp_conv_out_chunk_tiles(flags));
    for (uint32_t i = 0; i < 4u; ++i) {
      const uint32_t chunk =
        i == 3u ? htp_conv_out_chunk_tiles(flags) : htp_conv_chunk_tiles(flags);
      fail |= hexkl_conv_block_prepare_decode(
                &g_tbl, vtcm, sizeof vtcm, i, packed[i],
                variant < 16u ? 0u : chunk) != AEE_SUCCESS;
      if (variant >= 16u) {
        const hexkl_weight_u8i4 *w = &g_tbl.slots[i];
        const uint32_t kt = w->K / 32u, nt = w->N / 32u;
        for (uint32_t n0 = 0; n0 < nt; n0 += chunk) {
          const uint32_t cols = nt - n0 < chunk ? nt - n0 : chunk;
          for (uint32_t k = 0; k < kt; ++k)
            fail |= memcmp(packed[i] + ((size_t)n0 * kt + k * cols) * 512u,
                           w->wh_bytes + ((size_t)k * nt + n0) * 512u,
                           cols * 512u) != 0;
        }
      }
    }
    memset(got_state, 0, 2u * C * sizeof(float));
    memset(want_state, 0, 2u * C * sizeof(float));
    for (uint32_t row = 0; row < 3u; ++row) {
      const int r = hexkl_conv_block_run(
        &g_tbl, vtcm, sizeof vtcm, sizeof vtcm, 1u, K, C, N_out, 0, 1, 2, 3,
        conv_w, x + row * K, got, got_state, NULL, &scratch, 1, flags);
      ref_block(&wa, &wb, &wc, &wo, conv_w, x + row * K, 1u, K, C, N_out, want,
                want_state);
      fail |= r != AEE_SUCCESS;
      fail |= compare("HVX chunk output", got, want, N_out);
      fail |= compare("HVX chunk state", got_state, want_state, 2u * C);
    }
    if (variant >= 16u) {
      /* A different input chunk must safely use original WH while any
         still-matching output image remains usable. */
      const uint32_t other_flags = flags ^ (1u << HTP_CONV_CHUNK_SHIFT);
      fail |= hexkl_conv_block_run(&g_tbl, vtcm, sizeof vtcm, sizeof vtcm, 1u,
                                   K, C, N_out, 0, 1, 2, 3, conv_w, x + 3u * K,
                                   got, got_state, NULL, &scratch, 1,
                                   other_flags) != AEE_SUCCESS;
      ref_block(&wa, &wb, &wc, &wo, conv_w, x + 3u * K, 1u, K, C, N_out, want,
                want_state);
      fail |= compare("chunk fallback output", got, want, N_out);
      fail |= compare("chunk fallback state", got_state, want_state, 2u * C);
    }
    /* A rejected scratch size must leave both exported buffers untouched. */
    memcpy(want, got, N_out * sizeof(float));
    memcpy(want_state, got_state, 2u * C * sizeof(float));
    const int r = hexkl_conv_block_run(&g_tbl, vtcm, 128u, 128u, 1u, K, C,
                                       N_out, 0, 1, 2, 3, conv_w, x, got,
                                       got_state, NULL, &scratch, 1, flags);
    fail |= r != AEE_ENOMEMORY;
    fail |= memcmp(got, want, N_out * sizeof(float)) != 0;
    fail |= memcmp(got_state, want_state, 2u * C * sizeof(float)) != 0;
  }
  /* Invalid preparation is atomic: preserve the previously bound image. */
  const uint8_t *saved = g_tbl.slots[0].conv_decode_bytes;
  const uint32_t saved_chunk = g_tbl.slots[0].conv_decode_chunk;
  fail |= hexkl_conv_block_prepare_decode(&g_tbl, vtcm, 1u, 0u, packed[0],
                                          8u) != AEE_ENOMEMORY;
  fail |= hexkl_conv_block_prepare_decode(&g_tbl, vtcm, sizeof vtcm, 0u,
                                          packed[0], 3u) != AEE_EBADPARM;
  fail |= hexkl_conv_block_prepare_decode(&g_tbl, vtcm, sizeof vtcm, 0u,
                                          packed[1], 8u) != AEE_EBADPARM;
  fail |= g_tbl.slots[0].conv_decode_bytes != saved ||
          g_tbl.slots[0].conv_decode_chunk != saved_chunk;
  for (uint32_t i = 0; i < 4u; ++i) {
    fail |= hexkl_conv_block_prepare_decode(&g_tbl, NULL, 0u, i, NULL, 0u) !=
            AEE_SUCCESS;
    fail |= g_tbl.slots[i].conv_decode_bytes != NULL ||
            g_tbl.slots[i].conv_decode_chunk != 0u;
    free(packed_alloc[i]);
  }
  /* Edge: a wrong-shaped handle is refused before any work. */
  {
    int r = hexkl_conv_block_run(&g_tbl, vtcm, sizeof vtcm, sizeof vtcm, M, K,
                                 C, N_out, 0, 1, 3, 3, conv_w, x, got,
                                 got_state, NULL, &scratch, 1, 0u);
    printf("bad handle shape  : rc=%d (want %d)\n", r, AEE_EBADPARM);
    fail |= (r != AEE_EBADPARM);
  }
  /* One column batch per row block: accumulator parity must continue
     across row boundaries. The device fixture exercises the actual async
     reader; these stand-ins check its resulting row/column mapping. */
  {
    const uint32_t odd_m = 131, odd_c = 96, odd_n = 128;
    W oa, ob, oc, oo;
    make_weight(4, K, odd_c, &oa);
    make_weight(5, K, odd_c, &ob);
    make_weight(6, K, odd_c, &oc);
    make_weight(7, odd_c, odd_n, &oo);
    memset(got_state, 0, 2u * odd_c * sizeof(float));
    memset(want_state, 0, 2u * odd_c * sizeof(float));
    const int r = hexkl_conv_block_run(
      &g_tbl, vtcm, sizeof vtcm, sizeof vtcm, odd_m, K, odd_c, odd_n, 4, 5, 6,
      7, conv_w, x, got, got_state, NULL, &scratch, 1, 0u);
    ref_block(&oa, &ob, &oc, &oo, conv_w, x, odd_m, K, odd_c, odd_n, want,
              want_state);
    fail |= r != AEE_SUCCESS;
    fail |= compare("odd batches output", got, want, (size_t)odd_m * odd_n);
    fail |= compare("odd batches state", got_state, want_state, 2u * odd_c);
  }
  /* The model's shapes fit, with room: doc 51 section 2.2's arithmetic. */
  {
    hexkl_conv_block_layout R;
    int r = hexkl_conv_block_layout_get(2048, 2048, 2048, 8300u * 1024u, &R);
    printf("LFM2 shapes       : rc=%d total=%.2f MB tiles=%u\n", r,
           R.total / 1048576.0, R.acc_tiles);
    fail |= (r != 0);
    r = hexkl_conv_block_layout_get(2048, 2048, 2048, 4u << 20, &R);
    printf("4 MB arena        : rc=%d (want %d)\n", r, AEE_ENOMEMORY);
    fail |= (r != AEE_ENOMEMORY);
  }
  printf(fail ? "\nFAIL\n" : "\nALL CHECKS PASS\n");
  hexkl_moe_scratch_free(&scratch);
  return fail;
}
