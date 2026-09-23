/* Host harness for hexkl_mm_u8i4_moe_layer_run: scalar stand-ins for every
   primitive it calls, then the kernel against a straightforward reference.
   The point is the loop structure -- which rows each expert gets, which
   weights it uses, whether the buffer reuse clobbers anything, whether the
   scatter lands on the right output row with the right routing weight --
   not HMX's arithmetic, which the stubs define self-consistently for both
   sides. */
#include "hexkl_acc_tile.h"
#include "hexkl_dma_ring.h"
#include "hexkl_mm_u8i4_moe.h"
#include "hexkl_probe.h"
#include "hvx_dequant_i32.h"
#include "hvx_gather_ah_u8.h"
#include "hvx_gemm_u8i4_wh.h"
#include "hvx_scale_add_f32.h"
#include <AEEStdErr.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hvx_scalar_stubs.h"

int main(void) {
  const uint32_t M = 37, K = 64, inter = 32, N_out = 64, NE = 5;
  static uint8_t vtcm[8u << 20];

  hexkl_moe_layout L;
  int rc = hexkl_mm_u8i4_moe_layout(K, inter, N_out, sizeof vtcm, &L);
  printf(
    "layout rc=%d total=%u (act %u gu %u dn %u gate %u mid %u stage %u res "
    "%u)\n",
    rc, L.total, L.act_off, L.w_gu_off, L.w_dn_off, L.gate_off, L.mid_off,
    L.result_off, L.res_f32_off);
  if (rc)
    return 1;

  W wg[8], wd[8];
  for (uint32_t e = 0; e < NE; ++e) {
    make_weight(e, K, 2 * inter, &wg[e]);
    make_weight(NE + e, inter, N_out, &wd[e]);
  }
  uint32_t hg[8], hd[8];
  for (uint32_t e = 0; e < NE; ++e) {
    hg[e] = e;
    hd[e] = NE + e;
  }

  float *act = (float *)malloc(sizeof(float) * M * K);
  for (uint32_t i = 0; i < M * K; ++i)
    act[i] = rndf();

  /* Routing: expert 0 gets many rows (multi-block, a 6-row tail for the
     HVX path), expert 2 gets none, expert 3 a tail of exactly the
     threshold, expert 4 a second block too big for it (stays on the HMX);
     rows repeat across experts the way top-k does. */
  uint32_t rc_[8] = {70, 12, 0, 64 + HVX_GEMM_U8I4_MAX_ROWS,
                     64 + HVX_GEMM_U8I4_MAX_ROWS + 10};
  uint32_t n_rows = 0;
  for (uint32_t e = 0; e < NE; ++e)
    n_rows += rc_[e];
  uint32_t *ridx = (uint32_t *)malloc(sizeof(uint32_t) * n_rows);
  float *rw = (float *)malloc(sizeof(float) * n_rows);
  for (uint32_t i = 0; i < n_rows; ++i) {
    ridx[i] = rnd() % M;
    rw[i] = 0.1f + 0.9f * ((float)(rnd() % 100u) / 100.f);
  }

  float *got = (float *)malloc(sizeof(float) * M * N_out);
  /* One scratch across every call below, the way the session holds it: the
     later, smaller calls must work out of the block the first one grew. */
  hexkl_moe_scratch scratch = {NULL, NULL, 0};
  rc = hexkl_mm_u8i4_moe_layer_run(&g_tbl, vtcm, sizeof vtcm, sizeof vtcm, M, K,
                                   inter, N_out, NE, hg, hd, ridx, rc_, rw, act,
                                   got, NULL, &scratch);
  printf("run rc=%d  n_rows=%u\n", rc, n_rows);
  if (rc)
    return 1;

  /* reference */
  float *want = (float *)calloc(M * N_out, sizeof(float));
  uint8_t *aq = (uint8_t *)malloc(K);
  uint8_t *mq = (uint8_t *)malloc(inter);
  float *gu = (float *)malloc(sizeof(float) * 2 * inter);
  float *dn = (float *)malloc(sizeof(float) * N_out);
  float *mid = (float *)malloc(sizeof(float) * inter);
  uint32_t base = 0;
  for (uint32_t e = 0; e < NE; ++e) {
    for (uint32_t i = 0; i < rc_[e]; ++i) {
      uint32_t row = ridx[base + i];
      float as;
      int32_t az;
      quant_row(act + (size_t)row * K, K, aq, &as, &az);
      ref_mm(&wg[e], aq, as, az, gu);
      for (uint32_t j = 0; j < inter; ++j)
        mid[j] = gu[j] / (1.f + expf(-gu[j])) * gu[inter + j];
      float ms;
      int32_t mz;
      quant_row(mid, inter, mq, &ms, &mz);
      ref_mm(&wd[e], mq, ms, mz, dn);
      for (uint32_t c = 0; c < N_out; ++c) {
        /* Two operations through a volatile, matching what the kernel and
           the ARM path both do -- see hvx_scale_add_rows_f32's stub. */
        volatile float p = dn[c] * rw[base + i];
        want[(size_t)row * N_out + c] = want[(size_t)row * N_out + c] + p;
      }
    }
    base += rc_[e];
  }

  double worst = 0.0;
  uint32_t bad = 0;
  for (uint32_t i = 0; i < M * N_out; ++i) {
    double d = fabs((double)got[i] - (double)want[i]);
    double s = fabs((double)want[i]) + 1e-6;
    if (d / s > 1e-5) {
      ++bad;
    }
    if (d / s > worst)
      worst = d / s;
  }
  printf("mismatches=%u of %u   worst_rel=%g\n", bad, M * N_out, worst);
  printf(bad == 0 ? "MOE KERNEL MATCHES REFERENCE\n" : "MOE KERNEL DIFFERS\n");
  int fail = (bad != 0);

  /* Every active expert's gate_up and down must have been pushed. The DMA
     stub here completes immediately, so a missing push cannot show up as a
     wrong result the way a premature one does -- this counter is the only
     thing that catches it, and it is also what the device profile divides
     by to get GB/s, so a wrong count would quietly misreport the bandwidth
     the whole plan is gated on. */
  {
    uint32_t active = 0;
    for (uint32_t e = 0; e < NE; ++e) {
      if (rc_[e] != 0u)
        ++active;
    }
    const uint32_t gu_kb = ((K / 32u) * ((2u * inter) / 32u) * 512u) >> 10;
    const uint32_t dn_kb = ((inter / 32u) * (N_out / 32u) * 512u) >> 10;
    const uint64_t want = (uint64_t)active * (gu_kb + dn_kb);
    const uint64_t got_kb = hexkl_probe_us[HEXKL_PROBE_DMA_KB];
    printf("weight DMA        : %llu KB over %u experts (want %llu)\n",
           (unsigned long long)got_kb, active, (unsigned long long)want);
    if (got_kb != want)
      fail = 1;
    /* HMX blocks only: 70 -> 1 + a 6-row tail on the HVX, 12 -> 1,
       80 -> 1 + a 16-row tail, 90 -> 2 (its 26-row second block is over
       the tail threshold). 6 would mean a tail went to the HMX after all,
       4 that a full block was skipped. */
    printf("HMX blocks        : %llu (want 5: two tails on the HVX)\n",
           (unsigned long long)hexkl_probe_us[HEXKL_PROBE_BLOCKS]);
    if (hexkl_probe_us[HEXKL_PROBE_BLOCKS] != 5u)
      fail = 1;
  }

  /* QS2CX_WH: the same weights at two bits, expanded in VTCM after their
     DMA lands. The output must be BIT identical -- the expansion puts the
     int4 lattice back exactly, so every stage after the matmul sees what
     it saw before. A tolerance here would hide the one failure mode that
     matters, an expansion that is nearly right. This also exercises the
     in-place placement: the codes land in the top half of the slot they
     expand into, so a chunk boundary off by one clobbers a live tile and
     shows up as a wrong row. */
  {
    const int8_t pal[4] = {-6, -2, 1, 5};
    W wg2[8], wd2[8];
    uint32_t hg2[8], hd2[8];
    for (uint32_t e = 0; e < NE; ++e) {
      make_weight_pal(2u * NE + e, 4u * NE + e, K, 2 * inter, pal, &wg2[e]);
      make_weight_pal(3u * NE + e, 5u * NE + e, inter, N_out, pal, &wd2[e]);
      hg2[e] = 2u * NE + e;
      hd2[e] = 3u * NE + e;
    }
    float *four = (float *)malloc(sizeof(float) * M * N_out);
    float *two = (float *)malloc(sizeof(float) * M * N_out);
    hexkl_moe_scratch sc2 = {NULL, NULL, 0};
    const unsigned long long kb0 = hexkl_probe_us[HEXKL_PROBE_DMA_KB];
    int r4 = hexkl_mm_u8i4_moe_layer_run(&g_tbl, vtcm, sizeof vtcm, sizeof vtcm,
                                         M, K, inter, N_out, NE, hg2, hd2, ridx,
                                         rc_, rw, act, four, NULL, &sc2);
    const unsigned long long kb4 = hexkl_probe_us[HEXKL_PROBE_DMA_KB] - kb0;
    for (uint32_t e = 0; e < NE; ++e) {
      hg2[e] = 4u * NE + e;
      hd2[e] = 5u * NE + e;
    }
    int r2 = hexkl_mm_u8i4_moe_layer_run(&g_tbl, vtcm, sizeof vtcm, sizeof vtcm,
                                         M, K, inter, N_out, NE, hg2, hd2, ridx,
                                         rc_, rw, act, two, NULL, &sc2);
    const unsigned long long kb2 =
      hexkl_probe_us[HEXKL_PROBE_DMA_KB] - kb0 - kb4;
    uint32_t nb = 0;
    for (uint32_t i = 0; i < M * N_out; ++i) {
      if (memcmp(&four[i], &two[i], sizeof(float)) != 0)
        ++nb;
    }
    printf("2-bit experts     : rc=%d/%d  bitwise mismatches=%u of %u\n", r4,
           r2, nb, M * N_out);
    /* The point of the whole format: half the bytes cross the bus. Asserted
       rather than inspected, because a push that quietly kept the full
       width would still give the right answer. The probe counts whole
       kilobytes and this shape's chunks are smaller than that, so the
       2-bit side reads 0 -- what the bound catches is the regression that
       matters, a 2-bit weight pushed at full width (which would read the
       same as the 4-bit side, not half of it). */
    printf("2-bit weight DMA  : %llu KB vs %llu (want <= half)\n", kb2, kb4);
    fail |= (r4 != 0) || (r2 != 0) || (nb != 0) || (kb4 == 0ull) ||
            (kb2 * 2ull > kb4);
    free(four);
    free(two);
  }

  /* --- edge cases the routing can actually produce --------------------- */
  {
    uint32_t z[8] = {0, 0, 0, 0, 0};
    memset(got, 0xA5, sizeof(float) * M * N_out);
    int r = hexkl_mm_u8i4_moe_layer_run(&g_tbl, vtcm, sizeof vtcm, sizeof vtcm,
                                        M, K, inter, N_out, NE, hg, hd, ridx, z,
                                        rw, act, got, NULL, &scratch);
    int ok = (r == 0);
    for (uint32_t i = 0; i < M * N_out; ++i) {
      if (got[i] != 0.f) {
        ok = 0;
        break;
      }
    }
    printf("all-empty routing : %s\n", ok ? "zeroed, rc=0" : "WRONG");
    fail |= !ok;
  }
  {
    uint32_t c64[8] = {64, 0, 0, 0, 0};
    int r = hexkl_mm_u8i4_moe_layer_run(&g_tbl, vtcm, sizeof vtcm, sizeof vtcm,
                                        M, K, inter, N_out, NE, hg, hd, ridx,
                                        c64, rw, act, got, NULL, &scratch);
    printf("exactly 64 rows   : rc=%d\n", r);
    fail |= (r != 0);
  }
  {
    uint32_t c1[8] = {1, 0, 0, 0, 0};
    uint32_t bad_row[1] = {M};
    int r = hexkl_mm_u8i4_moe_layer_run(&g_tbl, vtcm, sizeof vtcm, sizeof vtcm,
                                        M, K, inter, N_out, NE, hg, hd, bad_row,
                                        c1, rw, act, got, NULL, &scratch);
    printf("row_index >= M    : rc=%d (want %d)\n", r, AEE_EBADPARM);
    fail |= (r != AEE_EBADPARM);
  }
  {
    hexkl_moe_layout R;
    int r = hexkl_mm_u8i4_moe_layout(2048, 1792, 2048, 8300u * 1024u, &R);
    printf("LFM2 shapes       : rc=%d total=%.2f MB (doc 47 section 11 says "
           "6.92)\n",
           r, R.total / 1048576.0);
    fail |= (r != 0);
    r = hexkl_mm_u8i4_moe_layout(2048, 1792, 2048, 4u << 20, &R);
    printf("4 MB arena        : rc=%d (want %d)\n", r, AEE_ENOMEMORY);
    fail |= (r != AEE_ENOMEMORY);
  }
  printf(fail ? "\nFAIL\n" : "\nALL CHECKS PASS\n");
  hexkl_moe_scratch_free(&scratch);
  return fail;
}
