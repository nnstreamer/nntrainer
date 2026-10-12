// SPDX-License-Identifier: Apache-2.0
/**
 * @file   swap_host_check.c
 * @brief  weight_swap_u8i4_arena (doc 52 section 10.12) on the host: the
 *         real entry point in nntr_hvx_mm_u8i4.c over the real weight
 *         registry (hexkl_mm_u8i4_dma.c), with an arena that is plain
 *         aligned memory set straight into the session instead of an
 *         HAP_mmap'd ION buffer. What it checks is the bookkeeping the host
 *         relies on: the new pair registered in place with the given
 *         scales and column sums and a zero bias, the old pair released
 *         only after, and on every error nothing released and nothing left
 *         registered.
 * @author NNTrainer contributors
 * @bug See the repository issue tracker for known issues.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nntr_hvx.h"
#include "nntr_hvx_expert_update.h"
#include "nntr_hvx_session.h"

/** Referenced by nntr_hvx_mm_u8i4.c's other entry points, none of which
   this check calls. Declared without their headers on purpose: C links
   them by name, and reaching one is a bug in the check. */
#define NOT_HERE(fn)                                                           \
  void fn(void) {                                                              \
    fprintf(stderr, "swap_host_check: %s called\n", #fn);                      \
    abort();                                                                   \
  }
NOT_HERE(hexkl_conv_block_run)
NOT_HERE(hexkl_conv_block_prepare_decode)
NOT_HERE(hexkl_mm_u8i4_bake_weights)
NOT_HERE(hexkl_mm_u8i4_plan)
NOT_HERE(hexkl_mm_u8i4_run)
NOT_HERE(hvx_rmsnorm_rows_ld_f32)
NOT_HERE(hvx_rope_rows_ld_f32)
NOT_HERE(hvx_rmsnorm_add_blocks_f32)
/** These two are declared by headers the session struct pulls in, so they
   get their real signatures. */
int hexkl_mm_u8i8_layer_run(hexkl_weight_u8i8_table *tbl, uint8_t *vtcm_base,
                            uint32_t vtcm_size, uint32_t config_off, uint32_t M,
                            uint32_t K, const uint32_t *handles,
                            uint32_t n_handles, const float *act_f32,
                            float *out_cat, const hexkl_mm_opts *opts) {
  fprintf(stderr, "swap_host_check: hexkl_mm_u8i8_layer_run called\n");
  abort();
}

int hexkl_mm_u8i4_moe_layer_run(
  hexkl_weight_u8i4_table *tbl, uint8_t *vtcm_base, uint32_t vtcm_size,
  uint32_t config_off, uint32_t M, uint32_t K, uint32_t inter, uint32_t N_out,
  uint32_t n_experts, const uint32_t *h_gate_up, const uint32_t *h_down,
  const uint32_t *row_index, const uint32_t *row_count, const float *row_weight,
  const float *act_f32, float *out_f32, hvx_worker_pool *pool,
  hexkl_moe_scratch *scratch, uint32_t flags) {
  fprintf(stderr, "swap_host_check: hexkl_mm_u8i4_moe_layer_run called\n");
  abort();
}
/* FC execution is outside this registry-only harness. */
int hexkl_moe_scratch_reserve(hexkl_moe_scratch *scratch, size_t bytes) {
  fprintf(stderr, "swap_host_check: hexkl_moe_scratch_reserve called\n");
  abort();
}

/** Norm-consumer entries are linked with the registry but are never called
   by this harness. Keep an accidental call a test failure, not a no-op. */
int nntr_hvx_inference_math_f32(remote_handle64 handle, uint32 op, uint32 M,
                                uint32 N, uint32 chunk, float eps, float scale,
                                const float *x, int xLen, const float *aux,
                                int auxLen, const float *param, int paramLen,
                                uint8 *result, int resultLen) {
  fprintf(stderr, "swap_host_check: nntr_hvx_inference_math_f32 called\n");
  abort();
}

/* The fused router shares this translation unit with the registry. */
int nntr_hvx_sigmoid_topk_f32(remote_handle64 handle, uint32 M, uint32 E,
                              uint32 top_k, uint32 n_sel, const float *logits,
                              int logitsLen, const float *bias, int biasLen,
                              uint32 *sel, int selLen, float *weight,
                              int weightLen) {
  fprintf(stderr, "swap_host_check: nntr_hvx_sigmoid_topk_f32 called\n");
  abort();
}

/** The skel asks these two before a release or a detach; this check has no
   decode graph and no Q4M1 slot, so nothing is ever bound or borrowed. */
int hexkl_graph_uses_handle(const hexkl_graph *g, uint32_t handle) {
  (void)g;
  (void)handle;
  return 0;
}
int nntr_hvx_q4m1_borrows(const nntr_hvx_session *s, const uint8_t *va,
                          uint32_t bytes) {
  (void)s;
  (void)va;
  (void)bytes;
  return 0;
}
void hexkl_probe_reset(int enable) {
  (void)enable;
  fprintf(stderr, "swap_host_check: hexkl_probe_reset called\n");
  abort();
}

#define NONE 0xFFFFFFFFu
#define CHECK(cond)                                                            \
  do {                                                                         \
    if (!(cond)) {                                                             \
      fprintf(stderr, "swap_host_check FAILED %s:%d: %s\n", __FILE__,          \
              __LINE__, #cond);                                                \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

static nntr_hvx_session g_s; /* big: the weight tables live inline */

static unsigned live(void) {
  unsigned n = 0;
  for (unsigned i = 0; i < HEXKL_MM_U8I4_MAX_WEIGHTS; ++i)
    n += g_s.weights_u8i4.slots[i].in_use ? 1u : 0u;
  return n;
}

enum { K = 64, INTER = 32, NOUT = 64, NGU = 2 * INTER };
enum { GU_BYTES = (K / 32) * (NGU / 32) * 512 };      /* 2048 */
enum { DN_BYTES = (INTER / 32) * (NOUT / 32) * 512 }; /* 1024 */
enum { ARENA = 4 * 4096 };

static float gu_s[NGU], dn_s[NOUT];
static int32_t gu_c[NGU], dn_c[NOUT];

static int swap(uint32_t og, uint32_t od, uint32_t off_gu, uint32_t off_dn,
                uint32_t *g, uint32_t *d) {
  return nntr_hvx_weight_swap_u8i4_arena(
    (remote_handle64)(uintptr_t)&g_s, og, od, K, INTER, NOUT, 0, off_gu, off_dn,
    gu_s, NGU, gu_c, NGU, dn_s, NOUT, dn_c, NOUT, g, d);
}

int main(void) {
  static uint8_t va[ARENA] __attribute__((aligned(4096)));
  g_s.vtcm_size = 8u << 20;
  g_s.arenas[0].fd = 3;
  g_s.arenas[0].va = va;
  g_s.arenas[0].bytes = ARENA;
  for (int i = 0; i < NGU; ++i) {
    gu_s[i] = 0.5f + (float)i;
    gu_c[i] = i - 7;
  }
  for (int i = 0; i < NOUT; ++i) {
    dn_s[i] = 2.0f + (float)i;
    dn_c[i] = 3 - i;
  }

  /* 1. Nothing to release: a plain two-weight registration in place. */
  uint32_t g = NONE, d = NONE;
  CHECK(swap(NONE, NONE, 0, 4096, &g, &d) == AEE_SUCCESS);
  CHECK(live() == 2);
  const hexkl_weight_u8i4 *wg = &g_s.weights_u8i4.slots[g];
  const hexkl_weight_u8i4 *wd = &g_s.weights_u8i4.slots[d];
  CHECK(wg->borrowed && wg->wh_bytes == va && wg->K == K && wg->N == NGU);
  CHECK(wd->borrowed && wd->wh_bytes == va + 4096 && wd->K == INTER &&
        wd->N == NOUT);
  CHECK(memcmp(wg->w_scale, gu_s, sizeof(gu_s)) == 0);
  CHECK(memcmp(wg->colsum_w, gu_c, sizeof(gu_c)) == 0);
  CHECK(memcmp(wd->w_scale, dn_s, sizeof(dn_s)) == 0);
  CHECK(memcmp(wd->colsum_w, dn_c, sizeof(dn_c)) == 0);
  for (int i = 0; i < NGU; ++i)
    CHECK(wg->bias[i] == 0.0f);

  /** A second arena is borrowed by the optional Conv decode image even
     when it contains none of the original WH. Rebind must invalidate it. */
  static uint8_t decode[GU_BYTES] __attribute__((aligned(512)));
  g_s.weights_u8i4.slots[g].conv_decode_bytes = decode;
  g_s.weights_u8i4.slots[g].conv_decode_chunk = 8u;
  CHECK(hexkl_weight_u8i4_borrows(&g_s.weights_u8i4, decode, sizeof decode));

  /** 2. The swap proper: new pair in, old pair out, in one call. Same shape,
        so (doc 52 section 10.25) the handles are rebound in place: same
        numbers, same arrays, new bytes and scales, bias still zero. */
  const float *arr_g = g_s.weights_u8i4.slots[g].w_scale;
  for (int i = 0; i < NGU; ++i)
    gu_s[i] = 100.0f + (float)i;
  uint32_t g2 = NONE, d2 = NONE;
  CHECK(swap(g, d, 8192, 12288, &g2, &d2) == AEE_SUCCESS);
  CHECK(live() == 2);
  CHECK(g2 == g && d2 == d);
  CHECK(g_s.weights_u8i4.slots[g2].w_scale == arr_g); /* not reallocated */
  CHECK(memcmp(g_s.weights_u8i4.slots[g2].w_scale, gu_s, sizeof(gu_s)) == 0);
  for (int i = 0; i < NGU; ++i)
    CHECK(g_s.weights_u8i4.slots[g2].bias[i] == 0.0f);
  CHECK(g_s.weights_u8i4.slots[g2].wh_bytes == va + 8192);
  CHECK(g_s.weights_u8i4.slots[d2].wh_bytes == va + 12288);
  CHECK(g_s.weights_u8i4.slots[g2].conv_decode_bytes == NULL);
  CHECK(g_s.weights_u8i4.slots[g2].conv_decode_chunk == 0u);
  CHECK(!hexkl_weight_u8i4_borrows(&g_s.weights_u8i4, decode, sizeof decode));

  /* 3. Refusals before anything is registered or released. */
  uint32_t x = 1234, y = 1234;
  CHECK(swap(g2, g2, 0, 4096, &x, &y) == AEE_EBADPARM);   /* same twice */
  CHECK(swap(g2, NONE, 0, 4096, &x, &y) == AEE_EBADPARM); /* half a pair */
  CHECK(swap(g2, 2000, 0, 4096, &x, &y) == AEE_EBADPARM); /* not live */
  CHECK(nntr_hvx_weight_swap_u8i4_arena(
          (remote_handle64)(uintptr_t)&g_s, g2, d2, K, INTER, NOUT, 0, 0, 4096,
          gu_s, NGU - 1, gu_c, NGU, dn_s, NOUT, dn_c, NOUT, &x,
          &y) == AEE_EBADPARM); /* wrong length */
  CHECK(live() == 2 && x == 1234 && y == 1234);
  CHECK(g_s.weights_u8i4.slots[g2].in_use && g_s.weights_u8i4.slots[d2].in_use);

  /** 4. gate_up fits, down does not (past the arena): refused before
        either handle is rebound, so the old pair is live and unchanged. */
  CHECK(swap(g2, d2, 0, ARENA - 512, &x, &y) != AEE_SUCCESS);
  CHECK(live() == 2);
  CHECK(g_s.weights_u8i4.slots[g2].in_use && g_s.weights_u8i4.slots[d2].in_use);
  CHECK(g_s.weights_u8i4.slots[g2].wh_bytes == va + 8192);
  CHECK(g_s.weights_u8i4.slots[d2].wh_bytes == va + 12288);

  /* 5. Unaligned down: refused, nothing changes -- not even gate_up. */
  CHECK(swap(g2, d2, 0, 4096 + 100, &x, &y) != AEE_SUCCESS);
  CHECK(live() == 2);
  CHECK(g_s.weights_u8i4.slots[g2].wh_bytes == va + 8192);

  /** 5b. A retired pair of another shape falls back to register + release:
         new handles, the old ones freed. */
  {
    enum { I2 = 64, NGU2 = 2 * I2 };
    static float s2[NGU2];
    static int32_t c2[NGU2];
    uint32_t og = NONE, od = NONE, ng = NONE, nd = NONE;
    CHECK(nntr_hvx_weight_swap_u8i4_arena((remote_handle64)(uintptr_t)&g_s,
                                          NONE, NONE, K, I2, NOUT, 0, 0, 8192,
                                          s2, NGU2, c2, NGU2, dn_s, NOUT, dn_c,
                                          NOUT, &og, &od) == AEE_SUCCESS);
    CHECK(live() == 4);
    CHECK(swap(og, od, 0, 4096, &ng, &nd) == AEE_SUCCESS);
    CHECK(live() == 4);
    CHECK(g_s.weights_u8i4.slots[ng].K == K &&
          g_s.weights_u8i4.slots[ng].N == NGU);
    CHECK(g_s.weights_u8i4.slots[ng].wh_bytes == va);
    /* back to two: release the fallback's pair */
    hexkl_weight_u8i4_release(&g_s.weights_u8i4, ng);
    hexkl_weight_u8i4_release(&g_s.weights_u8i4, nd);
    CHECK(live() == 2);
  }

  /** 6. The batch (doc 52 section 10.23): expert 0 swaps out the live pair
        into 0/4096, expert 1 is a fresh pair past the arena. Expert 0 must
        land, expert 1 must not, and the call itself succeeds so the host
        hears which. */
  {
    static float bgs[2 * NGU], bds[2 * NOUT];
    static int32_t bgc[2 * NGU], bdc[2 * NOUT];
    for (int e = 0; e < 2; ++e) {
      memcpy(bgs + e * NGU, gu_s, sizeof(gu_s));
      memcpy(bgc + e * NGU, gu_c, sizeof(gu_c));
      memcpy(bds + e * NOUT, dn_s, sizeof(dn_s));
      memcpy(bdc + e * NOUT, dn_c, sizeof(dn_c));
    }
    uint32_t og[2] = {g2, NONE}, od[2] = {d2, NONE}, ar[2] = {0, 0};
    uint32_t ofg[2] = {0, ARENA - 512}, ofd[2] = {4096, 8192};
    uint32_t hg[2] = {NONE, NONE}, hd[2] = {NONE, NONE}, done = 99;
    int32_t err = 0;
    CHECK(nntr_hvx_weight_swap_batch_u8i4_arena(
            (remote_handle64)(uintptr_t)&g_s, K, INTER, NOUT, og, 2, od, 2, ar,
            2, ofg, 2, ofd, 2, bgs, 2 * NGU, bgc, 2 * NGU, bds, 2 * NOUT, bdc,
            2 * NOUT, hg, 2, hd, 2, &done, &err) == AEE_SUCCESS);
    CHECK(done == 1 && err != AEE_SUCCESS);
    CHECK(live() == 2); /* old pair out, expert 0 in, expert 1 absent */
    CHECK(g_s.weights_u8i4.slots[hg[0]].wh_bytes == va);
    CHECK(g_s.weights_u8i4.slots[hd[0]].wh_bytes == va + 4096);
    /* Lengths that disagree: refused whole, before anything moves. */
    CHECK(nntr_hvx_weight_swap_batch_u8i4_arena(
            (remote_handle64)(uintptr_t)&g_s, K, INTER, NOUT, og, 2, od, 1, ar,
            2, ofg, 2, ofd, 2, bgs, 2 * NGU, bgc, 2 * NGU, bds, 2 * NOUT, bdc,
            2 * NOUT, hg, 2, hd, 2, &done, &err) == AEE_EBADPARM);
    CHECK(done == 0 && live() == 2);
  }

  /** 7. [doc 52 section 10.30] Empty sequences: the scales and column sums
        come from the arena, right after each weight's WH bytes -- N f32
        then N i32 -- and the bias is zero. Rebind in place first, then
        the register path (a retired pair of another shape), then the
        extent check that now covers the tail, then the batch. */
  {
    const uint32_t og0 = NONE, od0 = NONE;
    uint32_t g3 = NONE, d3 = NONE, x2 = 77, y2 = 77;
    float *gt = (float *)(va + 8192 + GU_BYTES);  /* gate_up's tail */
    float *dt = (float *)(va + 12288 + DN_BYTES); /* down's tail */
    for (int i = 0; i < NGU; ++i) {
      gt[i] = 7.5f - (float)i;
      ((int32_t *)gt)[NGU + i] = 1000 + i;
    }
    for (int i = 0; i < NOUT; ++i) {
      dt[i] = -1.0f * (float)i;
      ((int32_t *)dt)[NOUT + i] = -i;
    }
    /* the live pair from 6 has this shape: rebound in place */
    uint32_t lg = NONE, ld = NONE;
    for (unsigned i = 0; i < HEXKL_MM_U8I4_MAX_WEIGHTS; ++i) {
      const hexkl_weight_u8i4 *w = &g_s.weights_u8i4.slots[i];
      if (w->in_use && w->K == K && w->N == NGU)
        lg = i;
      if (w->in_use && w->K == INTER && w->N == NOUT)
        ld = i;
    }
    CHECK(lg != NONE && ld != NONE);
    const float *arr_lg = g_s.weights_u8i4.slots[lg].w_scale;
    CHECK(nntr_hvx_weight_swap_u8i4_arena((remote_handle64)(uintptr_t)&g_s, lg,
                                          ld, K, INTER, NOUT, 0, 8192, 12288,
                                          NULL, 0, NULL, 0, NULL, 0, NULL, 0,
                                          &g3, &d3) == AEE_SUCCESS);
    CHECK(g3 == lg && d3 == ld && live() == 2);
    CHECK(g_s.weights_u8i4.slots[g3].w_scale == arr_lg);
    CHECK(memcmp(g_s.weights_u8i4.slots[g3].w_scale, gt, sizeof(float) * NGU) ==
          0);
    CHECK(memcmp(g_s.weights_u8i4.slots[g3].colsum_w, gt + NGU,
                 sizeof(int32_t) * NGU) == 0);
    CHECK(memcmp(g_s.weights_u8i4.slots[d3].w_scale, dt,
                 sizeof(float) * NOUT) == 0);
    CHECK(memcmp(g_s.weights_u8i4.slots[d3].colsum_w, dt + NOUT,
                 sizeof(int32_t) * NOUT) == 0);
    for (int i = 0; i < NGU; ++i)
      CHECK(g_s.weights_u8i4.slots[g3].bias[i] == 0.0f);
    /* register path: nothing to release, same arena bytes */
    CHECK(nntr_hvx_weight_swap_u8i4_arena((remote_handle64)(uintptr_t)&g_s, og0,
                                          od0, K, INTER, NOUT, 0, 8192, 12288,
                                          NULL, 0, NULL, 0, NULL, 0, NULL, 0,
                                          &x2, &y2) == AEE_SUCCESS);
    CHECK(live() == 4 && x2 != g3 && y2 != d3);
    CHECK(memcmp(g_s.weights_u8i4.slots[x2].colsum_w, gt + NGU,
                 sizeof(int32_t) * NGU) == 0);
    for (int i = 0; i < NOUT; ++i)
      CHECK(g_s.weights_u8i4.slots[y2].bias[i] == 0.0f);
    hexkl_weight_u8i4_release(&g_s.weights_u8i4, x2);
    hexkl_weight_u8i4_release(&g_s.weights_u8i4, y2);
    CHECK(live() == 2);
    /** down's WH bytes fit the arena but its tail does not: refused, and
       gate_up (checked first) is untouched */
    CHECK(nntr_hvx_weight_swap_u8i4_arena(
            (remote_handle64)(uintptr_t)&g_s, g3, d3, K, INTER, NOUT, 0, 0,
            ARENA - DN_BYTES, NULL, 0, NULL, 0, NULL, 0, NULL, 0, &x2,
            &y2) == AEE_EBADPARM);
    CHECK(g_s.weights_u8i4.slots[g3].wh_bytes == va + 8192 && live() == 2);
    /* half empty is a length mismatch, not the tail protocol */
    CHECK(nntr_hvx_weight_swap_u8i4_arena((remote_handle64)(uintptr_t)&g_s, g3,
                                          d3, K, INTER, NOUT, 0, 8192, 12288,
                                          NULL, 0, NULL, 0, dn_s, NOUT, dn_c,
                                          NOUT, &x2, &y2) == AEE_EBADPARM);
    /* the batch, one expert, empty flat sequences */
    {
      uint32_t bog[1] = {g3}, bod[1] = {d3}, bar[1] = {0}, bofg[1] = {8192},
               bofd[1] = {12288}, bhg[1] = {NONE}, bhd[1] = {NONE}, done = 9;
      int32_t err = 1;
      gt[0] = 42.0f;
      CHECK(nntr_hvx_weight_swap_batch_u8i4_arena(
              (remote_handle64)(uintptr_t)&g_s, K, INTER, NOUT, bog, 1, bod, 1,
              bar, 1, bofg, 1, bofd, 1, NULL, 0, NULL, 0, NULL, 0, NULL, 0, bhg,
              1, bhd, 1, &done, &err) == AEE_SUCCESS);
      CHECK(done == 1 && err == AEE_SUCCESS && bhg[0] == g3 && bhd[0] == d3);
      CHECK(g_s.weights_u8i4.slots[g3].w_scale[0] == 42.0f);
    }
  }

  /* Compact tails, palette validation, bounds and int4<->int2 rebind. */
  {
    const uint32_t go = 0, dn = 4096;
    const uint32_t gcodes = K * NGU / 4u, dcodes = INTER * NOUT / 4u;
    const int8_t pal[4] = {-8, -1, 2, 7};
    memcpy(va + go + gcodes, gu_s, sizeof gu_s);
    memcpy(va + go + gcodes + sizeof gu_s, gu_c, sizeof gu_c);
    memcpy(va + go + gcodes + 8u * NGU, pal, 4);
    memcpy(va + dn + dcodes, dn_s, sizeof dn_s);
    memcpy(va + dn + dcodes + sizeof dn_s, dn_c, sizeof dn_c);
    memcpy(va + dn + dcodes + 8u * NOUT, pal, 4);
    uint32_t cg, cd, ng, nd;
    CHECK(nntr_hvx_weight_swap_u2i4_arena((remote_handle64)(uintptr_t)&g_s,
                                          NONE, NONE, K, INTER, NOUT, 0, go, dn,
                                          &cg, &cd) == AEE_SUCCESS);
    CHECK(g_s.weights_u8i4.slots[cg].bits == 2u);
    CHECK(memcmp(g_s.weights_u8i4.slots[cg].w_scale, gu_s, sizeof gu_s) == 0);
    CHECK(memcmp(g_s.weights_u8i4.slots[cd].colsum_w, dn_c, sizeof dn_c) == 0);
    const uint8_t *old = g_s.weights_u8i4.slots[cg].wh_bytes;
    va[dn + dcodes + 8u * NOUT] = 9;
    CHECK(nntr_hvx_weight_swap_u2i4_arena((remote_handle64)(uintptr_t)&g_s, cg,
                                          cd, K, INTER, NOUT, 0, go, dn, &ng,
                                          &nd) == AEE_EBADPARM);
    CHECK(g_s.weights_u8i4.slots[cg].wh_bytes == old);
    va[dn + dcodes + 8u * NOUT] = (uint8_t)pal[0];
    CHECK(nntr_hvx_weight_swap_u2i4_arena(
            (remote_handle64)(uintptr_t)&g_s, cg, cd, K, INTER, NOUT, 0, go,
            ARENA - 512u, &ng, &nd) == AEE_EBADPARM);
    CHECK(swap(cg, cd, 8192, 12288, &ng, &nd) == AEE_SUCCESS);
    CHECK(ng == cg && nd == cd && g_s.weights_u8i4.slots[cg].bits == 4u);
    CHECK(nntr_hvx_weight_swap_u2i4_arena((remote_handle64)(uintptr_t)&g_s, cg,
                                          cd, K, INTER, NOUT, 0, go, dn, &ng,
                                          &nd) == AEE_SUCCESS);
    CHECK(ng == cg && nd == cd && g_s.weights_u8i4.slots[cg].bits == 2u);
    const htp_dspq_expert_update update = {cg, cd, 0, go, dn};
    for (unsigned round = 0; round < 32; ++round) {
      const float previous = g_s.weights_u8i4.slots[cg].w_scale[0];
      float *gs = (float *)(va + go + gcodes);
      float *ds = (float *)(va + dn + dcodes);
      gs[0] = 1000.f + round;
      ds[0] = 2000.f + round;
      ((int32_t *)(gs + NGU))[0] = (int32_t)round - 50;
      va[go + gcodes + 8u * NGU] = (uint8_t)(round % 8u);
      va[dn + dcodes + 8u * NOUT] = 9; // Validate the whole batch first.
      CHECK(nntr_hvx_expert_update_batch((remote_handle64)(uintptr_t)&g_s, K,
                                         INTER, NOUT, &update,
                                         1) == AEE_EBADPARM);
      CHECK(g_s.weights_u8i4.slots[cg].w_scale[0] == previous);
      va[dn + dcodes + 8u * NOUT] = 3;
      CHECK(nntr_hvx_expert_update_batch((remote_handle64)(uintptr_t)&g_s, K,
                                         INTER, NOUT, &update,
                                         1) == AEE_SUCCESS);
      CHECK(g_s.weights_u8i4.slots[cg].w_scale[0] == 1000.f + round);
      CHECK(g_s.weights_u8i4.slots[cd].w_scale[0] == 2000.f + round);
      CHECK(g_s.weights_u8i4.slots[cg].colsum_w[0] == (int32_t)round - 50);
      CHECK(g_s.weights_u8i4.slots[cg].pal[0] == (int8_t)(round % 8u));
    }
    hexkl_weight_u8i4_release(&g_s.weights_u8i4, cg);
    hexkl_weight_u8i4_release(&g_s.weights_u8i4, cd);
  }

  printf("WEIGHT SWAP OK\n");
  return 0;
}
