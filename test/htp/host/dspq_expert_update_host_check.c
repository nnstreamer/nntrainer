// SPDX-License-Identifier: Apache-2.0
/** @file dspq_expert_update_host_check.c
 * @brief Real queue parser and update ordering; device tests cover arithmetic.
 * @author NNTrainer contributors
 * @bug See the repository issue tracker for known issues.
 */
#include <assert.h>
#include <malloc.h>
#include <remote.h>
#include <stdio.h>
#define AEESTDDEF_H
#include "../nntr_hvx_dspq.c"

static unsigned swaps, kernels, fail_swap;

int nntr_hvx_expert_update_batch(remote_handle64 handle, uint32_t K,
                                 uint32_t inter, uint32_t N,
                                 const htp_dspq_expert_update *updates,
                                 uint32_t count) {
  for (unsigned i = 0; i < count; ++i) {
    uint32_t gu, dn;
    const htp_dspq_expert_update *u = updates + i;
    const int rc =
      nntr_hvx_weight_swap_u2i4_arena(handle, u->h_gu, u->h_dn, K, inter, N,
                                      u->arena, u->off_gu, u->off_dn, &gu, &dn);
    if (rc)
      return rc;
  }
  return AEE_SUCCESS;
}

int nntr_hvx_weight_swap_u2i4_arena(remote_handle64 handle, uint32 old_gu,
                                    uint32 old_dn, uint32 K, uint32 inter,
                                    uint32 N, uint32 arena, uint32 off_gu,
                                    uint32 off_dn, uint32 *gu, uint32 *dn) {
  ++swaps;
  if (fail_swap == swaps)
    return AEE_EFAILED;
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  s->weights_u8i4.slots[old_gu].pal[0] = (int8_t)off_gu;
  s->weights_u8i4.slots[old_dn].pal[0] = (int8_t)off_dn;
  *gu = old_gu;
  *dn = old_dn;
  return AEE_SUCCESS;
}

int nntr_hvx_mm_u8i4_moe_layer(remote_handle64 handle, uint32 M, uint32 K,
                               uint32 inter, uint32 N, uint32 act,
                               const uint32 *gu, int gu_len, const uint32 *dn,
                               int dn_len, const uint32 *ri, int ri_len,
                               const uint32 *rc, int rc_len, const float *rw,
                               int rw_len, const float *in, int in_len,
                               float *out, int out_len) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  assert(swaps == 2 && gu_len == 2 && dn_len == 2 && M == 1);
  assert(s->weights_u8i4.slots[gu[0]].pal[0] == 1);
  assert(s->weights_u8i4.slots[dn[1]].pal[0] == 4);
  ++kernels;
  out[0] = 123.f;
  return AEE_SUCCESS;
}

int nntr_hvx_mm_u8i4_moe_layer_timed(
  remote_handle64 handle, uint32 M, uint32 K, uint32 inter, uint32 N,
  uint32 act, const uint32 *gu, int gu_len, const uint32 *dn, int dn_len,
  const uint32 *ri, int ri_len, const uint32 *rc, int rc_len, const float *rw,
  int rw_len, const float *in, int in_len, float *out, int out_len,
  uint32 *stages, int stages_len) {
  assert(stages_len == HTP_DSPQ_STAGES);
  stages[0] = 42;
  return nntr_hvx_mm_u8i4_moe_layer(handle, M, K, inter, N, act, gu, gu_len, dn,
                                    dn_len, ri, ri_len, rc, rc_len, rw, rw_len,
                                    in, in_len, out, out_len);
}

int main(void) {
  static nntr_hvx_session session;
  struct nntr_hvx_dspq q = {0};
  q.s = &session;
  for (unsigned h = 0; h < 4; ++h) {
    hexkl_weight_u8i4 *w = &session.weights_u8i4.slots[h];
    w->in_use = w->borrowed = 1;
    w->K = 32;
    w->N = h < 2 ? 64 : 32;
  }
  float in[32] = {0}, out[32] = {0};
  struct dspqueue_buffer bufs[2] = {0};
  bufs[0].ptr = in;
  bufs[0].size = sizeof(in);
  bufs[1].ptr = out;
  bufs[1].size = sizeof(out);
  dspq_msg msg = {0};
  msg.h = (htp_dspq_req_hdr){HTP_DSPQ_OP_MOE,
                             1,
                             HTP_DSPQ_FLAG_TIMED | HTP_DSPQ_FLAG_EXPERT_UPDATES,
                             1,
                             32,
                             32,
                             32,
                             2,
                             2};
  const unsigned o = sizeof(msg.h) / 4u;
  msg.u[o] = 0;
  msg.u[o + 1] = 1;
  msg.u[o + 2] = 2;
  msg.u[o + 3] = 3;
  msg.u[o + 4] = msg.u[o + 5] = 1;
  msg.f[o + 8] = msg.f[o + 9] = .5f;
  const unsigned base = htp_dspq_req_bytes(2, 2);
  msg.u[base / 4] = 2;
  htp_dspq_expert_update *u = (htp_dspq_expert_update *)&msg.u[base / 4 + 1];
  u[0] = (htp_dspq_expert_update){0, 2, 0, 1, 2};
  u[1] = (htp_dspq_expert_update){1, 3, 0, 3, 4};
  const unsigned len = base + 4 + 2 * sizeof(*u);
  assert(dspq_valid(&msg, len, 2, bufs));
  assert(!dspq_valid(&msg, sizeof(msg.h) - 1, 2, bufs));
  assert(!dspq_valid(&msg, base, 2, bufs));
  assert(!dspq_valid(&msg, len - 4, 2, bufs));
  assert(!dspq_valid(&msg, len, 1, bufs));
  msg.u[base / 4] = 9;
  assert(!dspq_valid(&msg, len, 2, bufs));
  msg.u[base / 4] = 0;
  assert(!dspq_valid(&msg, len, 2, bufs));
  msg.u[base / 4] = 2;
  msg.h.n_experts = UINT32_MAX;
  assert(!dspq_valid(&msg, len, 2, bufs));
  msg.h.n_experts = 2;
  htp_dspq_resp resp = {0};
  assert(dspq_run(&q, &msg, bufs, &resp) == AEE_SUCCESS);
  assert(kernels == 1 && out[0] == 123 && resp.stage_us[0] == 42);
  swaps = kernels = 0;
  msg.h.flags |= HTP_DSPQ_FLAG_EXPERT_BATCH_DMA;
  assert(dspq_valid(&msg, len, 2, bufs));
  assert(dspq_run(&q, &msg, bufs, &resp) == AEE_SUCCESS);
  assert(kernels == 1 && swaps == 2);
  msg.h.flags &= ~HTP_DSPQ_FLAG_EXPERT_UPDATES;
  assert(!dspq_valid(&msg, len, 2, bufs));
  msg.h.flags |= HTP_DSPQ_FLAG_EXPERT_UPDATES;
  msg.h.flags &= ~HTP_DSPQ_FLAG_EXPERT_BATCH_DMA;
  swaps = kernels = 0;
  out[0] = -99;
  u[1].h_gu = u[0].h_gu;
  assert(dspq_run(&q, &msg, bufs, &resp) == AEE_EBADPARM);
  assert(!swaps && !kernels && out[0] == -99);
  u[1].h_gu = 1;
  session.weights_u8i4.slots[1].N = 32;
  assert(dspq_run(&q, &msg, bufs, &resp) == AEE_EBADPARM);
  assert(!swaps && !kernels);
  session.weights_u8i4.slots[1].N = 64;
  fail_swap = 2;
  assert(dspq_run(&q, &msg, bufs, &resp) == AEE_EFAILED);
  assert(swaps == 2 && !kernels && out[0] == -99);
  puts("DSPQ EXPERT UPDATE: bounds, stable handles, ordering and failure PASS");
}
