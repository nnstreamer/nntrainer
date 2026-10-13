// SPDX-License-Identifier: Apache-2.0
/**
 * @file dspq_conv_cache_host_check.c
 * @brief Exercise the real queue cache parser with recording Conv stand-ins.
 * Checks packing, invalid packets, slot isolation, rebind and failed calls.
 * Arithmetic and DMA coherence are covered by the device dispatch test.
 * Build with generated host nntr_hvx.h, host/stub, SDK dspqueue.h, function
 * sections and --gc-sections (unrelated queue entry points are not linked).
 * @author NNTrainer contributors
 * @bug See the repository issue tracker for known issues.
 */
#include <assert.h>
#include <malloc.h>
#include <stdio.h>

// remote.h's host stand-in already defines the SDK's fixed-width types.
#include <remote.h>
#define AEESTDDEF_H
#include "../nntr_hvx_dspq.c"

static int fail_kernel, calls;

int nntr_hvx_inference_math_f32(remote_handle64 hs, uint32 op, uint32 M,
                                uint32 N, uint32 chunk, float eps, float scale,
                                const float *x, int x_len, const float *aux,
                                int aux_len, const float *gamma, int gamma_len,
                                uint8 *out, int out_len) {
  assert((op == 0 || op == 7) && M == 1 && x_len == N && gamma_len == N);
  float *y = (float *)out;
  for (uint32 n = 0; n < N; ++n) {
    const float sum = x[n] + (op == 7 ? aux[n] : 0.f);
    if (op == 7)
      y[n] = sum;
    y[(op == 7 ? N : 0) + n] = sum * gamma[n];
  }
  return 0;
}

int nntr_hvx_mm_u8i4_conv_block(remote_handle64 hs, uint32 M, uint32 K,
                                uint32 C, uint32 N, const uint32 *handles,
                                int handles_len, uint32 out_handle,
                                const float *conv, int conv_len,
                                const float *input, int input_len,
                                float *output, int output_len, float *state,
                                int state_len) {
  ++calls;
  if (fail_kernel)
    return AEE_EFAILED;
  assert(M == 1 && K == C && C == N && conv_len == 3 * C && input_len == K);
  for (uint32 n = 0; n < N; ++n)
    output[n] = input[n] + conv[n] + state[n];
  for (uint32 n = 0; n < 2 * C; ++n)
    state[n] += 1.f;
  return 0;
}

int nntr_hvx_mm_u8i4_conv_block_timed(
  remote_handle64 hs, uint32 M, uint32 K, uint32 C, uint32 N,
  const uint32 *handles, int handles_len, uint32 out_handle, const float *conv,
  int conv_len, const float *input, int input_len, float *output,
  int output_len, float *state, int state_len, uint32 *stages, int stages_len) {
  assert(stages_len == HTP_DSPQ_STAGES);
  stages[0] = 123;
  return nntr_hvx_mm_u8i4_conv_block(
    hs, M, K, C, N, handles, handles_len, out_handle, conv, conv_len, input,
    input_len, output, output_len, state, state_len);
}

int nntr_hvx_mm_u8i4_conv_block_norm(
  remote_handle64 hs, uint32 K, uint32 C, uint32 N, const uint32 *handles,
  int handles_len, uint32 out_handle, const float *conv, int conv_len,
  uint32 add, float eps, const float *input, int input_len, float *norm,
  int norm_len, float *output, int output_len, float *state, int state_len,
  uint32 *stages, int stages_len) {
  assert(input_len == (add ? 3 : 2) * K && norm_len == 2 * K);
  for (uint32 k = 0; k < K; ++k) {
    norm[k] = input[k] + (add ? input[K + k] : 0.f);
    norm[K + k] = norm[k] * input[(add ? 2 : 1) * K + k];
  }
  return nntr_hvx_mm_u8i4_conv_block(hs, 1, K, C, N, handles, handles_len,
                                     out_handle, conv, conv_len, norm + K, K,
                                     output, output_len, state, state_len);
}

static float input[8 * 64], output[5 * 64];
static struct dspqueue_buffer buffers[2];
static htp_dspq_resp response;

static void fill(float **dst, unsigned n, float value) {
  while (n--)
    *(*dst)++ = value;
}

static void pack(const htp_dspq_conv_cached_req *r, float x, float w, float g,
                 float s) {
  float *p = input;
  fill(&p, r->conv.K, x);
  if (r->conv.norm_kind == 2)
    fill(&p, r->conv.K, 4.f);
  if (r->updates & HTP_DSPQ_CONV_UPDATE_W)
    fill(&p, 3 * r->conv.C, w);
  if (r->updates & HTP_DSPQ_CONV_UPDATE_GAMMA)
    fill(&p, r->conv.K, g);
  if (r->updates & HTP_DSPQ_CONV_UPDATE_STATE)
    fill(&p, 2 * r->conv.C, s);
  buffers[0].ptr = input;
  buffers[1].ptr = output;
  buffers[0].size = (p - input) * sizeof(float);
  buffers[1].size = htp_dspq_conv_output_bytes(&r->conv);
}

static int run(struct nntr_hvx_dspq *d, htp_dspq_conv_cached_req *r) {
  return dspq_conv_cached(d, (const dspq_msg *)r, sizeof(*r), 2, buffers,
                          &response);
}

int main(void) {
  nntr_hvx_session session = {0};
  session.moe_flags = HTP_CONV_FLAG_M1_HVX;
  struct nntr_hvx_dspq d = {0};
  d.s = &session;
  htp_dspq_conv_cached_req r = {
    {HTP_DSPQ_OP_CONV_CACHED, 1, 0, 32, 32, 32, {1, 2, 3}, 4, 1, 1e-5f},
    0,
    HTP_DSPQ_CONV_UPDATES};
  pack(&r, 1, 3, 5, 7);
  assert(run(&d, &r) == 0 && output[0] == 15 && output[96] == 8);
  r.updates = 0;
  pack(&r, 2, 0, 0, 0);
  assert(run(&d, &r) == 0 && output[0] == 21 && output[96] == 9);
  r.slot = 1;
  assert(run(&d, &r) == AEE_EBADSTATE); // No cross-slot history.
  r.slot = 0;
  r.conv.norm_kind = 0;
  r.conv.flags = HTP_DSPQ_FLAG_TIMED;
  pack(&r, 2, 0, 0, 0);
  assert(run(&d, &r) == 0 && output[0] == 14 && response.stage_us[0] == 123);
  r.conv.norm_kind = 2;
  pack(&r, 2, 0, 0, 0);
  assert(run(&d, &r) == 0 && output[0] == 43);
  r.updates = HTP_DSPQ_CONV_UPDATES;
  pack(&r, 2, 9, 2, 11);
  assert(run(&d, &r) == 0 && output[0] == 32);
  const int before = calls;
  --buffers[0].size;
  assert(run(&d, &r) == AEE_EBADPARM);
  ++buffers[0].size;
  --buffers[1].size;
  assert(run(&d, &r) == AEE_EBADPARM);
  ++buffers[1].size;
  r.slot = HTP_DSPQ_CONV_CACHE_SLOTS;
  assert(run(&d, &r) == AEE_EBADPARM);
  r.slot = 0;
  r.updates = 8;
  assert(run(&d, &r) == AEE_EBADPARM);
  r.updates = 0;
  r.conv.h_out = 5;
  pack(&r, 2, 0, 0, 0);
  assert(run(&d, &r) == AEE_EBADSTATE);
  r.conv.norm_kind = 0;
  r.updates = HTP_DSPQ_CONV_UPDATES;
  pack(&r, 2, 9, 2, 11);
  assert(run(&d, &r) == AEE_EBADPARM); // Gamma without norm.
  assert(calls == before);
  r.updates &= ~HTP_DSPQ_CONV_UPDATE_GAMMA;
  pack(&r, 2, 9, 0, 11);
  assert(run(&d, &r) == 0 && output[0] == 22); // Complete rebind.
  r.conv.norm_kind = 1;
  r.updates = 0;
  pack(&r, 2, 0, 0, 0);
  assert(run(&d, &r) == AEE_EBADSTATE); // Old binding's gamma was discarded.
  r.updates = HTP_DSPQ_CONV_UPDATE_GAMMA;
  pack(&r, 2, 0, 2, 0);
  assert(run(&d, &r) == 0 && output[0] == 25);
  fail_kernel = 1;
  assert(run(&d, &r) == AEE_EFAILED);
  fail_kernel = 0;
  assert(run(&d, &r) == AEE_EBADSTATE); // Failed call requires reseeding.
  r.conv.K = r.conv.C = r.conv.N_out = 64;
  r.updates = HTP_DSPQ_CONV_UPDATES;
  pack(&r, 1, 3, 5, 7);
  assert(run(&d, &r) == 0 && output[0] == 15); // Reallocation on new shape.
  for (unsigned i = 0; i < HTP_DSPQ_CONV_CACHE_SLOTS; ++i)
    free(d.conv_cache[i].data);
  puts("PASS: Conv queue cache reuse, updates, malformed packets, rebind, "
       "failure");
  return 0;
}
