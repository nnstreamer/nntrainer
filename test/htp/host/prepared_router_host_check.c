// SPDX-License-Identifier: Apache-2.0
/**
 * @file prepared_router_host_check.c
 * @brief Check the current prepared router RPC transaction and exact Q8 reuse.
 * @author NNTrainer contributors
 * @bug See the repository issue tracker for known issues.
 *
 * The runner extracts production RPC control flow at build time. HMX and
 * FastRPC are replaced by observable stubs; quantization, AH packing and RMS
 * normalization use the real HVX sources on the existing lane emulation.
 */

#include "hvx_quant_u8.h"
#include "hvx_rmsnorm_rows_f32.h"
#include <assert.h>
#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef uint32_t uint32;
typedef uint64_t uint64, remote_handle64;
#define AEE_SUCCESS 0
#define AEE_EBADPARM 14
#define AEE_EUNSUPPORTED 20
#define AEE_EUNSUPPORTEDAPI 108
#define AEE_EFAILED 1
#define AEE_ENOMEMORY 2
#define HEXKL_MM_U8I4_MAX_WEIGHTS 4
#define HEXKL_MOE_FLAG_DMA_BYPASS 0x40000u

typedef struct {
  const uint8_t *rows;
  const float *scale;
  const int32_t *zp;
  uint32 M, K;
} hexkl_moe_prepared_input;
typedef struct {
  uint8_t *base;
} hexkl_moe_scratch;
typedef struct {
  int in_use;
  uint32 K, N, bits;
} weight;
typedef struct {
  weight slots[4];
} table;
typedef struct {
  hvx_worker_pool *quant_pool;
  uint8_t *vtcm_base;
  uint32 vtcm_size, config_off, moe_flags;
  int hmx_locked;
  table weights_u8i4;
  uint64 router_prepare_serial, router_prepare_cookie;
  hexkl_moe_prepared_input router_prepared;
  hexkl_moe_scratch router_q4_scratch, moe_scratch;
  float *norm_rows;
} nntr_hvx_session;
typedef struct {
  hvx_worker_pool *pool;
  const float *act_scale;
  const int32_t *act_zp;
  const uint8_t *act_ah_prepacked;
  int weight_src_bypass;
} hexkl_mm_opts;
static uint8_t scratch[1 << 20] __attribute__((aligned(128)));
static uint8_t vtcm[1 << 16] __attribute__((aligned(128)));
static float cached[6 * 128];
static int scratch_rc, norm_fail, fc_rc, selection_rc;
static unsigned published, fc_calls, select_calls;
static int hexkl_moe_scratch_reserve(hexkl_moe_scratch *s, size_t n) {
  assert(n <= sizeof scratch);
  if (scratch_rc)
    return scratch_rc;
  s->base = scratch;
  return 0;
}
static void *hexkl_moe_carve(uint8_t **p, size_t n) {
  void *out = *p;
  *p += n;
  return out;
}
static int norm_rows_reserve(nntr_hvx_session *s, size_t n) {
  assert(n <= 6 * 128);
  if (norm_fail)
    return 0;
  s->norm_rows = cached;
  return 1;
}
static void observed_norm(const float *x, const float *res, float *sum,
                          float *norm, float *cache, uint32 M, uint32 K,
                          const float *gamma, float eps) {
  hvx_rmsnorm_rows_cached_f32(x, res, sum, norm, cache, M, K, gamma, eps);
  published += (size_t)M * K;
}
void hvx_worker_pool_run(hvx_worker_pool *p, void (*f)(uint32, uint32, void *),
                         void *ctx, uint32 units) {
  (void)p;
  uint32 n = units < 4 ? units : 4;
  for (uint32 i = 0; i < n; ++i)
    f(n, i, ctx);
}
static int fc_layer_run(nntr_hvx_session *s, uint32 M, uint32 K,
                        const uint32 *h, uint32 n, const float *x, float *y,
                        const hexkl_mm_opts *o) {
  (void)s;
  (void)M;
  (void)K;
  (void)h;
  (void)n;
  (void)x;
  (void)o;
  fc_calls++;
  if (!fc_rc)
    y[0] = 1;
  return fc_rc;
}
static int nntr_hvx_sigmoid_topk_f32(remote_handle64 h, uint32 M, uint32 E,
                                     uint32 top, uint32 ns, const float *x,
                                     int xl, const float *b, int bl,
                                     uint32 *sel, int sl, float *w, int wl) {
  (void)h;
  (void)M;
  (void)E;
  (void)top;
  (void)ns;
  (void)x;
  (void)xl;
  (void)b;
  (void)bl;
  (void)sl;
  (void)wl;
  select_calls++;
  if (!selection_rc) {
    sel[0] = 0;
    w[0] = 1;
  }
  return selection_rc;
}

#define HEXKL_MOE_FLAG_GELU_TANH 1u
#define MOE_N_STAGES 31
#define FARF(...) ((void)0)
static int hexkl_probe_on;
static void hexkl_probe_reset(int on) { hexkl_probe_on = on; }
static uint64_t hexkl_probe_now(void) { return 0; }
static int nntr_hvx_moe_stage_fill(uint32 *stage, uint64 begin, uint64 end,
                                   int rc) {
  (void)begin;
  (void)end;
  memset(stage, 0, MOE_N_STAGES * sizeof(uint32));
  return rc;
}
static unsigned moe_calls;
static int moe_rc;
static int hexkl_mm_u8i4_moe_layer_run_prepared(
  table *tbl, uint8_t *vtcm_base, uint32 vtcm_size, uint32 config_off, uint32 M,
  uint32 K, uint32 inter, uint32 N, uint32 experts, const uint32 *gu,
  const uint32 *dn, const uint32 *rows, const uint32 *counts,
  const float *weights, const float *input, float *output,
  hvx_worker_pool *pool, hexkl_moe_scratch *scratch, uint32 flags,
  const hexkl_moe_prepared_input *p) {
  (void)tbl;
  (void)vtcm_base;
  (void)vtcm_size;
  (void)config_off;
  (void)inter;
  (void)N;
  (void)experts;
  (void)gu;
  (void)dn;
  (void)rows;
  (void)counts;
  (void)weights;
  (void)pool;
  (void)scratch;
  (void)flags;
  assert(!input && p && p->rows && p->scale && p->zp && p->M == M && p->K == K);
  ++moe_calls;
  if (!moe_rc)
    output[0] = p->scale[0];
  return moe_rc;
}
static int nntr_hvx_mm_u8i4_layer(remote_handle64 h, uint32 M, uint32 K,
                                  const uint32 *handle, int count,
                                  const float *x, int xlen, float *logits,
                                  int logitslen) {
  (void)h;
  (void)M;
  (void)K;
  (void)handle;
  (void)count;
  (void)x;
  (void)xlen;
  (void)logitslen;
  logits[0] = 1;
  return 0;
}
#define hvx_rmsnorm_rows_cached_f32 observed_norm
#include "prepared_router_under_test.inc"
#undef hvx_rmsnorm_rows_cached_f32

static nntr_hvx_session session;
static float x[5 * 128 + 32], res[5 * 128], sum[5 * 128], norm[5 * 128],
  gamma[128], bias[64], weights[5 * 4];
static uint32 selected[5 * 8];
static uint64 cookie;
static void reset(void) {
  memset(&session, 0, sizeof session);
  session.hmx_locked = 1;
  session.vtcm_base = vtcm;
  session.vtcm_size = session.config_off = sizeof vtcm;
  session.weights_u8i4.slots[0] = (weight){1, 128, 64, 4};
  session.router_prepare_cookie = 777;
  session.router_prepared =
    (hexkl_moe_prepared_input){scratch, cached, (int32_t *)cached, 5, 128};
  scratch_rc = norm_fail = fc_rc = selection_rc = 0;
  published = fc_calls = select_calls = 0;
  cookie = 123;
  for (unsigned j = 0; j < 128; ++j)
    gamma[j] = 1.0f;
  for (unsigned j = 0; j < 5 * 128; ++j) {
    x[j] = j;
    res[j] = 1;
    sum[j] = norm[j] = -99;
  }
}
static int call(float *out_sum, float *out_norm) {
  return nntr_hvx_router_q4_prefill_norm_prepare(
    (remote_handle64)(uintptr_t)&session, 5, 128, 64, 0, 4, 8, 1e-5f, x,
    5 * 128, res, 5 * 128, gamma, 128, out_sum, 5 * 128, out_norm, 5 * 128,
    bias, 64, selected, 5 * 8, weights, 5 * 4, &cookie);
}
static void early(int expected) {
  int rc = call(sum, norm);
  assert(rc == expected);
  assert(!published && !fc_calls && !select_calls && !cookie &&
         !session.router_prepare_cookie);
  assert(sum[0] == -99 && norm[0] == -99);
}
static int check_transaction(void) {
  unsigned cases = 0;
  reset();
  session.hmx_locked = 0;
  early(AEE_EUNSUPPORTED);
  cases++;
  reset();
  session.config_off = 1;
  early(AEE_EUNSUPPORTED);
  cases++;
  reset();
  session.weights_u8i4.slots[0].bits = 2;
  early(AEE_EBADPARM);
  cases++;
  reset();
  session.weights_u8i4.slots[0].N = 128;
  early(AEE_EBADPARM);
  cases++;
  reset();
  scratch_rc = AEE_ENOMEMORY;
  early(AEE_ENOMEMORY);
  cases++;
  reset();
  norm_fail = 1;
  early(AEE_ENOMEMORY);
  cases++;
  reset();
  assert(call(x + 32, norm) == AEE_EBADPARM && !published && !cookie &&
         !session.router_prepare_cookie);
  cases++;
  reset();
  assert(call(norm, norm) == AEE_EBADPARM && !published && !cookie &&
         !session.router_prepare_cookie);
  cases++;
  reset();
  assert(call(sum, NULL) == AEE_EBADPARM && !published && !cookie &&
         !session.router_prepare_cookie);
  cases++;
  const int errors[] = {AEE_EUNSUPPORTED, AEE_EUNSUPPORTEDAPI, AEE_EBADPARM,
                        AEE_ENOMEMORY};
  for (unsigned i = 0; i < sizeof errors / sizeof errors[0]; ++i) {
    int expected =
      (errors[i] == AEE_EUNSUPPORTED || errors[i] == AEE_EUNSUPPORTEDAPI)
        ? AEE_EFAILED
        : errors[i];
    reset();
    fc_rc = errors[i];
    assert(call(x, norm) == expected && published == 5 * 128 && fc_calls == 1 &&
           !select_calls && !cookie && !session.router_prepare_cookie);
    cases++;
    reset();
    selection_rc = errors[i];
    assert(call(sum, res) == expected && published == 5 * 128 &&
           fc_calls == 1 && select_calls == 1 && !cookie &&
           !session.router_prepare_cookie);
    cases++;
  }
  reset();
  assert(call(x, res) == AEE_SUCCESS && published == 5 * 128 && fc_calls == 1 &&
         select_calls == 1 && cookie == 1 &&
         session.router_prepare_cookie == 1);
  cases++;
  reset();
  fc_rc = AEE_EUNSUPPORTED;
  assert(nntr_hvx_router_q4_prefill_prepare(
           (remote_handle64)(uintptr_t)&session, 5, 128, 64, 0, x, 5 * 128, 4,
           8, bias, 64, selected, 5 * 8, weights, 5 * 4,
           &cookie) == AEE_EUNSUPPORTED &&
         !published && !cookie);
  cases++;
  printf("NORM_TRANSACTION_HOST_PASS cases=%u\n", cases);
  return 0;
}

static uint32 gu[1] = {0}, dn[1] = {0}, row_index[5] = {0, 1, 2, 3, 4},
              row_count[1] = {5};
static float row_weight[5] = {1, 1, 1, 1, 1}, out[5 * 128];
static int prepare(void) {
  return nntr_hvx_router_q4_prefill_prepare(
    (remote_handle64)(uintptr_t)&session, 5, 128, 64, 0, x, 5 * 128, 4, 8, bias,
    64, selected, 5 * 8, weights, 5 * 4, &cookie);
}
static int consume(uint64 lease, uint32 M, uint32 K, uint32 I, uint32 N,
                   const uint32 *g, const uint32 *d, const uint32 *ri,
                   const uint32 *rc, const float *rw) {
  return nntr_hvx_mm_u8i4_moe_layer_prepared(
    (remote_handle64)(uintptr_t)&session, lease, M, K, I, N, 0, g, 1, d, 1, ri,
    5, rc, 1, rw, 5, out, 5 * 128, NULL, 0);
}
static int consume_current(uint64 c) {
  return consume(c, 5, 128, 128, 128, gu, dn, row_index, row_count, row_weight);
}
static void assert_prepared_quant(void) {
  uint8_t reference_ah[64 * 128], gathered[64 * 128];
  float scale[64];
  int32_t zp[64];
  uint32_t map[64];
  assert(!hvx_quant_params_pack_u8_ah(x, 5, 64, 128, NULL, 0, NULL, scale, zp,
                                      reference_ah, NULL));
  for (unsigned j = 0; j < 64; ++j)
    map[j] = j;
  const hexkl_moe_prepared_input *p = &session.router_prepared;
  hvx_gather_rowmajor_u8_ah_rows(p->rows, map, 0, 64, 128, gathered);
  assert(!memcmp(scale, p->scale, sizeof scale) &&
         !memcmp(zp, p->zp, sizeof zp));
  assert(!memcmp(reference_ah, gathered, sizeof gathered));
  assert(!memcmp(reference_ah, vtcm, sizeof reference_ah));
}
static int check_cookie(void) {
  unsigned cases = 0;
  reset();
  moe_calls = 0;
  moe_rc = 0;
  assert(!prepare());
  assert_prepared_quant();
  uint64 first = cookie;
  assert(first && session.router_prepare_cookie == first);
  assert(!consume_current(first));
  assert(moe_calls == 1 && !session.router_prepare_cookie);
  assert(consume_current(first) == AEE_EBADPARM && moe_calls == 1);
  cases++;
  reset();
  assert(!prepare());
  first = cookie;
  assert(!prepare());
  assert(cookie != first);
  assert(consume_current(first) == AEE_EBADPARM);
  assert(consume_current(cookie) == AEE_EBADPARM);
  cases++;
  reset();
  assert(!prepare());
  assert(consume(cookie, 5, 256, 128, 128, gu, dn, row_index, row_count,
                 row_weight) == AEE_EBADPARM);
  assert(consume_current(cookie) == AEE_EBADPARM);
  cases++;
  reset();
  assert(!prepare());
  first = cookie;
  assert(nntr_hvx_router_q4_prefill((remote_handle64)(uintptr_t)&session, 5,
                                    128, 64, 0, x, 5 * 128, 4, 8, bias, 64,
                                    selected, 5 * 8, weights, 5 * 4) == 0);
  assert(consume_current(first) == AEE_EBADPARM);
  cases++;
  reset();
  session.router_prepare_serial = UINT64_MAX;
  assert(!prepare() && cookie == 1);
  cases++;
  for (unsigned missing = 0; missing < 5; ++missing) {
    reset();
    assert(!prepare());
    unsigned before = moe_calls;
    assert(consume(cookie, 5, 128, 128, 128, missing == 0 ? NULL : gu,
                   missing == 1 ? NULL : dn, missing == 2 ? NULL : row_index,
                   missing == 3 ? NULL : row_count,
                   missing == 4 ? NULL : row_weight) == AEE_EBADPARM);
    assert(moe_calls == before && !session.router_prepare_cookie);
    cases++;
  }
  const uint32 shapes[][4] = {{UINT32_MAX, 128, 128, 128},
                              {5, 128, 128, UINT32_MAX},
                              {5, 128, UINT32_MAX, 128}};
  for (unsigned j = 0; j < 3; ++j) {
    reset();
    session.router_prepare_cookie = 123;
    session.router_prepared.M = shapes[j][0];
    session.router_prepared.K = shapes[j][1];
    unsigned before = moe_calls;
    assert(consume(123, shapes[j][0], shapes[j][1], shapes[j][2], shapes[j][3],
                   gu, dn, row_index, row_count, row_weight) == AEE_EBADPARM &&
           moe_calls == before && !session.router_prepare_cookie);
    cases++;
  }
  reset();
  assert(!prepare());
  row_count[0] = 4;
  assert(consume_current(cookie) == AEE_EBADPARM);
  row_count[0] = 5;
  assert(consume_current(cookie) == AEE_EBADPARM);
  cases++;
  reset();
  assert(!prepare());
  moe_rc = AEE_EFAILED;
  assert(consume_current(cookie) == AEE_EFAILED &&
         !session.router_prepare_cookie);
  moe_rc = 0;
  assert(consume_current(cookie) == AEE_EBADPARM);
  cases++;
  printf("PREPARED_COOKIE_HOST_PASS cases=%u\n", cases);
  return 0;
}
static int check_quant(void) {
  const uint32_t ms[] = {5, 63, 64, 65, 129, 256, 1024, 1087};
  const uint32_t ks[] = {128, 256, 384, 2048};
  uint32_t seed = 19, checked = 0;
  for (uint32_t mi = 0; mi < 8; ++mi)
    for (uint32_t ki = 0; ki < 4; ++ki) {
      const uint32_t m = ms[mi], k = ks[ki], slots = 128,
                     pad = (m + 63u) / 64u * 64u;
      float *x = malloc((size_t)m * k * sizeof(float)),
            *scale = malloc(pad * sizeof(float)),
            *rs = malloc(pad * sizeof(float));
      int32_t *zp = malloc(pad * sizeof(int32_t)),
              *rz = malloc(pad * sizeof(int32_t));
      uint8_t *q = calloc((size_t)pad, k), *ref = malloc((size_t)slots * k),
              *got = malloc((size_t)slots * k);
      float ss[128];
      int32_t zz[128];
      uint32_t map[128];
      for (uint32_t r = 0; r < m; ++r)
        for (uint32_t c = 0; c < k; ++c) {
          seed = seed * 1664525u + 1013904223u;
          x[(size_t)r * k + c] =
            r == 0   ? 0.0f
            : r == 1 ? 3.125f
            : r == 2 ? -2.75f
                     : ((int32_t)(seed % 200001u) - 100000) * 0.00731f;
        }
      hvx_quant_rows_u8_params(x, m, pad, k, rs, rz, NULL);
      memset(scale, 0x99, pad * sizeof(float));
      memset(zp, 0x99, pad * sizeof(int32_t));
      hvx_quant_rows_u8_params_rowmajor(x, m, pad, k, scale, zp, q, NULL);
      if (memcmp(rs, scale, pad * sizeof(float)) ||
          memcmp(rz, zp, pad * sizeof(int32_t))) {
        printf("PARAMS FAIL m=%u k=%u\n", m, k);
        return 1;
      }
      uint32_t *identity = malloc(pad * sizeof(uint32_t));
      uint8_t *old_ah = malloc((size_t)pad * k),
              *new_ah = malloc((size_t)pad * k);
      float *old_sc = malloc(pad * sizeof(float));
      int32_t *old_zp = malloc(pad * sizeof(int32_t));
      for (uint32_t r = 0; r < pad; ++r)
        identity[r] = r;
      if (hvx_quant_params_pack_u8_ah(x, m, pad, k, NULL, 0.0f, NULL, old_sc,
                                      old_zp, old_ah, NULL))
        return 3;
      hvx_gather_rowmajor_u8_ah_rows(q, identity, 0, pad, k, new_ah);
      if (memcmp(old_sc, scale, pad * sizeof(float)) ||
          memcmp(old_zp, zp, pad * sizeof(int32_t)) ||
          memcmp(old_ah, new_ah, (size_t)pad * k)) {
        printf("FUSED FC QUANT DIFFER m=%u k=%u\n", m, k);
        return 4;
      }
      free(identity);
      free(old_ah);
      free(new_ah);
      free(old_sc);
      free(old_zp);
      for (uint32_t r = 0; r < slots; ++r) {
        map[r] = (r * 37u + 11u) % m;
        ss[r] = scale[map[r]];
        zz[r] = zp[map[r]];
      }
      if (hvx_quant_pack_u8_ah_mapped(x, map, slots, slots, k, ss, zz, ref,
                                      NULL))
        return 2;
      memset(got, 0x9b, (size_t)slots * k);
      for (uint32_t r = 0; r < slots; r += 16u)
        hvx_gather_rowmajor_u8_ah_rows(q, map, r, r + 16u, k, got);
      if (memcmp(ref, got, (size_t)slots * k)) {
        printf("QUANT REUSE FAIL m=%u k=%u\n", m, k);
        return 1;
      }
      for (uint32_t r = 0; r < m; ++r)
        for (uint32_t c = 0; c < k; ++c)
          q[(size_t)r * k + c] = (uint8_t)((r * 19u + c * 31u) ^ (c >> 5));
      for (uint32_t r = 0; r < slots; r += 16u)
        hvx_gather_rowmajor_u8_ah_rows(q, map, r, r + 16u, k, got);
      for (uint32_t r = 0; r < slots; ++r)
        for (uint32_t c = 0; c < k; ++c) {
          size_t off = (size_t)(r / 64u) * 64u * k + (c / 32u) * 2048u +
                       (r % 64u) * 32u + c % 32u;
          if (got[off] != q[(size_t)map[r] * k + c]) {
            printf("TRANSPOSE FAIL m=%u k=%u\n", m, k);
            return 1;
          }
        }
      free(x);
      free(scale);
      free(zp);
      free(rs);
      free(rz);
      free(q);
      free(ref);
      free(got);
      ++checked;
    }
  printf("Q8 ROW REUSE OK: %u shapes, exact AH bytes and independent transpose "
         "reference\n",
         checked);
  return 0;
}

int main(void) {
  return check_transaction() || check_cookie() || check_quant();
}
