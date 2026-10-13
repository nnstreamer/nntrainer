// SPDX-License-Identifier: Apache-2.0
// The extracted production entry/dispatch uses scalar arithmetic stand-ins.
// This checks row ownership, validation and alias transactions, not HVX math.
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "htp_rope_prefill_shape.h"

typedef uint32_t uint32;
typedef uint16_t uint16;
typedef uintptr_t remote_handle64;
typedef struct {
  uint32_t workers;
} hvx_worker_pool;
typedef struct {
  hvx_worker_pool *quant_pool;
} nntr_hvx_session;
#define AEE_SUCCESS 0
#define AEE_EINVALIDFORMAT 1
static uint64_t HAP_perf_get_qtimer_count(void) { return 0; }
static uint64_t HAP_perf_qtimer_count_to_us(uint64_t x) { return x; }
static void hvx_worker_pool_run(hvx_worker_pool *pool,
                                void (*fn)(uint32_t, uint32_t, void *),
                                void *ctx, uint32_t jobs) {
  const uint32_t n = pool->workers < jobs ? pool->workers : jobs;
  for (uint32_t i = 0; i < n; ++i)
    fn(n, i, ctx);
}
static uint16_t to_half(float x) {
  _Float16 y = (_Float16)x;
  uint16_t bits;
  memcpy(&bits, &y, sizeof(bits));
  return bits;
}
static void hvx_rope_row_f32(float *x, uint32_t n, uint32_t hd,
                             const float *cs) {
  for (uint32_t h = 0; h < n; h += hd)
    for (uint32_t j = 0; j < hd / 2; ++j) {
      const float a = x[h + j], b = x[h + j + hd / 2];
      x[h + j] = a * cs[j] - b * cs[hd + j];
      x[h + j + hd / 2] = a * cs[hd + j] + b * cs[j];
    }
}
typedef struct {
  const float *x, *cs;
  uint16_t *out;
  uint32_t rows, width;
} rope_half64_ctx;
static void rope_half64_worker(uint32_t threads, uint32_t tid, void *opaque) {
  const rope_half64_ctx *c = opaque;
  const uint32_t lo = (uint32_t)((uint64_t)c->rows * tid / threads);
  const uint32_t hi = (uint32_t)((uint64_t)c->rows * (tid + 1) / threads);
  for (uint32_t r = lo; r < hi; ++r) {
    const float *cs = c->cs + (size_t)r * 128;
    for (uint32_t h = 0; h < c->width; h += 64)
      for (uint32_t j = 0; j < 32; ++j) {
        const size_t at = (size_t)r * c->width + h + j;
        const float a = c->x[at], b = c->x[at + 32];
        c->out[at] = to_half(a * cs[j] - b * cs[64 + j]);
        c->out[at + 32] = to_half(a * cs[64 + j] + b * cs[j]);
      }
  }
}
typedef struct {
  const float *x, *aux;
  void *out;
  uint32_t count, op;
  float scale;
} math_ctx;
static void math_worker(uint32_t threads, uint32_t tid, void *opaque) {
  const math_ctx *c = opaque;
  assert(c->op == 6);
  const uint32_t blocks = (c->count + 63) / 64;
  const uint32_t lo = (uint32_t)((uint64_t)blocks * tid / threads) * 64;
  uint32_t hi = (uint32_t)((uint64_t)blocks * (tid + 1) / threads) * 64;
  if (hi > c->count)
    hi = c->count;
  for (uint32_t i = lo; i < hi; ++i)
    ((uint16_t *)c->out)[i] = to_half(c->x[i]);
}

// Generated verbatim from nntr_hvx_small_ops.c by run_rope_prefill_host.py.
#include "rope_prefill_under_test.inc"

static unsigned cases, rejected;
static void run_case(uint32_t rows, uint32_t qw, uint32_t kw, uint32_t workers,
                     uint32_t offset, uint32_t position) {
  const size_t nq = (size_t)rows * qw, nk = (size_t)rows * kw,
               nc = (size_t)rows * 128;
  float *qb = calloc(nq + 32, sizeof(float)),
        *kb = calloc(nk + 32, sizeof(float));
  float *vb = calloc(nk + 32, sizeof(float)),
        *cb = calloc(nc + 32, sizeof(float));
  float *q = qb + offset, *k = kb + offset, *v = vb + offset, *cs = cb + offset;
  uint16_t *kob = malloc((nk + 64) * 2), *vob = malloc((nk + 64) * 2);
  uint16_t *ko = kob + offset, *vo = vob + offset;
  float *qr = malloc(nq * 4);
  uint16_t *kr = malloc(nk * 2), *vr = malloc(nk * 2);
  assert(qb && kb && vb && cb && kob && vob && qr && kr && vr);
  memset(kob, 0xa5, (nk + 64) * 2);
  memset(vob, 0x5a, (nk + 64) * 2);
  uint32_t seed = 17 + position;
  for (size_t i = 0; i < nq; ++i) {
    seed = seed * 1664525u + 1013904223u;
    q[i] = ((int32_t)seed / 2147483648.f) * 8.f;
  }
  for (size_t i = 0; i < nk; ++i) {
    seed = seed * 1664525u + 1013904223u;
    k[i] = ((int32_t)seed / 2147483648.f) * 8.f;
    v[i] = k[i] * 1.001f;
  }
  const uint32_t bits[] = {0, 0x80000000u, 0x3f801000u, 0x477fe000u,
                           1, 0x33800000u, 0x7f800000u, 0x7fc12345u};
  for (size_t i = 0; i < nk; i += 71) {
    memcpy(k + i, &bits[(i / 71) % 8], 4);
    memcpy(v + i, &bits[(i / 71 + 3) % 8], 4);
  }
  for (uint32_t r = 0; r < rows; ++r)
    for (uint32_t d = 0; d < 64; ++d) {
      float angle = (float)((r + position) * (d % 32 + 1)) / 8192.f;
      cs[(size_t)r * 128 + d] = cosf(angle);
      cs[(size_t)r * 128 + 64 + d] = sinf(angle);
    }
  memcpy(qr, q, nq * 4);
  rope_half64_ctx kc = {k, cs, kr, rows, kw};
  rope_half64_worker(1, 0, &kc);
  math_ctx vc = {v, NULL, vr, (uint32_t)nk, 6, 1};
  math_worker(1, 0, &vc);
  for (uint32_t r = 0; r < rows; ++r)
    hvx_rope_row_f32(qr + (size_t)r * qw, qw, 64, cs + (size_t)r * 128);
  hvx_worker_pool pool = {workers};
  nntr_hvx_session session = {&pool};
  uint32_t elapsed = 99;
  assert(nntr_hvx_rope_qkv_prefill_f32((uintptr_t)&session, rows, qw, kw, 64,
                                       cs, nc, q, nq, k, nk, v, nk, q, nq, ko,
                                       nk, vo, nk, &elapsed, 1) == 0);
  assert(memcmp(q, qr, nq * 4) == 0 && memcmp(ko, kr, nk * 2) == 0 &&
         memcmp(vo, vr, nk * 2) == 0);
  assert(kob[nk + offset] == 0xa5a5 && vob[nk + offset] == 0x5a5a);
  if (offset)
    assert(kob[0] == 0xa5a5 && vob[0] == 0x5a5a);
  // Reject malformed lengths, shape and every writable cross-alias atomically.
  for (unsigned bad = 0; bad < 18; ++bad) {
    uint32_t m = rows, hd = 64, et = 1;
    int ql = nq, kl = nk, vl = nk, col = nk, vol = nk, cl = nc;
    const float *phase = cs, *kin = k, *vin = v;
    float *qo = q;
    uint16_t *kout = ko, *vout = vo;
    remote_handle64 h = (uintptr_t)&session;
    switch (bad) {
    case 0:
      m = 1;
      break;
    case 1:
      hd = 128;
      break;
    case 2:
      ql--;
      break;
    case 3:
      kl--;
      break;
    case 4:
      vl--;
      break;
    case 5:
      col--;
      break;
    case 6:
      vol--;
      break;
    case 7:
      cl--;
      break;
    case 8:
      et = 0;
      break;
    case 9:
      phase = NULL;
      break;
    case 10:
      h = 0;
      break;
    case 11:
      kout = vo;
      break;
    case 12:
      kout = (uint16_t *)k;
      break;
    case 13:
      vout = (uint16_t *)v;
      break;
    case 14:
      kin = q;
      break;
    case 15:
      vin = q;
      break;
    case 16:
      kout = (uint16_t *)cs;
      break;
    case 17:
      qo = q + 1;
      break;
    }
    assert(nntr_hvx_rope_qkv_prefill_f32(h, m, qw, kw, hd, phase, cl, q, ql,
                                         kin, kl, vin, vl, qo, nq, kout, col,
                                         vout, vol, &elapsed, et) != 0);
    assert(memcmp(q, qr, nq * 4) == 0 && memcmp(ko, kr, nk * 2) == 0 &&
           memcmp(vo, vr, nk * 2) == 0);
    ++rejected;
  }
  free(qb);
  free(kb);
  free(vb);
  free(cb);
  free(kob);
  free(vob);
  free(qr);
  free(kr);
  free(vr);
  ++cases;
}
int main(void) {
  const uint32_t rows[] = {2, 3, 5, 31, 32, 33, 65, 1024},
                 widths[][2] = {{64, 64}, {128, 64}, {512, 512}, {2048, 512}};
  const uint32_t workers[] = {1, 4, 8}, positions[] = {0, 31, 3072};
  for (unsigned r = 0; r < 8; ++r)
    for (unsigned w = 0; w < 4; ++w)
      for (unsigned t = 0; t < 3; ++t)
        for (unsigned a = 0; a < 2; ++a)
          for (unsigned p = 0; p < 3; ++p)
            run_case(rows[r], widths[w][0], widths[w][1], workers[t], a,
                     positions[p]);
  printf("ROPE PREFILL HOST PASS cases=%u invalid_transactions=%u\n", cases,
         rejected);
  return 0;
}
