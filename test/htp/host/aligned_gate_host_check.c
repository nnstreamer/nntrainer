// SPDX-License-Identifier: Apache-2.0
/** @file aligned_gate_host_check.c
 * @brief Compare actual HVX gate source against the separate FP32 scalar spec.
 */
#include "hvx_conv_gate_f32.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hvx_alignment_host_alloc.h"

static unsigned misaligned;
static void *test_alloc(size_t size) {
  return hvx_alignment_host_alloc(size, misaligned == 1u ? 4u : 0u);
}
static void test_free(void *p) { hvx_alignment_host_free(p); }

static uint32_t threads = 1;
void hvx_worker_pool_run(hvx_worker_pool *pool, hvx_worker_pool_func func,
                         void *ctx, uint32_t units) {
  (void)pool;
  uint32_t n = units < threads ? units : threads;
  for (uint32_t i = 0; i < n; ++i)
    func(n, i, ctx);
}

static float random_value(uint32_t *seed) {
  *seed = *seed * 1664525u + 1013904223u;
  return ((int32_t)(*seed >> 8) - 8388608) * 0.0000003f;
}

/* The prefill gate contract has a separate FP32 multiply at each tap,
 * two FP32 additions, then the final gate multiply. M1 uses a different
 * fused order, so its scalar helper is deliberately not used here.
 * Volatile intermediates also match the SF operations in hvx_emu/.
 */
static void gate_reference(float *z, uint32_t zs, const float *g, uint32_t gs,
                           uint32_t t0, uint32_t rows, uint32_t width,
                           const float *w) {
  for (uint32_t r = 0; r < rows; ++r) {
    const uint32_t t = t0 + r;
    for (uint32_t j = 0; j < width; ++j) {
      const float x0 = g[(size_t)t * gs + j];
      const float x1 = t >= 1u ? g[(size_t)(t - 1u) * gs + j] : 0.0f;
      const float x2 = t >= 2u ? g[(size_t)(t - 2u) * gs + j] : 0.0f;
      volatile float p0 = w[j] * x0;
      volatile float p1 = w[width + j] * x1;
      volatile float p2 = w[2u * width + j] * x2;
      volatile float a01 = p0 + p1;
      volatile float a012 = a01 + p2;
      volatile float result = z[(size_t)r * zs + j] * a012;
      z[(size_t)r * zs + j] = result;
    }
  }
}

int main(int argc, char **argv) {
  misaligned = argc > 1 ? (unsigned)atoi(argv[1]) : 0u;
  if (argc > 2 || misaligned > 2u)
    return 2;
  const uint32_t widths[] = {32, 96, 2048};
  const uint32_t rows[] = {0, 1, 2, 3, 4, 5, 7, 15, 16, 31, 64};
  const uint32_t starts[] = {0, 1, 2, 7};
  unsigned checked = 0;
  uint32_t seed = 0xc0a71234u;
  for (threads = 1; threads <= 4; ++threads) {
    for (unsigned wi = 0; wi < sizeof widths / sizeof widths[0]; ++wi) {
      const uint32_t C = widths[wi], zs = C + (misaligned == 2u ? 7u : 0u),
                     gs = C + (misaligned == 2u ? 11u : 0u);
      float *g = test_alloc(80u * gs * sizeof(float));
      float *w = test_alloc(3u * C * sizeof(float));
      float *ref = test_alloc(64u * zs * sizeof(float));
      float *got = test_alloc(64u * zs * sizeof(float));
      if (!g || !w || !ref || !got)
        return 2;
      for (uint32_t i = 0; i < 80u * gs; ++i)
        g[i] = random_value(&seed);
      for (uint32_t i = 0; i < 3u * C; ++i)
        w[i] = random_value(&seed);
      for (unsigned ri = 0; ri < sizeof rows / sizeof rows[0]; ++ri) {
        for (unsigned ti = 0; ti < sizeof starts / sizeof starts[0]; ++ti) {
          for (uint32_t i = 0; i < 64u * zs; ++i)
            got[i] = ref[i] = random_value(&seed);
          gate_reference(ref, zs, g, gs, starts[ti], rows[ri], C, w);
          hvx_conv_gate_f32(got, zs, g, gs, starts[ti], rows[ri], C, w, NULL);
          if (memcmp(ref, got, 64u * zs * sizeof(float))) {
            printf("FAIL C=%u rows=%u start=%u threads=%u\n", C, rows[ri],
                   starts[ti], threads);
            return 1;
          }
          ++checked;
        }
      }
      test_free(g);
      test_free(w);
      test_free(ref);
      test_free(got);
    }
  }
  printf("ALIGNED CONV GATE PASS: %u exact cases, mode=%u, full stride span\n",
         checked, misaligned);
  return 0;
}
