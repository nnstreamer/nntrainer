// SPDX-License-Identifier: Apache-2.0
#include <fenv.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "m1_ops_det.h"
#include "router_bounds_reference.h"

/* The production helper calls the exact sigmoid defined by the reference. */
#include "router_bound_select.h"

static uint32_t state = 721;
static uint32_t random_bits(void) {
  state = state * 1664525u + 1013904223u;
  return state;
}
static float random_unit(void) {
  return (float)(random_bits() >> 8) * 0x1p-23f - 1.f;
}

static int check_table_boundaries(void) {
  unsigned cases = 0;
  static const uint32_t offsets[] = {0, 1, 0x3ffffu, 0x7fffeu, 0x7ffffu};
  for (unsigned sign = 0; sign < 2; ++sign) {
    for (unsigned bucket = 0; bucket < 448; ++bucket) {
      const uint32_t *bound = router_sigmoid_bounds[sign][bucket];
      if (bound[0] > bound[1])
        return 1;
      for (unsigned j = 0; j < sizeof(offsets) / sizeof(*offsets); ++j) {
        const uint32_t bits =
          (0x33800000u + (bucket << 19) + offsets[j]) | (sign << 31);
        float sig, score;
        prefill_router_sigmoid(m1_det_float(bits), 0.f, &sig, &score);
        if (m1_det_bits(sig) < bound[0] || m1_det_bits(sig) > bound[1]) {
          fprintf(stderr, "INNER_BOUND_FAIL logit_bits=0x%08x\n", bits);
          return 1;
        }
        ++cases;
      }
    }
    static const uint32_t outer[] = {
      0, 1, 0x337fffffu, 0x41800000u, 0x41800001u, 0x42b00000u, 0x7f7fffffu};
    for (unsigned j = 0; j < sizeof(outer) / sizeof(*outer); ++j) {
      const uint32_t *bound =
        router_sigmoid_outer_bounds[sign][outer[j] >= 0x41800000u];
      float sig, score;
      prefill_router_sigmoid(m1_det_float(outer[j] | (sign << 31)), 0.f, &sig,
                             &score);
      if (m1_det_bits(sig) < bound[0] || m1_det_bits(sig) > bound[1]) {
        fprintf(stderr, "OUTER_BOUND_FAIL logit_bits=0x%08x\n",
                outer[j] | (sign << 31));
        return 1;
      }
      ++cases;
    }
  }
  printf("ROUTER_BOUND_BOUNDARIES_PASSED cases=%u\n", cases);
  return 0;
}

static void fill(unsigned mode, unsigned E, unsigned row, float *x,
                 float *bias) {
  for (unsigned e = 0; e < E; ++e) {
    x[e] = random_unit() * (mode == 0 ? 8.f : 1.f);
    bias[e] = random_unit() * (mode == 0 ? 0.05f : 0.2f);
    if (mode == 2)
      x[e] *= 0.03125f;
    else if (mode == 3) {
      x[e] = (float)((int)(row % 17) - 8);
      bias[e] = 0.f;
    } else if (mode == 4) {
      x[e] = m1_det_float(random_bits());
      bias[e] = m1_det_float(random_bits());
    } else if (mode == 5) {
      uint32_t bucket = random_bits() % 448u;
      uint32_t bits = 0x33800000u + (bucket << 19);
      bits += (e % 3u == 0 ? 0u : e % 3u == 1 ? 0x7ffffu : 1u);
      x[e] = m1_det_float(bits | ((e & 1u) << 31));
    } else if (mode == 6) {
      float sig, score;
      prefill_router_sigmoid(x[e], 0.f, &sig, &score);
      bias[e] = 0.5f - sig;
    } else if (mode == 7 && e == row % E) {
      x[e] = m1_det_float(0x7f800000u + (row & 1u));
    } else if (mode == 8) {
      x[e] *= 128.f;
    } else if (mode == 9) {
      static const uint32_t special[] = {0,           0x80000000u, 1,
                                         0x80000001u, 0x7f7fffffu, 0xff7fffffu};
      bias[e] = m1_det_float(special[(e + row) % 6]);
    }
  }
}

static unsigned run(int bounded, const float *x, const float *bias, unsigned E,
                    unsigned top_k, unsigned n_sel, uint32_t *sel, float *w) {
  float sig[128], score[128];
  uint32_t index[128];
  unsigned exact =
    bounded ? prefill_router_bound_row(x, bias, E, n_sel, sig, score, index)
            : 0;
  const unsigned count = exact;
  if (!exact) {
    for (unsigned e = 0; e < E; ++e)
      prefill_router_sigmoid(x[e], bias[e], &sig[e], &score[e]);
    exact = E;
  }
  sigmoid_select(sig, score, exact, top_k, n_sel, sel, w);
  if (count)
    for (unsigned k = 0; k < n_sel; ++k)
      sel[k] = index[sel[k]];
  return exact;
}

int main(void) {
  if (fesetround(FE_TONEAREST) || check_table_boundaries())
    return 1;
  unsigned cases = 0;
  uint64_t exact[10] = {0}, total[10] = {0};
  for (unsigned mode = 0; mode < 10; ++mode)
    for (unsigned row = 0; row < 20000; ++row) {
      static const unsigned shapes[] = {1, 4, 16, 32, 64, 128};
      unsigned E = row % 2 ? 64 : shapes[(row / 2) % 6];
      static const unsigned selections[] = {1, 2, 4, 8, 9, 12, 15, 16, 32};
      unsigned n_sel = row % 5 == 0 ? selections[(row / 12) % 9] : 9;
      if (n_sel > E || row % 13 == 0)
        n_sel = E;
      unsigned top_k = row % 9 == 0 ? 1 : n_sel >= 4 ? 4 : n_sel;
      float x[128], bias[128], wa[128], wb[128];
      uint32_t sa[128], sb[128];
      fill(mode, E, row, x, bias);
      run(0, x, bias, E, top_k, n_sel, sa, wa);
      const unsigned count = run(1, x, bias, E, top_k, n_sel, sb, wb);
      if (memcmp(sa, sb, n_sel * sizeof(*sa)) ||
          memcmp(wa, wb, top_k * sizeof(*wa))) {
        fprintf(stderr, "PARITY_FAIL mode=%u row=%u E=%u n_sel=%u\n", mode, row,
                E, n_sel);
        return 1;
      }
      if (E == 64 && n_sel == 9) {
        exact[mode] += count;
        total[mode] += E;
      }
      ++cases;
    }
  printf("ROUTER_BOUND_ROWS_PASSED cases=%u\n", cases);
  for (unsigned mode = 0; mode < 10; ++mode)
    printf("COUNTS mode=%u exact=%llu total=%llu pruned_percent=%.3f\n", mode,
           (unsigned long long)exact[mode], (unsigned long long)total[mode],
           100.0 * (1.0 - (double)exact[mode] / total[mode]));
  return 0;
}
