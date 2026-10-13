// SPDX-License-Identifier: Apache-2.0
/** Conservative pruning before the unchanged exact sigmoid/top-k path.
 *
 * The table enumerates every binary32 input in 2^-24 <= |x| < 16 using
 * the current Bionic expf sequence, rounded addition, and RN division.
 * The denominator and reciprocal are normal throughout that domain.
 * The outer table exhaustively covers every remaining finite input.
 *
 * RN addition by a fixed finite bias is monotone, so bounds remain valid
 * after bias. Surviving candidates are stored in ascending original index
 * order, so the unchanged selector has the same tie order. The nth
 * greatest lower bound is no greater than the nth greatest exact score.
 * Prune only upper < threshold: equality always
 * receives the original exact sigmoid and ordered selection, preserving
 * index ties and every selected weight. Non-finite rows use the old path.
 */
#ifndef NNTR_ROUTER_BOUND_SELECT_H
#define NNTR_ROUTER_BOUND_SELECT_H

#include "router_sigmoid_bounds.h"
#include "router_sigmoid_outer_bounds.h"

static inline uint32_t prefill_router_bound_row(const float *logits,
                                                const float *bias, uint32_t E,
                                                uint32_t n_sel, float *sig,
                                                float *score, uint32_t *index) {
  if (E < 32u || n_sel > 16u || n_sel == E)
    return 0;
  float upper[128], best[16];
  const float neg_inf = m1_det_float(0xff800000u);
  for (uint32_t k = 0; k < n_sel; ++k)
    best[k] = neg_inf;
  for (uint32_t e = 0; e < E; ++e) {
    const uint32_t bits = m1_det_bits(logits[e]);
    const uint32_t mag = bits & 0x7fffffffu;
    if (mag >= 0x7f800000u ||
        (m1_det_bits(bias[e]) & 0x7fffffffu) >= 0x7f800000u)
      return 0;
    float low, high;
    if (mag >= 0x33800000u && mag < 0x41800000u) {
      const uint32_t bucket = (mag - 0x33800000u) >> 19;
      low = m1_det_float(router_sigmoid_bounds[bits >> 31][bucket][0]);
      high = m1_det_float(router_sigmoid_bounds[bits >> 31][bucket][1]);
    } else {
      const uint32_t region = mag >= 0x41800000u;
      low = m1_det_float(router_sigmoid_outer_bounds[bits >> 31][region][0]);
      high = m1_det_float(router_sigmoid_outer_bounds[bits >> 31][region][1]);
    }
    const float lower = m1_det_add(low, bias[e]);
    upper[e] = m1_det_add(high, bias[e]);
    if (lower > best[n_sel - 1u]) {
      uint32_t k = n_sel - 1u;
      while (k > 0u && lower > best[k - 1u]) {
        best[k] = best[k - 1u];
        --k;
      }
      best[k] = lower;
    }
  }
  uint32_t exact = 0;
  for (uint32_t e = 0; e < E; ++e) {
    if (upper[e] < best[n_sel - 1u])
      continue;
    prefill_router_sigmoid(logits[e], bias[e], &sig[exact], &score[exact]);
    index[exact] = e;
    ++exact;
  }
  return exact;
}

#endif /* NNTR_ROUTER_BOUND_SELECT_H */
