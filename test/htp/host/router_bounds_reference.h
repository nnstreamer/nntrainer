// SPDX-License-Identifier: Apache-2.0
/** Original exact router sequence used as the independent selection reference.
 */
#ifndef NNTR_ROUTER_BOUNDS_HOST_REFERENCE_H
#define NNTR_ROUTER_BOUNDS_HOST_REFERENCE_H

#include "m1_ops_det.h"

static inline void sigmoid_select(const float *sig, const float *score,
                                  uint32_t E, uint32_t top_k, uint32_t n_sel,
                                  uint32_t *sel, float *weight) {
  uint8_t taken[128] = {0};
  for (uint32_t k = 0; k < n_sel; ++k) {
    uint32_t best = E;
    for (uint32_t e = 0; e < E; ++e)
      if (!taken[e] && (best == E || score[e] > score[best]))
        best = e;
    taken[best] = 1;
    sel[k] = best;
  }
  float sum = 0.0f;
  for (uint32_t k = 0; k < top_k; ++k)
    sum += sig[sel[k]];
  const float inv = 1.0f / (sum + 1e-6f);
  for (uint32_t k = 0; k < top_k; ++k)
    weight[k] = sig[sel[k]] * inv;
}

/** Exact prefill sigmoid. Scalar division uses the compiler's correctly
 *  rounded reciprocal/refinement sequence for the normal denominator and
 *  result. Keep the integer fallback for exceptional inputs and subnormals. */
static inline void prefill_router_sigmoid(float logit, float bias, float *sig,
                                          float *score) {
  const float ex =
    m1_expf_bionic_det(m1_det_float(m1_det_bits(logit) ^ 0x80000000u));
  const float denominator = m1_det_add(ex, 1.0f);
  const uint32_t bits = m1_det_bits(denominator);
  *sig = bits >= 0x3f800000u && bits < 0x7e800000u
           ? 1.0f / denominator
           : cpu_det_div_rn(1.0f, denominator);
  *score = m1_det_add(*sig, bias);
}

#endif /* NNTR_ROUTER_BOUNDS_HOST_REFERENCE_H */
