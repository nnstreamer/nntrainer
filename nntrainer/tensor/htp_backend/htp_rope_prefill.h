// SPDX-License-Identifier: Apache-2.0
#ifndef NNTR_HTP_ROPE_PREFILL_H
#define NNTR_HTP_ROPE_PREFILL_H

#include <cstdint>

namespace nntrainer {

/** HTP-only prefill preparation. Q is rotated in place in FP32; K is rotated
 * and rounded to FP16; V is rounded to FP16. cs is [rows][cos64,sin64].
 * False is unsupported and leaves all outputs untouched. No ComputeOps
 * vtable change: the existing generic and decode interfaces remain intact. */
#ifdef ENABLE_HEXKL
bool htp_rope_qkv_prefill(unsigned rows, unsigned q_width, unsigned kv_width,
                          unsigned head_dim, const float *cs, float *q,
                          const float *k, const float *v, uint16_t *k_out,
                          uint16_t *v_out);
#else
inline bool htp_rope_qkv_prefill(unsigned, unsigned, unsigned, unsigned,
                                 const float *, float *, const float *,
                                 const float *, uint16_t *, uint16_t *) {
  return false;
}
#endif

} // namespace nntrainer
#endif
