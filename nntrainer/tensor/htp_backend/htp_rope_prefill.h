// SPDX-License-Identifier: Apache-2.0
#ifndef NNTR_HTP_ROPE_PREFILL_H
#define NNTR_HTP_ROPE_PREFILL_H

#include <cstdint>

namespace nntrainer {

enum class HtpPrefillAttentionResult { Unsupported, Prepared, Complete };

/** One exact preparation/attention RPC for the initial 1024-row LFM2 shape.
 * Prepared means Q and the public FP16 K/V cache were written but attention
 * failed; the caller must retire its quantized handle and skip preparation
 * before the existing attention fallback. Transport uncertainty throws.
 * Unsupported alone guarantees no input/output or cache publication. */
#ifdef ENABLE_HEXKL
HtpPrefillAttentionResult
htp_rope_q8_attention_prefill(int kv_handle, unsigned rows, const float *cs,
                              float *q, const float *k, const float *v,
                              uint16_t *k_out, uint16_t *v_out, float *out);
#else
inline HtpPrefillAttentionResult
htp_rope_q8_attention_prefill(int, unsigned, const float *, float *,
                              const float *, const float *, uint16_t *,
                              uint16_t *, float *) {
  return HtpPrefillAttentionResult::Unsupported;
}
#endif

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
