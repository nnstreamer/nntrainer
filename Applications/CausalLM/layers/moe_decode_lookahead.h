// SPDX-License-Identifier: Apache-2.0
/** @file moe_decode_lookahead.h
 * @brief Experimental decode prefetch at mixer entry or previous router input.
 */
#ifndef NNTR_MOE_DECODE_LOOKAHEAD_H
#define NNTR_MOE_DECODE_LOOKAHEAD_H
#include <string>

namespace causallm {
using MoeDecodeLookaheadCallback = void (*)(const std::string &, unsigned,
                                            unsigned, const float *,
                                            const float *, const float *,
                                            unsigned, float);
/** Optional callback keeps the standalone RMSNorm library independent of MoE.
 */
void setMoeDecodeLookaheadCallback(MoeDecodeLookaheadCallback callback);
/** Register FFN gamma during prefill; predict at the selected decode boundary.
 * Input operands are read-only, including when the true norm is deferred.
 * NNTR_MOE_DECODE_PREFETCH=off|block|block_ffn|previous|previous_probe.
 * Unset disables it; previous_probe predicts without issuing reads.
 * NNTR_MOE_DECODE_PREFETCH_LIMIT=0..4 caps missing candidates to read.
 * NNTR_MOE_DECODE_PREFETCH_SELECTIVE=0 restores whole-batch waiting;
 * default 1 waits only for the actual selection and discards unfinished
 * guesses. Finished valid guesses are retained unless PREFETCH_KEEP_READY=0
 * (same NNTR_MOE_DECODE_ prefix). PREFETCH_ADAPTIVE=1 skips prediction when the
 * target's previous token required no reads, before prefetch collection.
 * PREFETCH_MMAP=1 (default) applies the demand mmap setting to decode reads;
 * prefill and preload retain their own read policy.
 * block uses the mixer's norm; block_ffn uses the target FFN gamma at the
 * SAME early boundary. previous uses the preceding MoE router's norm.
 */
void moeDecodeLookaheadNorm(const std::string &name, unsigned from, unsigned to,
                            const float *input, const float *addend,
                            const float *gamma, unsigned width, float epsilon);
} // namespace causallm
#endif
