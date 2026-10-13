// SPDX-License-Identifier: Apache-2.0
/** @file generation_prefill_tail.h
 * @brief Scoped last-row contract for a terminal, tokenwise LFM2 MoE suffix.
 * @author NNTrainer contributors
 * @bug See the repository issue tracker for known issues.
 */
#ifndef __CAUSALLM_GENERATION_PREFILL_TAIL_H__
#define __CAUSALLM_GENERATION_PREFILL_TAIL_H__

#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <string>
#include <utility>

namespace causallm {
/** A complete expert group must fit without changing FP32 split boundaries. */
bool generationPrefillTailFitsExpertPool(unsigned experts);

struct GenerationPrefillTailState {
  unsigned rows = 0;
  std::string norm_name, moe_name;
};

inline GenerationPrefillTailState &generationPrefillTailState() {
  static thread_local GenerationPrefillTailState state;
  return state;
}

/** Only the text-generation run opens this scope. Direct model inference,
 * training, all-token PPL and tensor-dump calls keep their full outputs.
 * A scope is thread-local because graph layers execute synchronously on
 * their owning inference thread; internal NPU workers do not use it. */
class GenerationPrefillTailScope {
public:
  static unsigned requestedRows() {
    const char *value = std::getenv("NNTR_LFM2_PREFILL_TAIL_ROWS");
    if (value && std::strcmp(value, "32") == 0)
      return 32;
    if (value && std::strcmp(value, "64") == 0)
      return 64;
    return 0;
  }

  static bool diagnostics() {
    if (std::getenv("NNTR_PARITY_DUMP")) {
      const char *logits_only = std::getenv("NNTR_PARITY_LOGITS_ONLY");
      if (!logits_only || std::strcmp(logits_only, "1") != 0)
        return true;
    }
    for (const char *name :
         {"NNTR_PPL", "NNTR_PPL_DECODE", "NNTR_PPL_DECODE_ALTS",
          "NNTR_HTP_DUMP", "NNTR_HTP_DUMP_ALL", "NNTR_MOE_LOOKAHEAD_DUMP",
          "NNTR_MOE_DIFF", "NNTR_MOE_SHADOW", "NNTR_MOE_TRACE", "NNTR_L2_DIFF",
          "NNTR_L2_SHADOW", "NNTR_CONV_BLOCK_DIFF", "NNTR_CONV_BLOCK_SHADOW",
          "NNTR_MOE_SPLIT", "NNTR_ACT_STATS"})
      if (std::getenv(name))
        return true;
    return false;
  }

  GenerationPrefillTailScope(const std::string &prefix, bool enabled) :
    previous_(std::move(generationPrefillTailState())) {
    auto &state = generationPrefillTailState();
    state = {};
    if (enabled && !diagnostics()) {
      state.rows = requestedRows();
      state.norm_name = prefix + "_ffn_norm";
      state.moe_name = prefix + "_ffn_down";
    }
  }
  ~GenerationPrefillTailScope() {
    generationPrefillTailState() = std::move(previous_);
  }
  GenerationPrefillTailScope(const GenerationPrefillTailScope &) = delete;
  GenerationPrefillTailScope &
  operator=(const GenerationPrefillTailScope &) = delete;

private:
  GenerationPrefillTailState previous_;
};

/** The public tensor keeps its original shape and last-row position.
 * Return the number of leading rows whose values this generation does not
 * consume. Restrict this to initial prefill; decode and continuation keep
 * their existing paths. */
inline unsigned generationPrefillTailOffset(const std::string &name,
                                            unsigned from, unsigned to,
                                            unsigned batch, bool training) {
  const auto &state = generationPrefillTailState();
  if (!state.rows || training || batch != 1 || from != 0 || to <= state.rows ||
      (name != state.norm_name && name != state.moe_name &&
       name != "output_norm"))
    return 0;
  return to - state.rows;
}
} // namespace causallm
#endif
