// SPDX-License-Identifier: Apache-2.0
#include "generation_prefill_tail.h"
#include <cassert>
#include <cstdio>
#include <stdexcept>
#include <thread>

unsigned offsetFromAnotherTranslationUnit(const char *, unsigned, unsigned,
                                          unsigned, bool);

int main() {
  using causallm::generationPrefillTailOffset;
  using causallm::GenerationPrefillTailScope;
  unsigned tests = 0;
  auto check = [&](bool ok) {
    ++tests;
    if (!ok)
      throw std::runtime_error("tail-scope contract failure");
  };
  const char *diagnostics[] = {"NNTR_PPL",
                               "NNTR_PPL_DECODE",
                               "NNTR_PPL_DECODE_ALTS",
                               "NNTR_HTP_DUMP",
                               "NNTR_HTP_DUMP_ALL",
                               "NNTR_MOE_LOOKAHEAD_DUMP",
                               "NNTR_MOE_DIFF",
                               "NNTR_MOE_SHADOW",
                               "NNTR_MOE_TRACE",
                               "NNTR_L2_DIFF",
                               "NNTR_L2_SHADOW",
                               "NNTR_CONV_BLOCK_DIFF",
                               "NNTR_CONV_BLOCK_SHADOW",
                               "NNTR_MOE_SPLIT",
                               "NNTR_ACT_STATS",
                               "NNTR_PARITY_DUMP",
                               "NNTR_PARITY_LOGITS_ONLY"};
  for (const char *env : diagnostics)
    unsetenv(env);
  unsetenv("NNTR_LFM2_PREFILL_TAIL_ROWS");
  check(GenerationPrefillTailScope::requestedRows() == 0);
  for (const char *value : {"", "0", "1", "4", "31", "33", "63", "65", "064",
                            "64 ", "-32", "32x", "1024"}) {
    setenv("NNTR_LFM2_PREFILL_TAIL_ROWS", value, 1);
    check(GenerationPrefillTailScope::requestedRows() == 0);
  }
  for (const char *value : {"32", "64"}) {
    const unsigned tail = std::atoi(value);
    setenv("NNTR_LFM2_PREFILL_TAIL_ROWS", value, 1);
    check(GenerationPrefillTailScope::requestedRows() == tail);
    check(generationPrefillTailOffset("output_norm", 0, 1024, 1, false) == 0);
    {
      GenerationPrefillTailScope scope("layer39", true);
      for (const char *name :
           {"layer39_ffn_norm", "layer39_ffn_down", "output_norm"}) {
        for (unsigned rows : {1u, 31u, 32u, 33u, 63u, 64u, 65u, 128u, 1023u,
                              1024u, 1025u, 2048u}) {
          const unsigned expected = rows > tail ? rows - tail : 0;
          check(generationPrefillTailOffset(name, 0, rows, 1, false) ==
                expected);
          check(offsetFromAnotherTranslationUnit(name, 0, rows, 1, false) ==
                expected);
          const unsigned offset =
            generationPrefillTailOffset(name, 0, rows, 1, false);
          check(offset + (rows - offset) == rows);
          check(generationPrefillTailOffset(name, 0, rows, 1, true) == 0);
          check(generationPrefillTailOffset(name, 0, rows, 2, false) == 0);
          check(generationPrefillTailOffset(name, 0, rows, 0, false) == 0);
          check(generationPrefillTailOffset(name, 1, rows + 1, 1, false) == 0);
        }
      }
      for (const char *name :
           {"layer38_ffn_norm", "layer38_ffn_down", "layer39_conv_norm",
            "layer39_conv_block", "lm_head", "layer39_ffn_norm_0", ""})
        check(generationPrefillTailOffset(name, 0, 1024, 1, false) == 0);
      {
        GenerationPrefillTailScope disabled("layer39", false);
        check(generationPrefillTailOffset("output_norm", 0, 1024, 1, false) ==
              0);
      }
      check(generationPrefillTailOffset("output_norm", 0, 1024, 1, false) ==
            1024 - tail);
      bool peer_inactive = false;
      std::thread peer([&] {
        peer_inactive = offsetFromAnotherTranslationUnit("output_norm", 0, 1024,
                                                         1, false) == 0;
      });
      peer.join();
      check(peer_inactive);
      try {
        GenerationPrefillTailScope nested("layer37", true);
        check(generationPrefillTailOffset("layer39_ffn_down", 0, 1024, 1,
                                          false) == 0);
        check(generationPrefillTailOffset("layer37_ffn_down", 0, 1024, 1,
                                          false) == 1024 - tail);
        throw std::logic_error("exercise exception unwind");
      } catch (const std::logic_error &) {
      }
      check(generationPrefillTailOffset("layer39_ffn_down", 0, 1024, 1,
                                        false) == 1024 - tail);
    }
    check(generationPrefillTailOffset("output_norm", 0, 1024, 1, false) == 0);
  }
  setenv("NNTR_LFM2_PREFILL_TAIL_ROWS", "64", 1);
  for (const char *env : diagnostics) {
    if (std::strcmp(env, "NNTR_PARITY_LOGITS_ONLY") == 0)
      continue;
    setenv(env, "0", 1);
    {
      GenerationPrefillTailScope scope("layer39", true);
      check(generationPrefillTailOffset("output_norm", 0, 1024, 1, false) == 0);
    }
    unsetenv(env);
  }
  setenv("NNTR_PARITY_DUMP", "/ignored/diagnostic/path", 1);
  for (const char *value : {"", "0", "01", "1", "true"}) {
    setenv("NNTR_PARITY_LOGITS_ONLY", value, 1);
    GenerationPrefillTailScope scope("layer39", true);
    check(generationPrefillTailOffset("output_norm", 0, 1024, 1, false) ==
          (std::strcmp(value, "1") == 0 ? 960u : 0u));
  }
  setenv("NNTR_PARITY_LOGITS_ONLY", "1", 1);
  setenv("NNTR_PPL", "1", 1);
  {
    GenerationPrefillTailScope scope("layer39", true);
    check(generationPrefillTailOffset("output_norm", 0, 1024, 1, false) == 0);
  }
  std::printf("PASS %u scope/offset/diagnostic/thread/TU checks\n", tests);
}
