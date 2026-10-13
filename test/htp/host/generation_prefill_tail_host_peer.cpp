// SPDX-License-Identifier: Apache-2.0
#include "generation_prefill_tail.h"

unsigned offsetFromAnotherTranslationUnit(const char *name, unsigned from,
                                          unsigned to, unsigned batch,
                                          bool training) {
  return causallm::generationPrefillTailOffset(name, from, to, batch, training);
}
