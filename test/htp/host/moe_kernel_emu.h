// SPDX-License-Identifier: Apache-2.0
/**
 * @file moe_kernel_emu.h
 * @brief The byte lookup used by the actual W2 expansion kernel.
 * @author NNTrainer contributors
 * @bug See the repository issue tracker for known issues.
 */
#ifndef NNTRAINER_MOE_KERNEL_EMU_H
#define NNTRAINER_MOE_KERNEL_EMU_H

#include <hvx_hexagon_protos.h>
#include <string.h>

/** The hardware mapping probe established the even-byte LUT indexing.
 * The test's expected output decodes the original palette independently. */
static inline HVX_Vector Q6_Vb_vlut32_VbVbI(HVX_Vector index, HVX_Vector table,
                                            int unused) {
  (void)unused;
  HVX_Vector out;
  uint8_t codes[128], values[128], result[128];
  memcpy(codes, &index, sizeof codes);
  memcpy(values, &table, sizeof values);
  for (unsigned i = 0; i < sizeof result; ++i)
    result[i] = values[(i & ~31u) + 2u * (codes[i] & 15u)];
  memcpy(&out, result, sizeof out);
  return out;
}

#endif
