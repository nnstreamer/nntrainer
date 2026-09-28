// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jungwon Lee <jungone.lee@ax.samsung.com>
 *
 * @file   nntr_stack_chk.c
 * @date   28 Sep 2026
 * @brief  __stack_chk_* for the skel: the DSP image does not provide them
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jungwon Lee <jungone.lee@ax.samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * A HexKL object built with the stack protector leaves __stack_chk_guard
 * and __stack_chk_fail undefined, and the on-device loader rejects a skel
 * with undefined symbols (0x80000406, #97). The first one this tree pulls
 * in is hexkl_micro_hmx_rm_to_wh_i8, which the M=1 GEMV path references;
 * no source of ours asks for a canary (checked: no ssp attribute in any
 * translation unit's IR), so this is purely about what the archive
 * brings. build.sh's guard says to add the defining .c, and this is it.
 *
 * The guard is a constant. There is no entropy source in the skel and the
 * alternative is not linking the check at all; a constant canary still
 * catches a plain overrun, which is the failure this would ever see.
 */

#include <stdint.h>

/** @brief The canary HexKL's prologue stores and its epilogue compares. */
uintptr_t __stack_chk_guard = (uintptr_t)0x0b3a5f17u;

void __stack_chk_fail(void);

/** @brief Trap rather than abort(): abort() would be one more undefined
 *  symbol, and a trap is what a corrupted frame deserves on a DSP with no
 *  unwinder. */
void __stack_chk_fail(void) {
  for (;;) {
    __builtin_trap();
  }
}
