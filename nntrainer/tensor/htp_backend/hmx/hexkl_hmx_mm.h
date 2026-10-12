// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Haehun Yang <haehun.yang@ax.samsung.com>
 *
 * @file   hexkl_hmx_mm.h
 * @date   02 Oct 2026
 * @brief  The HMX matrix-multiply packet, issued inline
 * @see    https://github.com/nntrainer/nntrainer
 * @author Haehun Yang <haehun.yang@ax.samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * hexkl_micro_hmx_mm_u8i8() wraps a single instruction packet in a stack
 * frame, two alignment tests, the constants an error path needs and a
 * return code. Measured on v81 that is ~50 processor cycles for ~9 of
 * array time, and the quantized attention kernel issues 80k of them for
 * one 1024-row prefill of a Gemma-4 sliding layer -- half of the kernel's
 * HMX time is the wrapper. The packet itself, read out of
 * libhexkl_micro.a, is
 *
 *   { activation.ub = mxmem(act, 0x1f):cm
 *     weight.b      = mxmem(wt,  0x380) }        u8 x i8
 *   { activation.ub = mxmem(act, 0x1f):cm
 *     weight.n      = mxmem(wt,  0x180) }        u8 x i4
 *
 * where the two descriptors are compile-time constants in the library as
 * well. Issued inline, the compiler hoists both descriptors out of the
 * loop and keeps the tile pointers in registers, so a multiply costs the
 * packet and its address arithmetic.
 *
 * What the wrapper checked per call -- activation 2048-byte aligned,
 * weight 128 -- is a property of the tile layout rather than of the call:
 * hexkl_attn_q_plan places activations at multiples of
 * HEXKL_ATTN_TILE_BYTES (2048) and weights at multiples of 1024 (int8) or
 * 512 (int4). hexkl_hmx_mm_aligned() states that as one check per kernel
 * call instead of one per multiply.
 *
 * hexkl_hmx_acc_clear_int32() is the same argument for the clear that
 * opens every accumulation. hexkl_micro_hmx_acc_clear_int32() takes no
 * arguments and returns a code nobody reads, but it is still a call: the
 * compiler gives the caller a stack frame and spills what is live across
 * it. The a16 attention kernel clears twice per 16 multiplies, and on an
 * S26 Ultra (v81, 2112 MHz) issuing the packet inline measured qk -6.3 %
 * and pv -6.7 % on the 1024-row Gemma-4 sliding shape with the kernel's
 * output unchanged. The packet is the one the library issues:
 *
 *   mxclracc                                    int32 accumulator
 *
 * (the fp16 accumulator's is mxclracc.hf; this header has no caller for
 * it yet). The clear takes no memory operand, so there is nothing to
 * check per call and no host-side alignment question -- off the DSP it
 * stays the library call, which the HMX stand-in implements.
 */

#ifndef __NNTRAINER_HEXKL_HMX_MM_H__
#define __NNTRAINER_HEXKL_HMX_MM_H__

#include <stdint.h>

#include <hexkl_micro.h>

/** @brief Activation tile descriptor: 64 rows of 32 uint8. */
#define HEXKL_HMX_MM_ACT_DESC 0x1f
/** @brief Weight tile descriptor, 32x32 int8. */
#define HEXKL_HMX_MM_WT_DESC_I8 0x380
/** @brief Weight tile descriptor, 32x32 int4 (packed nibbles). */
#define HEXKL_HMX_MM_WT_DESC_I4 0x180

/**
 * @brief True when @a act and @a wt satisfy what the HMX load instructions
 *        require of every tile in a region.
 */
static inline int hexkl_hmx_mm_aligned(const void *act, const void *wt) {
#if defined(__hexagon__)
  return (((uintptr_t)act & (HEXKL_HMX_ACTIVATION_ALIGNMENT - 1u)) == 0u) &&
         (((uintptr_t)wt & (HEXKL_HMX_WEIGHTS_ALIGNMENT - 1u)) == 0u);
#else
  /* [#260 r3] Host builds (the in-process E2E, the host checks): the
     inline multiplies below are no-ops there, so the kernels must keep the
     library call, which the HMX stand-in implements. */
  (void)act;
  (void)wt;
  return 0;
#endif
}

#if defined(__hexagon__)

/** @brief The int32 accumulator back to zero, one packet. Requires the
 *         HMX lock. */
static inline void hexkl_hmx_acc_clear_int32(void) {
  __asm__ __volatile__("mxclracc" : : : "memory");
}

/**
 * @brief acc += act(64x32 uint8) . wt(32x32 int8), one packet.
 *
 * @param act 2048-byte aligned, 2 KiB
 * @param wt  128-byte aligned, 1 KiB
 */
static inline void hexkl_hmx_mm_u8i8(const void *act, const void *wt) {
  __asm__ __volatile__("{ activation.ub = mxmem(%0,%2):cm\n\t"
                       "  weight.b = mxmem(%1,%3) }"
                       :
                       : "r"(act), "r"(wt), "r"(HEXKL_HMX_MM_ACT_DESC),
                         "r"(HEXKL_HMX_MM_WT_DESC_I8)
                       : "memory");
}

/**
 * @brief acc += act(64x32 uint8) . wt(32x32 int4), one packet.
 *
 * @param act 2048-byte aligned, 2 KiB
 * @param wt  128-byte aligned, 512 B
 */
static inline void hexkl_hmx_mm_u8i4(const void *act, const void *wt) {
  __asm__ __volatile__("{ activation.ub = mxmem(%0,%2):cm\n\t"
                       "  weight.n = mxmem(%1,%3) }"
                       :
                       : "r"(act), "r"(wt), "r"(HEXKL_HMX_MM_ACT_DESC),
                         "r"(HEXKL_HMX_MM_WT_DESC_I4)
                       : "memory");
}

#else

/** @brief Host build: the library call, which the stand-in implements. */
static inline void hexkl_hmx_acc_clear_int32(void) {
  (void)hexkl_micro_hmx_acc_clear_int32();
}

/** @brief Host build: the library call, which takes base + offsets. */
static inline void hexkl_hmx_mm_u8i8(const void *act, const void *wt) {
  (void)act;
  (void)wt;
}
static inline void hexkl_hmx_mm_u8i4(const void *act, const void *wt) {
  (void)act;
  (void)wt;
}

#endif /* __hexagon__ */

#endif /* __NNTRAINER_HEXKL_HMX_MM_H__ */
