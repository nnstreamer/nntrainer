// SPDX-License-Identifier: Apache-2.0
/** @file htp_qs2_gemv_opts.h
 * @brief Opt-in QS2 decode kernels; bits 27..29 of the shared options.
 */
#ifndef __NNTRAINER_HTP_QS2_GEMV_OPTS_H__
#define __NNTRAINER_HTP_QS2_GEMV_OPTS_H__
#include <stdint.h>
#include <stdlib.h>
#define HTP_QS2_GEMV_SHIFT 27u
#define HTP_QS2_GEMV_MASK 0x38000000u
/** NNTR_HTP_QS2_GEMV: 0=existing expand, 1=register expansion, 2=pair LUT
 * with in-place HVX layout conversion, 3=shared-register panel pairs,
 * 4=panel pairs with two concurrent experts, 5=centered exact pair LUT,
 * 6=dense four-table LUT, 7=dense LUT with offline-prepared weights.
 * Off until explicitly selected. */
static inline uint32_t htp_qs2_gemv_flags(const char *env) {
  const int mode = env ? atoi(env) : 0;
  return mode >= 1 && mode <= 7 ? (uint32_t)mode << HTP_QS2_GEMV_SHIFT : 0u;
}
#endif
