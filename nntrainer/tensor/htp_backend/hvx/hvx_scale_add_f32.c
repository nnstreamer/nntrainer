// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file   hvx_scale_add_f32.c
 * @date   10 Sep 2026
 * @brief  dst += src * scale over a row, on HVX
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include "hvx_scale_add_f32.h"

#include <stddef.h>

#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>

#include "hvx_convert.h"

/** @brief f32 lanes per HVX vector at 128B. */
#define LANES 32u

void hvx_scale_add_scatter_tile_f32(float *out, const float *src,
                                    const uint32_t *rows, const float *weights,
                                    uint32_t m_count, uint32_t stride) {
  if ((((uintptr_t)out | (uintptr_t)src | (uintptr_t)stride * sizeof(float)) &
       (128u - 1u)) == 0u) {
    for (uint32_t m = 0; m < m_count; ++m) {
      HVX_Vector *dst = (HVX_Vector *)(out + (size_t)rows[m] * stride);
      const HVX_Vector s = ((const HVX_Vector *)(src + (size_t)m * stride))[0];
      dst[0] = Q6_Vsf_vadd_VsfVsf(
        dst[0], Q6_Vsf_vmpy_VsfVsf(s, hvx_splat_sf(weights[m])));
    }
    return;
  }
  for (uint32_t m = 0; m < m_count; ++m) {
    HVX_UVector *dst = (HVX_UVector *)(out + (size_t)rows[m] * stride);
    const HVX_Vector s = ((const HVX_UVector *)(src + (size_t)m * stride))[0];
    dst[0] = Q6_Vsf_vadd_VsfVsf(
      dst[0], Q6_Vsf_vmpy_VsfVsf(s, hvx_splat_sf(weights[m])));
  }
}

void hvx_scale_add_rows_f32(float *dst, const float *src, float scale,
                            uint32_t n) {
  const HVX_Vector wv = hvx_splat_sf(scale);
  uint32_t i = 0;

  /** Unaligned vector types throughout: src is a VTCM block and dst is a row
     of a heap allocation, and neither is promised 128-byte alignment. */
  for (; i + LANES <= n; i += LANES) {
    const HVX_Vector d = ((const HVX_UVector *)(dst + i))[0];
    const HVX_Vector s = ((const HVX_UVector *)(src + i))[0];
    ((HVX_UVector *)(dst + i))[0] =
      Q6_Vsf_vadd_VsfVsf(d, Q6_Vsf_vmpy_VsfVsf(s, wv));
  }
  /** Same two operations in the tail, so a row whose width is not a multiple
     of 32 does not get different arithmetic in its last few columns. */
  for (; i < n; ++i) {
    const float p = src[i] * scale;
    dst[i] = dst[i] + p;
  }
}
