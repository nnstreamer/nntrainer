// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   hvx_dequant_i32.c
 * @date   03 Aug 2026
 * @brief  int32 accumulator to f32 dequantization for the A8W4 path
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 */

#include <stddef.h>

#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>

#include "hexkl_acc_tile.h" /* HEXKL_ACC_TILE_COLS, for the width assert */
#include "hvx_convert.h"
#include "hvx_dequant_i32.h"

/** @brief HVX vector width in bytes (128B mode). */
#define VLEN 128u
/** @brief f32 or int32 lanes per HVX vector. */
#define LANES (VLEN / 4u)

void hvx_dequant_i32_to_f32(const int32_t *acc, uint32_t m_valid,
                            uint32_t m_pad, uint32_t n, const float *act_scale,
                            const int32_t *act_zp, const int32_t *colsum_w,
                            const float *w_scale, const float *bias,
                            float *out) {
  (void)m_pad;

  for (uint32_t m = 0; m < m_valid; ++m) {
    const int32_t *arow = acc + (size_t)m * n;
    float *orow = out + (size_t)m * n;
    const float s = act_scale[m];
    const int32_t z = act_zp[m];

    const uint32_t n_vec = n / LANES;
    const HVX_Vector vs = hvx_splat_sf(s);
    const HVX_Vector vzf = Q6_Vsf_equals_Vw(Q6_V_vsplat_R(z));

    const HVX_UVector *vacc = (const HVX_UVector *)arow;
    const HVX_UVector *vcs = (const HVX_UVector *)colsum_w;
    const HVX_UVector *vws = (const HVX_UVector *)w_scale;
    const HVX_UVector *vb = (const HVX_UVector *)bias;
    HVX_UVector *vout = (HVX_UVector *)orow;

    for (uint32_t v = 0; v < n_vec; ++v) {
      const HVX_Vector af = Q6_Vsf_equals_Vw(vacc[v]);
      const HVX_Vector csf = Q6_Vsf_equals_Vw(vcs[v]);
      const HVX_Vector corrected =
        Q6_Vsf_vsub_VsfVsf(af, Q6_Vsf_vmpy_VsfVsf(vzf, csf));
      const HVX_Vector scaled =
        Q6_Vsf_vmpy_VsfVsf(Q6_Vsf_vmpy_VsfVsf(corrected, vs), vws[v]);
      vout[v] = Q6_Vsf_vadd_VsfVsf(scaled, vb[v]);
    }

    for (uint32_t j = n_vec * LANES; j < n; ++j) {
      const int32_t corrected = arow[j] - z * colsum_w[j];
      orow[j] = (float)corrected * s * w_scale[j] + bias[j];
    }
  }
}

void hvx_dequant_acc_tile_to_f32(const int32_t *tile, uint32_t row_stride,
                                 uint32_t m_count, const float *act_scale,
                                 const int32_t *act_zp, const int32_t *colsum_w,
                                 const float *w_scale, const float *bias,
                                 float *out, uint32_t out_stride) {
  /* One vector per tile row, so the loop above's n_vec/tail split collapses.
     If a future part changes the tile width this stops being true, and the
     build should stop with it rather than silently emit 32 of n columns. */
  _Static_assert(HEXKL_ACC_TILE_COLS == LANES,
                 "an accumulator tile row must be exactly one HVX vector");

  const HVX_UVector *vcs = (const HVX_UVector *)colsum_w;
  const HVX_UVector *vws = (const HVX_UVector *)w_scale;
  const HVX_UVector *vb = (const HVX_UVector *)bias;
  /* Loop-invariant: the column tile is fixed, so this is the same value the
     row loop below would recompute every iteration. */
  const HVX_Vector csf = Q6_Vsf_equals_Vw(vcs[0]);

  for (uint32_t m = 0; m < m_count; ++m) {
    const HVX_UVector *vacc =
      (const HVX_UVector *)(tile + (size_t)m * row_stride);
    HVX_UVector *vout = (HVX_UVector *)(out + (size_t)m * out_stride);
    const HVX_Vector vs = hvx_splat_sf(act_scale[m]);
    const HVX_Vector vzf = Q6_Vsf_equals_Vw(Q6_V_vsplat_R(act_zp[m]));

    const HVX_Vector af = Q6_Vsf_equals_Vw(vacc[0]);
    const HVX_Vector corrected =
      Q6_Vsf_vsub_VsfVsf(af, Q6_Vsf_vmpy_VsfVsf(vzf, csf));
    const HVX_Vector scaled =
      Q6_Vsf_vmpy_VsfVsf(Q6_Vsf_vmpy_VsfVsf(corrected, vs), vws[0]);
    vout[0] = Q6_Vsf_vadd_VsfVsf(scaled, vb[0]);
  }
}
