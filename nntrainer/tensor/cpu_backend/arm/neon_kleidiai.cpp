// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Arm Limited and/or its affiliates
 * Copyright (C) 2024 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file   neon_kleidiai.cpp
 * @date   15 September 2025
 * @see    https://github.com/ARM-software/kleidiai
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Sungsik Kong
 *
 * @brief  Modified computational backend components of
 * kleidiai. Portions of this file are derived from Arm
 * Limited code licensed under the Apache License, Version 2.0, with
 * modifications
 *
 * @note   Licensed under the Apache License, Version 2.0 (the "License");
 *         you may not use this file except in compliance with the License.
 *         You may obtain a copy of the License at
 *             http://www.apache.org/licenses/LICENSE-2.0
 *
 * @modifications
 *   - [2025-09-15] Integrated and adapted Arm-provided code into
 *     nntrainer CPU backend
 *
 * @bug    No known bugs except for NYI items
 */
//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates
// <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include <cassert>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <neon_kleidiai.h>
#include <string>

// Include micro-kernel variants
#include "./kai/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_lhs_quant_pack_qai8dxp_f32.h"
#include "./kai/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod.h"
#include "./kai/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod.h"
#include "./kai/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x4_16x4x32_neon_dotprod.h"
#include "./kai/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x4_8x8x32_neon_dotprod.h"
#include "./kai/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp_qsi4cxp_interface.h"
#include "./kai/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_rhs_pack_kxn_qsi4cxp_qs4cxs1s0.h"
#include "./kai/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_rhs_pack_nxk_qsi4cxp_qs4cxs1s0.h"

#define INT4_MIN (-8)
#define INT4_MAX (7)
/**
 * @brief rhs_format
 *
 */
enum class rhs_format {
  nxk,
  kxn,
};

// Micro-kernel interface
/**
 * @brief kai_matmul_ukernel_f32_qa8dxp_qs4cxp
 *
 */
struct kai_matmul_ukernel_f32_qa8dxp_qs4cxp {
  kai_matmul_clamp_f32_qai8dxp_qsi4cxp_ukernel ukernel;
  std::string name = {};
};

kai_matmul_ukernel_f32_qa8dxp_qs4cxp ukernel_variants[] = {
  {kai_get_m_step_matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod,
   kai_get_n_step_matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod,
   kai_get_mr_matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod,
   kai_get_nr_matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod,
   kai_get_kr_matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod,
   kai_get_sr_matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod,
   kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod,
   kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod,
   kai_get_dst_offset_matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod,
   kai_get_dst_size_matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod,
   kai_run_matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod,
   "matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod"},
  {kai_get_m_step_matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod,
   kai_get_n_step_matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod,
   kai_get_mr_matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod,
   kai_get_nr_matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod,
   kai_get_kr_matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod,
   kai_get_sr_matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod,
   kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod,
   kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod,
   kai_get_dst_offset_matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod,
   kai_get_dst_size_matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod,
   kai_run_matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod,
   "matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod"},
  {kai_get_m_step_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x4_8x8x32_neon_dotprod,
   kai_get_n_step_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x4_8x8x32_neon_dotprod,
   kai_get_mr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x4_8x8x32_neon_dotprod,
   kai_get_nr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x4_8x8x32_neon_dotprod,
   kai_get_kr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x4_8x8x32_neon_dotprod,
   kai_get_sr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x4_8x8x32_neon_dotprod,
   kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x4_8x8x32_neon_dotprod,
   kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x4_8x8x32_neon_dotprod,
   kai_get_dst_offset_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x4_8x8x32_neon_dotprod,
   kai_get_dst_size_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x4_8x8x32_neon_dotprod,
   kai_run_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x4_8x8x32_neon_dotprod,
   "matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x4_8x8x32_neon_dotprod"},
  {kai_get_m_step_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x4_16x4x32_neon_dotprod,
   kai_get_n_step_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x4_16x4x32_neon_dotprod,
   kai_get_mr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x4_16x4x32_neon_dotprod,
   kai_get_nr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x4_16x4x32_neon_dotprod,
   kai_get_kr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x4_16x4x32_neon_dotprod,
   kai_get_sr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x4_16x4x32_neon_dotprod,
   kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x4_16x4x32_neon_dotprod,
   kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x4_16x4x32_neon_dotprod,
   kai_get_dst_offset_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x4_16x4x32_neon_dotprod,
   kai_get_dst_size_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x4_16x4x32_neon_dotprod,
   kai_run_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x4_16x4x32_neon_dotprod,
   "matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x4_16x4x32_neon_dotprod"},

};

static size_t roundup(size_t a, size_t b) { return ((a + b - 1) / b) * b; }

void nntr_gemm_qai8dxp_qsi4cxp_rtp(size_t m, size_t n, size_t k,
                                   void *lhs_native_mtx_f32,
                                   void *rhs_native_mtx_qs4cx,
                                   void *rhs_scales_f32, float *dst_act_mtx_f32,
                                   bool transB, float lower_bound,
                                   float upper_bound) {
  ///@todo check for optimal variant, or check for optimal variant config for
  /// specific M-N-K combination
  //   int idx_variant = 0;
  for (int idx_variant = 0; idx_variant < 4; idx_variant++) {
    rhs_format format = rhs_format::nxk;
    if (!transB) {
      format = rhs_format::kxn;
    }

    const size_t mr = ukernel_variants[idx_variant].ukernel.get_mr();
    const size_t nr = ukernel_variants[idx_variant].ukernel.get_nr();
    const size_t kr = ukernel_variants[idx_variant].ukernel.get_kr();
    const size_t sr = ukernel_variants[idx_variant].ukernel.get_sr();

    // Get the size in bytes for the packed matrices
    const size_t lhs_packed_size =
      kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32(m, k, mr, kr, sr);
    size_t rhs_packed_size = 0;

    if (format == rhs_format::nxk) {
      rhs_packed_size = kai_get_rhs_packed_size_rhs_pack_nxk_qsi4cxp_qs4cxs1s0(
        n, k, nr, kr, sr);

    } else {
      rhs_packed_size = kai_get_rhs_packed_size_rhs_pack_kxn_qsi4cxp_qs4cxs1s0(
        n, k, nr, kr, sr);
    }

    const size_t dst_size =
      ukernel_variants[idx_variant].ukernel.get_dst_size(m, n);

    // Allocate the matrices
    uint8_t *lhs_packed_mtx_qa8dx = new uint8_t[lhs_packed_size];
    uint8_t *rhs_packed_mtx_qs4cx = new uint8_t[rhs_packed_size];

    // If the RHS matrix contains constant values, the packing can be performed
    // only once
    if (format == rhs_format::nxk) {
      struct kai_rhs_pack_nxk_qsi4cxp_qs4cxs1s0_params nxk_params;

      nxk_params.lhs_zero_point = 1;
      nxk_params.rhs_zero_point = 8;
      // RHS packing
      kai_run_rhs_pack_nxk_qsi4cxp_qs4cxs1s0(
        1, n, k, nr, kr, sr,                     // Packing arguments
        (const uint8_t *)(rhs_native_mtx_qs4cx), // RHS
        NULL,                                    // Bias
        (const float *)(rhs_scales_f32),         // Scale
        rhs_packed_mtx_qs4cx,                    // RHS packed
        0, &nxk_params);

    } else {
      struct kai_rhs_pack_kxn_qsi4cxp_qs4cxs1s0_params kxn_params;
      kxn_params.lhs_zero_point = 1;
      kxn_params.rhs_zero_point = 8;
      // RHS packing
      kai_run_rhs_pack_kxn_qsi4cxp_qs4cxs1s0(
        1, n, k, nr, kr, sr,                     // Packing arguments
        (const uint8_t *)(rhs_native_mtx_qs4cx), // RHS
        NULL,                                    // Bias
        (const float *)(rhs_scales_f32),         // Scale
        rhs_packed_mtx_qs4cx,                    // RHS packed
        0, &kxn_params);
    }

    // LHS packing
    kai_run_lhs_quant_pack_qai8dxp_f32(m, k, mr, kr, sr, 0, // Packing arguments
                                       (const float *)lhs_native_mtx_f32, // LHS
                                       k * sizeof(float),     // LHS stride
                                       lhs_packed_mtx_qa8dx); // LHS packed

    {
      const size_t dst_stride = n * sizeof(float);
      const size_t lhs_offset =
        ukernel_variants[idx_variant].ukernel.get_lhs_packed_offset(0, k);
      const size_t rhs_offset =
        ukernel_variants[idx_variant].ukernel.get_rhs_packed_offset(0, k);
      const size_t dst_offset =
        ukernel_variants[idx_variant].ukernel.get_dst_offset(0, 0, dst_stride);

      const void *lhs_ptr =
        (const void *)((const char *)lhs_packed_mtx_qa8dx + lhs_offset);
      const void *rhs_ptr =
        (const void *)((const char *)rhs_packed_mtx_qs4cx + rhs_offset);
      float *dst_ptr = (float *)((uint8_t *)dst_act_mtx_f32 + dst_offset);

      ukernel_variants[idx_variant].ukernel.run_matmul(
        m, n, k,                 // Dimensions
        lhs_ptr,                 // LHS packed
        rhs_ptr,                 // RHS packed
        dst_ptr,                 // DST
        dst_stride,              // DST stride (row)
        sizeof(float),           // DST stride (col)
        lower_bound, upper_bound // Min and max for the clamp operation
      );
    }

    delete[] lhs_packed_mtx_qa8dx;
    delete[] rhs_packed_mtx_qs4cx;
  }
}
