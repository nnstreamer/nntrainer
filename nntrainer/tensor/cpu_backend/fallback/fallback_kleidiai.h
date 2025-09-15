// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Arm Limited and/or its affiliates
 * Copyright (C) 2024 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file   fallback_kleidiai.h
 * @date   15 September 2025
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Sungsik Kong
 *
 * @brief  Modified computational backend components of matmul_clamp_f32_qai8dxp_qsi4cxp.
 *         Portions of this file are derived from Arm Limited code licensed
 *         under the Apache License, Version 2.0, with modifications
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

#include <cstdint>
#include <stddef.h>

enum class rhs_format {
  nxk,
  kxn,
};

void quant_qs4cx_f32(size_t n, size_t k, rhs_format format,
                     const float *rhs_f32, uint8_t *rhs_qs4cx,
                     float *rhs_scales_f32);

void ref_quant_qa8dx_f32(size_t m, size_t k, const float *lhs_f32,
                         int8_t *lhs_qa8dx);

void ref_matmul_f32_qa8dx_qs4cx(size_t m, size_t n, size_t k, rhs_format format,
                                const int8_t *lhs_qa8dx,
                                const uint8_t *rhs_qs4cx,
                                const float *rhs_scales_f32, float *dst_f32,
                                float scalar_min, float scalar_max);