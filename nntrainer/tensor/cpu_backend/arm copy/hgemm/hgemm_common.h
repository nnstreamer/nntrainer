// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file   hgemm_common.h
 * @date   01 April 2024
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is common settings for hgemm
 *
 */

#define N_BLOCKING (768)
#define K_BLOCKING (256)
#define M_BLOCKING (4096)
#define GEMM_UNROLLING_16 (16)
#define GEMM_UNROLLING_8 (8)
#define GEMM_UNROLLING_4 (4)
#define GEMM_UNROLLING_1 (1)
#define VL_FP16 (8)
#define VL_FP16_HALF (4)
