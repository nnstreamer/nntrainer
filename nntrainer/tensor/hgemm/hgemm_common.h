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

#define A(i, j) a[(i)*lda + (j)]
#define B(i, j) b[(i)*ldb + (j)]
#define C(i, j) c[(i)*ldc + (j)]

#define GEMM_N (384)
#define GEMM_K (256)
#define GEMM_M (4096)
#define GEMM_UNROLLING_8 (8)
#define GEMM_UNROLLING_4 (4)
#define VL_FP16 (8)
#define VL_FP16_HALF (4)

/**
 * @todo Add macro for instructions in other CPU architectures
 */
