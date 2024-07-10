// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file   hgemm_kernel_8x16.cpp
 * @date   04 April 2024
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is half-precision GEMM 8x16 kernel
 *
 */

#include <arm_neon.h>
#include <assert.h>
#include <hgemm_kernel.h>
#include <stdlib.h>

#define INIT_KERNEL_8X16()       \
  do {                           \
    v0_7 = vdupq_n_f16(0.F);     \
    v8_15 = vdupq_n_f16(0.F);    \
    v16_23 = vdupq_n_f16(0.F);   \
    v24_31 = vdupq_n_f16(0.F);   \
    v32_39 = vdupq_n_f16(0.F);   \
    v40_47 = vdupq_n_f16(0.F);   \
    v48_55 = vdupq_n_f16(0.F);   \
    v56_63 = vdupq_n_f16(0.F);   \
    v64_71 = vdupq_n_f16(0.F);   \
    v72_79 = vdupq_n_f16(0.F);   \
    v80_87 = vdupq_n_f16(0.F);   \
    v88_95 = vdupq_n_f16(0.F);   \
    v96_103 = vdupq_n_f16(0.F);  \
    v104_111 = vdupq_n_f16(0.F); \
    v112_119 = vdupq_n_f16(0.F); \
    v120_127 = vdupq_n_f16(0.F); \
  } while (0)

// 1. Partial sum 2048 digits
#define KERNEL_8x16_ACC16()                            \
  do {                                                 \
    va0 = vld1q_f16(a + 8 * 0);                        \
    vb1 = vld1q_f16(b + 8 * 0);                        \
    vb2 = vld1q_f16(b + 8 * 1);                        \
    v0_7 = vfmaq_laneq_f16(v0_7, vb1, va0, 0);         \
    v8_15 = vfmaq_laneq_f16(v8_15, vb1, va0, 1);       \
    v16_23 = vfmaq_laneq_f16(v16_23, vb1, va0, 2);     \
    v24_31 = vfmaq_laneq_f16(v24_31, vb1, va0, 3);     \
    v32_39 = vfmaq_laneq_f16(v32_39, vb1, va0, 4);     \
    v40_47 = vfmaq_laneq_f16(v40_47, vb1, va0, 5);     \
    v48_55 = vfmaq_laneq_f16(v48_55, vb1, va0, 6);     \
    v56_63 = vfmaq_laneq_f16(v56_63, vb1, va0, 7);     \
    v64_71 = vfmaq_laneq_f16(v64_71, vb2, va0, 0);     \
    v72_79 = vfmaq_laneq_f16(v72_79, vb2, va0, 1);     \
    v80_87 = vfmaq_laneq_f16(v80_87, vb2, va0, 2);     \
    v88_95 = vfmaq_laneq_f16(v88_95, vb2, va0, 3);     \
    v96_103 = vfmaq_laneq_f16(v96_103, vb2, va0, 4);   \
    v104_111 = vfmaq_laneq_f16(v104_111, vb2, va0, 5); \
    v112_119 = vfmaq_laneq_f16(v112_119, vb2, va0, 6); \
    v120_127 = vfmaq_laneq_f16(v120_127, vb2, va0, 7); \
    va0 = vld1q_f16(a + 8 * 1);                        \
    vb1 = vld1q_f16(b + 8 * 2);                        \
    vb2 = vld1q_f16(b + 8 * 3);                        \
    v0_7 = vfmaq_laneq_f16(v0_7, vb1, va0, 0);         \
    v8_15 = vfmaq_laneq_f16(v8_15, vb1, va0, 1);       \
    v16_23 = vfmaq_laneq_f16(v16_23, vb1, va0, 2);     \
    v24_31 = vfmaq_laneq_f16(v24_31, vb1, va0, 3);     \
    v32_39 = vfmaq_laneq_f16(v32_39, vb1, va0, 4);     \
    v40_47 = vfmaq_laneq_f16(v40_47, vb1, va0, 5);     \
    v48_55 = vfmaq_laneq_f16(v48_55, vb1, va0, 6);     \
    v56_63 = vfmaq_laneq_f16(v56_63, vb1, va0, 7);     \
    v64_71 = vfmaq_laneq_f16(v64_71, vb2, va0, 0);     \
    v72_79 = vfmaq_laneq_f16(v72_79, vb2, va0, 1);     \
    v80_87 = vfmaq_laneq_f16(v80_87, vb2, va0, 2);     \
    v88_95 = vfmaq_laneq_f16(v88_95, vb2, va0, 3);     \
    v96_103 = vfmaq_laneq_f16(v96_103, vb2, va0, 4);   \
    v104_111 = vfmaq_laneq_f16(v104_111, vb2, va0, 5); \
    v112_119 = vfmaq_laneq_f16(v112_119, vb2, va0, 6); \
    v120_127 = vfmaq_laneq_f16(v120_127, vb2, va0, 7); \
    va0 = vld1q_f16(a + 8 * 2);                        \
    vb1 = vld1q_f16(b + 8 * 4);                        \
    vb2 = vld1q_f16(b + 8 * 5);                        \
    v0_7 = vfmaq_laneq_f16(v0_7, vb1, va0, 0);         \
    v8_15 = vfmaq_laneq_f16(v8_15, vb1, va0, 1);       \
    v16_23 = vfmaq_laneq_f16(v16_23, vb1, va0, 2);     \
    v24_31 = vfmaq_laneq_f16(v24_31, vb1, va0, 3);     \
    v32_39 = vfmaq_laneq_f16(v32_39, vb1, va0, 4);     \
    v40_47 = vfmaq_laneq_f16(v40_47, vb1, va0, 5);     \
    v48_55 = vfmaq_laneq_f16(v48_55, vb1, va0, 6);     \
    v56_63 = vfmaq_laneq_f16(v56_63, vb1, va0, 7);     \
    v64_71 = vfmaq_laneq_f16(v64_71, vb2, va0, 0);     \
    v72_79 = vfmaq_laneq_f16(v72_79, vb2, va0, 1);     \
    v80_87 = vfmaq_laneq_f16(v80_87, vb2, va0, 2);     \
    v88_95 = vfmaq_laneq_f16(v88_95, vb2, va0, 3);     \
    v96_103 = vfmaq_laneq_f16(v96_103, vb2, va0, 4);   \
    v104_111 = vfmaq_laneq_f16(v104_111, vb2, va0, 5); \
    v112_119 = vfmaq_laneq_f16(v112_119, vb2, va0, 6); \
    v120_127 = vfmaq_laneq_f16(v120_127, vb2, va0, 7); \
    va0 = vld1q_f16(a + 8 * 3);                        \
    vb1 = vld1q_f16(b + 8 * 6);                        \
    vb2 = vld1q_f16(b + 8 * 7);                        \
    v0_7 = vfmaq_laneq_f16(v0_7, vb1, va0, 0);         \
    v8_15 = vfmaq_laneq_f16(v8_15, vb1, va0, 1);       \
    v16_23 = vfmaq_laneq_f16(v16_23, vb1, va0, 2);     \
    v24_31 = vfmaq_laneq_f16(v24_31, vb1, va0, 3);     \
    v32_39 = vfmaq_laneq_f16(v32_39, vb1, va0, 4);     \
    v40_47 = vfmaq_laneq_f16(v40_47, vb1, va0, 5);     \
    v48_55 = vfmaq_laneq_f16(v48_55, vb1, va0, 6);     \
    v56_63 = vfmaq_laneq_f16(v56_63, vb1, va0, 7);     \
    v64_71 = vfmaq_laneq_f16(v64_71, vb2, va0, 0);     \
    v72_79 = vfmaq_laneq_f16(v72_79, vb2, va0, 1);     \
    v80_87 = vfmaq_laneq_f16(v80_87, vb2, va0, 2);     \
    v88_95 = vfmaq_laneq_f16(v88_95, vb2, va0, 3);     \
    v96_103 = vfmaq_laneq_f16(v96_103, vb2, va0, 4);   \
    v104_111 = vfmaq_laneq_f16(v104_111, vb2, va0, 5); \
    v112_119 = vfmaq_laneq_f16(v112_119, vb2, va0, 6); \
    v120_127 = vfmaq_laneq_f16(v120_127, vb2, va0, 7); \
    va0 = vld1q_f16(a + 8 * 4);                        \
    vb1 = vld1q_f16(b + 8 * 8);                        \
    vb2 = vld1q_f16(b + 8 * 9);                        \
    v0_7 = vfmaq_laneq_f16(v0_7, vb1, va0, 0);         \
    v8_15 = vfmaq_laneq_f16(v8_15, vb1, va0, 1);       \
    v16_23 = vfmaq_laneq_f16(v16_23, vb1, va0, 2);     \
    v24_31 = vfmaq_laneq_f16(v24_31, vb1, va0, 3);     \
    v32_39 = vfmaq_laneq_f16(v32_39, vb1, va0, 4);     \
    v40_47 = vfmaq_laneq_f16(v40_47, vb1, va0, 5);     \
    v48_55 = vfmaq_laneq_f16(v48_55, vb1, va0, 6);     \
    v56_63 = vfmaq_laneq_f16(v56_63, vb1, va0, 7);     \
    v64_71 = vfmaq_laneq_f16(v64_71, vb2, va0, 0);     \
    v72_79 = vfmaq_laneq_f16(v72_79, vb2, va0, 1);     \
    v80_87 = vfmaq_laneq_f16(v80_87, vb2, va0, 2);     \
    v88_95 = vfmaq_laneq_f16(v88_95, vb2, va0, 3);     \
    v96_103 = vfmaq_laneq_f16(v96_103, vb2, va0, 4);   \
    v104_111 = vfmaq_laneq_f16(v104_111, vb2, va0, 5); \
    v112_119 = vfmaq_laneq_f16(v112_119, vb2, va0, 6); \
    v120_127 = vfmaq_laneq_f16(v120_127, vb2, va0, 7); \
    va0 = vld1q_f16(a + 8 * 5);                        \
    vb1 = vld1q_f16(b + 8 * 10);                       \
    vb2 = vld1q_f16(b + 8 * 11);                       \
    v0_7 = vfmaq_laneq_f16(v0_7, vb1, va0, 0);         \
    v8_15 = vfmaq_laneq_f16(v8_15, vb1, va0, 1);       \
    v16_23 = vfmaq_laneq_f16(v16_23, vb1, va0, 2);     \
    v24_31 = vfmaq_laneq_f16(v24_31, vb1, va0, 3);     \
    v32_39 = vfmaq_laneq_f16(v32_39, vb1, va0, 4);     \
    v40_47 = vfmaq_laneq_f16(v40_47, vb1, va0, 5);     \
    v48_55 = vfmaq_laneq_f16(v48_55, vb1, va0, 6);     \
    v56_63 = vfmaq_laneq_f16(v56_63, vb1, va0, 7);     \
    v64_71 = vfmaq_laneq_f16(v64_71, vb2, va0, 0);     \
    v72_79 = vfmaq_laneq_f16(v72_79, vb2, va0, 1);     \
    v80_87 = vfmaq_laneq_f16(v80_87, vb2, va0, 2);     \
    v88_95 = vfmaq_laneq_f16(v88_95, vb2, va0, 3);     \
    v96_103 = vfmaq_laneq_f16(v96_103, vb2, va0, 4);   \
    v104_111 = vfmaq_laneq_f16(v104_111, vb2, va0, 5); \
    v112_119 = vfmaq_laneq_f16(v112_119, vb2, va0, 6); \
    v120_127 = vfmaq_laneq_f16(v120_127, vb2, va0, 7); \
    va0 = vld1q_f16(a + 8 * 6);                        \
    vb1 = vld1q_f16(b + 8 * 12);                       \
    vb2 = vld1q_f16(b + 8 * 13);                       \
    v0_7 = vfmaq_laneq_f16(v0_7, vb1, va0, 0);         \
    v8_15 = vfmaq_laneq_f16(v8_15, vb1, va0, 1);       \
    v16_23 = vfmaq_laneq_f16(v16_23, vb1, va0, 2);     \
    v24_31 = vfmaq_laneq_f16(v24_31, vb1, va0, 3);     \
    v32_39 = vfmaq_laneq_f16(v32_39, vb1, va0, 4);     \
    v40_47 = vfmaq_laneq_f16(v40_47, vb1, va0, 5);     \
    v48_55 = vfmaq_laneq_f16(v48_55, vb1, va0, 6);     \
    v56_63 = vfmaq_laneq_f16(v56_63, vb1, va0, 7);     \
    v64_71 = vfmaq_laneq_f16(v64_71, vb2, va0, 0);     \
    v72_79 = vfmaq_laneq_f16(v72_79, vb2, va0, 1);     \
    v80_87 = vfmaq_laneq_f16(v80_87, vb2, va0, 2);     \
    v88_95 = vfmaq_laneq_f16(v88_95, vb2, va0, 3);     \
    v96_103 = vfmaq_laneq_f16(v96_103, vb2, va0, 4);   \
    v104_111 = vfmaq_laneq_f16(v104_111, vb2, va0, 5); \
    v112_119 = vfmaq_laneq_f16(v112_119, vb2, va0, 6); \
    v120_127 = vfmaq_laneq_f16(v120_127, vb2, va0, 7); \
    va0 = vld1q_f16(a + 8 * 7);                        \
    vb1 = vld1q_f16(b + 8 * 14);                       \
    vb2 = vld1q_f16(b + 8 * 15);                       \
    v0_7 = vfmaq_laneq_f16(v0_7, vb1, va0, 0);         \
    v8_15 = vfmaq_laneq_f16(v8_15, vb1, va0, 1);       \
    v16_23 = vfmaq_laneq_f16(v16_23, vb1, va0, 2);     \
    v24_31 = vfmaq_laneq_f16(v24_31, vb1, va0, 3);     \
    v32_39 = vfmaq_laneq_f16(v32_39, vb1, va0, 4);     \
    v40_47 = vfmaq_laneq_f16(v40_47, vb1, va0, 5);     \
    v48_55 = vfmaq_laneq_f16(v48_55, vb1, va0, 6);     \
    v56_63 = vfmaq_laneq_f16(v56_63, vb1, va0, 7);     \
    v64_71 = vfmaq_laneq_f16(v64_71, vb2, va0, 0);     \
    v72_79 = vfmaq_laneq_f16(v72_79, vb2, va0, 1);     \
    v80_87 = vfmaq_laneq_f16(v80_87, vb2, va0, 2);     \
    v88_95 = vfmaq_laneq_f16(v88_95, vb2, va0, 3);     \
    v96_103 = vfmaq_laneq_f16(v96_103, vb2, va0, 4);   \
    v104_111 = vfmaq_laneq_f16(v104_111, vb2, va0, 5); \
    v112_119 = vfmaq_laneq_f16(v112_119, vb2, va0, 6); \
    v120_127 = vfmaq_laneq_f16(v120_127, vb2, va0, 7); \
    va0 = vld1q_f16(a + 8 * 8);                        \
    vb1 = vld1q_f16(b + 8 * 16);                       \
    vb2 = vld1q_f16(b + 8 * 17);                       \
    v0_7 = vfmaq_laneq_f16(v0_7, vb1, va0, 0);         \
    v8_15 = vfmaq_laneq_f16(v8_15, vb1, va0, 1);       \
    v16_23 = vfmaq_laneq_f16(v16_23, vb1, va0, 2);     \
    v24_31 = vfmaq_laneq_f16(v24_31, vb1, va0, 3);     \
    v32_39 = vfmaq_laneq_f16(v32_39, vb1, va0, 4);     \
    v40_47 = vfmaq_laneq_f16(v40_47, vb1, va0, 5);     \
    v48_55 = vfmaq_laneq_f16(v48_55, vb1, va0, 6);     \
    v56_63 = vfmaq_laneq_f16(v56_63, vb1, va0, 7);     \
    v64_71 = vfmaq_laneq_f16(v64_71, vb2, va0, 0);     \
    v72_79 = vfmaq_laneq_f16(v72_79, vb2, va0, 1);     \
    v80_87 = vfmaq_laneq_f16(v80_87, vb2, va0, 2);     \
    v88_95 = vfmaq_laneq_f16(v88_95, vb2, va0, 3);     \
    v96_103 = vfmaq_laneq_f16(v96_103, vb2, va0, 4);   \
    v104_111 = vfmaq_laneq_f16(v104_111, vb2, va0, 5); \
    v112_119 = vfmaq_laneq_f16(v112_119, vb2, va0, 6); \
    v120_127 = vfmaq_laneq_f16(v120_127, vb2, va0, 7); \
    va0 = vld1q_f16(a + 8 * 9);                        \
    vb1 = vld1q_f16(b + 8 * 18);                       \
    vb2 = vld1q_f16(b + 8 * 19);                       \
    v0_7 = vfmaq_laneq_f16(v0_7, vb1, va0, 0);         \
    v8_15 = vfmaq_laneq_f16(v8_15, vb1, va0, 1);       \
    v16_23 = vfmaq_laneq_f16(v16_23, vb1, va0, 2);     \
    v24_31 = vfmaq_laneq_f16(v24_31, vb1, va0, 3);     \
    v32_39 = vfmaq_laneq_f16(v32_39, vb1, va0, 4);     \
    v40_47 = vfmaq_laneq_f16(v40_47, vb1, va0, 5);     \
    v48_55 = vfmaq_laneq_f16(v48_55, vb1, va0, 6);     \
    v56_63 = vfmaq_laneq_f16(v56_63, vb1, va0, 7);     \
    v64_71 = vfmaq_laneq_f16(v64_71, vb2, va0, 0);     \
    v72_79 = vfmaq_laneq_f16(v72_79, vb2, va0, 1);     \
    v80_87 = vfmaq_laneq_f16(v80_87, vb2, va0, 2);     \
    v88_95 = vfmaq_laneq_f16(v88_95, vb2, va0, 3);     \
    v96_103 = vfmaq_laneq_f16(v96_103, vb2, va0, 4);   \
    v104_111 = vfmaq_laneq_f16(v104_111, vb2, va0, 5); \
    v112_119 = vfmaq_laneq_f16(v112_119, vb2, va0, 6); \
    v120_127 = vfmaq_laneq_f16(v120_127, vb2, va0, 7); \
    va0 = vld1q_f16(a + 8 * 10);                       \
    vb1 = vld1q_f16(b + 8 * 20);                       \
    vb2 = vld1q_f16(b + 8 * 21);                       \
    v0_7 = vfmaq_laneq_f16(v0_7, vb1, va0, 0);         \
    v8_15 = vfmaq_laneq_f16(v8_15, vb1, va0, 1);       \
    v16_23 = vfmaq_laneq_f16(v16_23, vb1, va0, 2);     \
    v24_31 = vfmaq_laneq_f16(v24_31, vb1, va0, 3);     \
    v32_39 = vfmaq_laneq_f16(v32_39, vb1, va0, 4);     \
    v40_47 = vfmaq_laneq_f16(v40_47, vb1, va0, 5);     \
    v48_55 = vfmaq_laneq_f16(v48_55, vb1, va0, 6);     \
    v56_63 = vfmaq_laneq_f16(v56_63, vb1, va0, 7);     \
    v64_71 = vfmaq_laneq_f16(v64_71, vb2, va0, 0);     \
    v72_79 = vfmaq_laneq_f16(v72_79, vb2, va0, 1);     \
    v80_87 = vfmaq_laneq_f16(v80_87, vb2, va0, 2);     \
    v88_95 = vfmaq_laneq_f16(v88_95, vb2, va0, 3);     \
    v96_103 = vfmaq_laneq_f16(v96_103, vb2, va0, 4);   \
    v104_111 = vfmaq_laneq_f16(v104_111, vb2, va0, 5); \
    v112_119 = vfmaq_laneq_f16(v112_119, vb2, va0, 6); \
    v120_127 = vfmaq_laneq_f16(v120_127, vb2, va0, 7); \
    va0 = vld1q_f16(a + 8 * 11);                       \
    vb1 = vld1q_f16(b + 8 * 22);                       \
    vb2 = vld1q_f16(b + 8 * 23);                       \
    v0_7 = vfmaq_laneq_f16(v0_7, vb1, va0, 0);         \
    v8_15 = vfmaq_laneq_f16(v8_15, vb1, va0, 1);       \
    v16_23 = vfmaq_laneq_f16(v16_23, vb1, va0, 2);     \
    v24_31 = vfmaq_laneq_f16(v24_31, vb1, va0, 3);     \
    v32_39 = vfmaq_laneq_f16(v32_39, vb1, va0, 4);     \
    v40_47 = vfmaq_laneq_f16(v40_47, vb1, va0, 5);     \
    v48_55 = vfmaq_laneq_f16(v48_55, vb1, va0, 6);     \
    v56_63 = vfmaq_laneq_f16(v56_63, vb1, va0, 7);     \
    v64_71 = vfmaq_laneq_f16(v64_71, vb2, va0, 0);     \
    v72_79 = vfmaq_laneq_f16(v72_79, vb2, va0, 1);     \
    v80_87 = vfmaq_laneq_f16(v80_87, vb2, va0, 2);     \
    v88_95 = vfmaq_laneq_f16(v88_95, vb2, va0, 3);     \
    v96_103 = vfmaq_laneq_f16(v96_103, vb2, va0, 4);   \
    v104_111 = vfmaq_laneq_f16(v104_111, vb2, va0, 5); \
    v112_119 = vfmaq_laneq_f16(v112_119, vb2, va0, 6); \
    v120_127 = vfmaq_laneq_f16(v120_127, vb2, va0, 7); \
    va0 = vld1q_f16(a + 8 * 12);                       \
    vb1 = vld1q_f16(b + 8 * 24);                       \
    vb2 = vld1q_f16(b + 8 * 25);                       \
    v0_7 = vfmaq_laneq_f16(v0_7, vb1, va0, 0);         \
    v8_15 = vfmaq_laneq_f16(v8_15, vb1, va0, 1);       \
    v16_23 = vfmaq_laneq_f16(v16_23, vb1, va0, 2);     \
    v24_31 = vfmaq_laneq_f16(v24_31, vb1, va0, 3);     \
    v32_39 = vfmaq_laneq_f16(v32_39, vb1, va0, 4);     \
    v40_47 = vfmaq_laneq_f16(v40_47, vb1, va0, 5);     \
    v48_55 = vfmaq_laneq_f16(v48_55, vb1, va0, 6);     \
    v56_63 = vfmaq_laneq_f16(v56_63, vb1, va0, 7);     \
    v64_71 = vfmaq_laneq_f16(v64_71, vb2, va0, 0);     \
    v72_79 = vfmaq_laneq_f16(v72_79, vb2, va0, 1);     \
    v80_87 = vfmaq_laneq_f16(v80_87, vb2, va0, 2);     \
    v88_95 = vfmaq_laneq_f16(v88_95, vb2, va0, 3);     \
    v96_103 = vfmaq_laneq_f16(v96_103, vb2, va0, 4);   \
    v104_111 = vfmaq_laneq_f16(v104_111, vb2, va0, 5); \
    v112_119 = vfmaq_laneq_f16(v112_119, vb2, va0, 6); \
    v120_127 = vfmaq_laneq_f16(v120_127, vb2, va0, 7); \
    va0 = vld1q_f16(a + 8 * 13);                       \
    vb1 = vld1q_f16(b + 8 * 26);                       \
    vb2 = vld1q_f16(b + 8 * 27);                       \
    v0_7 = vfmaq_laneq_f16(v0_7, vb1, va0, 0);         \
    v8_15 = vfmaq_laneq_f16(v8_15, vb1, va0, 1);       \
    v16_23 = vfmaq_laneq_f16(v16_23, vb1, va0, 2);     \
    v24_31 = vfmaq_laneq_f16(v24_31, vb1, va0, 3);     \
    v32_39 = vfmaq_laneq_f16(v32_39, vb1, va0, 4);     \
    v40_47 = vfmaq_laneq_f16(v40_47, vb1, va0, 5);     \
    v48_55 = vfmaq_laneq_f16(v48_55, vb1, va0, 6);     \
    v56_63 = vfmaq_laneq_f16(v56_63, vb1, va0, 7);     \
    v64_71 = vfmaq_laneq_f16(v64_71, vb2, va0, 0);     \
    v72_79 = vfmaq_laneq_f16(v72_79, vb2, va0, 1);     \
    v80_87 = vfmaq_laneq_f16(v80_87, vb2, va0, 2);     \
    v88_95 = vfmaq_laneq_f16(v88_95, vb2, va0, 3);     \
    v96_103 = vfmaq_laneq_f16(v96_103, vb2, va0, 4);   \
    v104_111 = vfmaq_laneq_f16(v104_111, vb2, va0, 5); \
    v112_119 = vfmaq_laneq_f16(v112_119, vb2, va0, 6); \
    v120_127 = vfmaq_laneq_f16(v120_127, vb2, va0, 7); \
    va0 = vld1q_f16(a + 8 * 14);                       \
    vb1 = vld1q_f16(b + 8 * 28);                       \
    vb2 = vld1q_f16(b + 8 * 29);                       \
    v0_7 = vfmaq_laneq_f16(v0_7, vb1, va0, 0);         \
    v8_15 = vfmaq_laneq_f16(v8_15, vb1, va0, 1);       \
    v16_23 = vfmaq_laneq_f16(v16_23, vb1, va0, 2);     \
    v24_31 = vfmaq_laneq_f16(v24_31, vb1, va0, 3);     \
    v32_39 = vfmaq_laneq_f16(v32_39, vb1, va0, 4);     \
    v40_47 = vfmaq_laneq_f16(v40_47, vb1, va0, 5);     \
    v48_55 = vfmaq_laneq_f16(v48_55, vb1, va0, 6);     \
    v56_63 = vfmaq_laneq_f16(v56_63, vb1, va0, 7);     \
    v64_71 = vfmaq_laneq_f16(v64_71, vb2, va0, 0);     \
    v72_79 = vfmaq_laneq_f16(v72_79, vb2, va0, 1);     \
    v80_87 = vfmaq_laneq_f16(v80_87, vb2, va0, 2);     \
    v88_95 = vfmaq_laneq_f16(v88_95, vb2, va0, 3);     \
    v96_103 = vfmaq_laneq_f16(v96_103, vb2, va0, 4);   \
    v104_111 = vfmaq_laneq_f16(v104_111, vb2, va0, 5); \
    v112_119 = vfmaq_laneq_f16(v112_119, vb2, va0, 6); \
    v120_127 = vfmaq_laneq_f16(v120_127, vb2, va0, 7); \
    va0 = vld1q_f16(a + 8 * 15);                       \
    vb1 = vld1q_f16(b + 8 * 30);                       \
    vb2 = vld1q_f16(b + 8 * 31);                       \
    v0_7 = vfmaq_laneq_f16(v0_7, vb1, va0, 0);         \
    v8_15 = vfmaq_laneq_f16(v8_15, vb1, va0, 1);       \
    v16_23 = vfmaq_laneq_f16(v16_23, vb1, va0, 2);     \
    v24_31 = vfmaq_laneq_f16(v24_31, vb1, va0, 3);     \
    v32_39 = vfmaq_laneq_f16(v32_39, vb1, va0, 4);     \
    v40_47 = vfmaq_laneq_f16(v40_47, vb1, va0, 5);     \
    v48_55 = vfmaq_laneq_f16(v48_55, vb1, va0, 6);     \
    v56_63 = vfmaq_laneq_f16(v56_63, vb1, va0, 7);     \
    v64_71 = vfmaq_laneq_f16(v64_71, vb2, va0, 0);     \
    v72_79 = vfmaq_laneq_f16(v72_79, vb2, va0, 1);     \
    v80_87 = vfmaq_laneq_f16(v80_87, vb2, va0, 2);     \
    v88_95 = vfmaq_laneq_f16(v88_95, vb2, va0, 3);     \
    v96_103 = vfmaq_laneq_f16(v96_103, vb2, va0, 4);   \
    v104_111 = vfmaq_laneq_f16(v104_111, vb2, va0, 5); \
    v112_119 = vfmaq_laneq_f16(v112_119, vb2, va0, 6); \
    v120_127 = vfmaq_laneq_f16(v120_127, vb2, va0, 7); \
    __builtin_prefetch(b + 256, 0, 3);                 \
    __builtin_prefetch(a + 128, 0, 3);                 \
    l += 16;                                           \
    b += 16 * 16;                                      \
    a += 8 * 16;                                       \
  } while (0)

// 2. Partial sum 1024 digits
#define KERNEL_8x16_ACC8()                             \
  do {                                                 \
    va0 = vld1q_f16(a);                                \
    vb1 = vld1q_f16(b);                                \
    vb2 = vld1q_f16(b + 8);                            \
    v0_7 = vfmaq_laneq_f16(v0_7, vb1, va0, 0);         \
    v8_15 = vfmaq_laneq_f16(v8_15, vb1, va0, 1);       \
    v16_23 = vfmaq_laneq_f16(v16_23, vb1, va0, 2);     \
    v24_31 = vfmaq_laneq_f16(v24_31, vb1, va0, 3);     \
    v32_39 = vfmaq_laneq_f16(v32_39, vb1, va0, 4);     \
    v40_47 = vfmaq_laneq_f16(v40_47, vb1, va0, 5);     \
    v48_55 = vfmaq_laneq_f16(v48_55, vb1, va0, 6);     \
    v56_63 = vfmaq_laneq_f16(v56_63, vb1, va0, 7);     \
    v64_71 = vfmaq_laneq_f16(v64_71, vb2, va0, 0);     \
    v72_79 = vfmaq_laneq_f16(v72_79, vb2, va0, 1);     \
    v80_87 = vfmaq_laneq_f16(v80_87, vb2, va0, 2);     \
    v88_95 = vfmaq_laneq_f16(v88_95, vb2, va0, 3);     \
    v96_103 = vfmaq_laneq_f16(v96_103, vb2, va0, 4);   \
    v104_111 = vfmaq_laneq_f16(v104_111, vb2, va0, 5); \
    v112_119 = vfmaq_laneq_f16(v112_119, vb2, va0, 6); \
    v120_127 = vfmaq_laneq_f16(v120_127, vb2, va0, 7); \
    va0 = vld1q_f16(a + 8);                            \
    vb1 = vld1q_f16(b + 16);                           \
    vb2 = vld1q_f16(b + 24);                           \
    v0_7 = vfmaq_laneq_f16(v0_7, vb1, va0, 0);         \
    v8_15 = vfmaq_laneq_f16(v8_15, vb1, va0, 1);       \
    v16_23 = vfmaq_laneq_f16(v16_23, vb1, va0, 2);     \
    v24_31 = vfmaq_laneq_f16(v24_31, vb1, va0, 3);     \
    v32_39 = vfmaq_laneq_f16(v32_39, vb1, va0, 4);     \
    v40_47 = vfmaq_laneq_f16(v40_47, vb1, va0, 5);     \
    v48_55 = vfmaq_laneq_f16(v48_55, vb1, va0, 6);     \
    v56_63 = vfmaq_laneq_f16(v56_63, vb1, va0, 7);     \
    v64_71 = vfmaq_laneq_f16(v64_71, vb2, va0, 0);     \
    v72_79 = vfmaq_laneq_f16(v72_79, vb2, va0, 1);     \
    v80_87 = vfmaq_laneq_f16(v80_87, vb2, va0, 2);     \
    v88_95 = vfmaq_laneq_f16(v88_95, vb2, va0, 3);     \
    v96_103 = vfmaq_laneq_f16(v96_103, vb2, va0, 4);   \
    v104_111 = vfmaq_laneq_f16(v104_111, vb2, va0, 5); \
    v112_119 = vfmaq_laneq_f16(v112_119, vb2, va0, 6); \
    v120_127 = vfmaq_laneq_f16(v120_127, vb2, va0, 7); \
    va0 = vld1q_f16(a + 16);                           \
    vb1 = vld1q_f16(b + 32);                           \
    vb2 = vld1q_f16(b + 40);                           \
    v0_7 = vfmaq_laneq_f16(v0_7, vb1, va0, 0);         \
    v8_15 = vfmaq_laneq_f16(v8_15, vb1, va0, 1);       \
    v16_23 = vfmaq_laneq_f16(v16_23, vb1, va0, 2);     \
    v24_31 = vfmaq_laneq_f16(v24_31, vb1, va0, 3);     \
    v32_39 = vfmaq_laneq_f16(v32_39, vb1, va0, 4);     \
    v40_47 = vfmaq_laneq_f16(v40_47, vb1, va0, 5);     \
    v48_55 = vfmaq_laneq_f16(v48_55, vb1, va0, 6);     \
    v56_63 = vfmaq_laneq_f16(v56_63, vb1, va0, 7);     \
    v64_71 = vfmaq_laneq_f16(v64_71, vb2, va0, 0);     \
    v72_79 = vfmaq_laneq_f16(v72_79, vb2, va0, 1);     \
    v80_87 = vfmaq_laneq_f16(v80_87, vb2, va0, 2);     \
    v88_95 = vfmaq_laneq_f16(v88_95, vb2, va0, 3);     \
    v96_103 = vfmaq_laneq_f16(v96_103, vb2, va0, 4);   \
    v104_111 = vfmaq_laneq_f16(v104_111, vb2, va0, 5); \
    v112_119 = vfmaq_laneq_f16(v112_119, vb2, va0, 6); \
    v120_127 = vfmaq_laneq_f16(v120_127, vb2, va0, 7); \
    va0 = vld1q_f16(a + 24);                           \
    vb1 = vld1q_f16(b + 48);                           \
    vb2 = vld1q_f16(b + 56);                           \
    v0_7 = vfmaq_laneq_f16(v0_7, vb1, va0, 0);         \
    v8_15 = vfmaq_laneq_f16(v8_15, vb1, va0, 1);       \
    v16_23 = vfmaq_laneq_f16(v16_23, vb1, va0, 2);     \
    v24_31 = vfmaq_laneq_f16(v24_31, vb1, va0, 3);     \
    v32_39 = vfmaq_laneq_f16(v32_39, vb1, va0, 4);     \
    v40_47 = vfmaq_laneq_f16(v40_47, vb1, va0, 5);     \
    v48_55 = vfmaq_laneq_f16(v48_55, vb1, va0, 6);     \
    v56_63 = vfmaq_laneq_f16(v56_63, vb1, va0, 7);     \
    v64_71 = vfmaq_laneq_f16(v64_71, vb2, va0, 0);     \
    v72_79 = vfmaq_laneq_f16(v72_79, vb2, va0, 1);     \
    v80_87 = vfmaq_laneq_f16(v80_87, vb2, va0, 2);     \
    v88_95 = vfmaq_laneq_f16(v88_95, vb2, va0, 3);     \
    v96_103 = vfmaq_laneq_f16(v96_103, vb2, va0, 4);   \
    v104_111 = vfmaq_laneq_f16(v104_111, vb2, va0, 5); \
    v112_119 = vfmaq_laneq_f16(v112_119, vb2, va0, 6); \
    v120_127 = vfmaq_laneq_f16(v120_127, vb2, va0, 7); \
    va0 = vld1q_f16(a + 32);                           \
    vb1 = vld1q_f16(b + 64);                           \
    vb2 = vld1q_f16(b + 72);                           \
    v0_7 = vfmaq_laneq_f16(v0_7, vb1, va0, 0);         \
    v8_15 = vfmaq_laneq_f16(v8_15, vb1, va0, 1);       \
    v16_23 = vfmaq_laneq_f16(v16_23, vb1, va0, 2);     \
    v24_31 = vfmaq_laneq_f16(v24_31, vb1, va0, 3);     \
    v32_39 = vfmaq_laneq_f16(v32_39, vb1, va0, 4);     \
    v40_47 = vfmaq_laneq_f16(v40_47, vb1, va0, 5);     \
    v48_55 = vfmaq_laneq_f16(v48_55, vb1, va0, 6);     \
    v56_63 = vfmaq_laneq_f16(v56_63, vb1, va0, 7);     \
    v64_71 = vfmaq_laneq_f16(v64_71, vb2, va0, 0);     \
    v72_79 = vfmaq_laneq_f16(v72_79, vb2, va0, 1);     \
    v80_87 = vfmaq_laneq_f16(v80_87, vb2, va0, 2);     \
    v88_95 = vfmaq_laneq_f16(v88_95, vb2, va0, 3);     \
    v96_103 = vfmaq_laneq_f16(v96_103, vb2, va0, 4);   \
    v104_111 = vfmaq_laneq_f16(v104_111, vb2, va0, 5); \
    v112_119 = vfmaq_laneq_f16(v112_119, vb2, va0, 6); \
    v120_127 = vfmaq_laneq_f16(v120_127, vb2, va0, 7); \
    va0 = vld1q_f16(a + 40);                           \
    vb1 = vld1q_f16(b + 80);                           \
    vb2 = vld1q_f16(b + 88);                           \
    v0_7 = vfmaq_laneq_f16(v0_7, vb1, va0, 0);         \
    v8_15 = vfmaq_laneq_f16(v8_15, vb1, va0, 1);       \
    v16_23 = vfmaq_laneq_f16(v16_23, vb1, va0, 2);     \
    v24_31 = vfmaq_laneq_f16(v24_31, vb1, va0, 3);     \
    v32_39 = vfmaq_laneq_f16(v32_39, vb1, va0, 4);     \
    v40_47 = vfmaq_laneq_f16(v40_47, vb1, va0, 5);     \
    v48_55 = vfmaq_laneq_f16(v48_55, vb1, va0, 6);     \
    v56_63 = vfmaq_laneq_f16(v56_63, vb1, va0, 7);     \
    v64_71 = vfmaq_laneq_f16(v64_71, vb2, va0, 0);     \
    v72_79 = vfmaq_laneq_f16(v72_79, vb2, va0, 1);     \
    v80_87 = vfmaq_laneq_f16(v80_87, vb2, va0, 2);     \
    v88_95 = vfmaq_laneq_f16(v88_95, vb2, va0, 3);     \
    v96_103 = vfmaq_laneq_f16(v96_103, vb2, va0, 4);   \
    v104_111 = vfmaq_laneq_f16(v104_111, vb2, va0, 5); \
    v112_119 = vfmaq_laneq_f16(v112_119, vb2, va0, 6); \
    v120_127 = vfmaq_laneq_f16(v120_127, vb2, va0, 7); \
    va0 = vld1q_f16(a + 48);                           \
    vb1 = vld1q_f16(b + 96);                           \
    vb2 = vld1q_f16(b + 104);                          \
    v0_7 = vfmaq_laneq_f16(v0_7, vb1, va0, 0);         \
    v8_15 = vfmaq_laneq_f16(v8_15, vb1, va0, 1);       \
    v16_23 = vfmaq_laneq_f16(v16_23, vb1, va0, 2);     \
    v24_31 = vfmaq_laneq_f16(v24_31, vb1, va0, 3);     \
    v32_39 = vfmaq_laneq_f16(v32_39, vb1, va0, 4);     \
    v40_47 = vfmaq_laneq_f16(v40_47, vb1, va0, 5);     \
    v48_55 = vfmaq_laneq_f16(v48_55, vb1, va0, 6);     \
    v56_63 = vfmaq_laneq_f16(v56_63, vb1, va0, 7);     \
    v64_71 = vfmaq_laneq_f16(v64_71, vb2, va0, 0);     \
    v72_79 = vfmaq_laneq_f16(v72_79, vb2, va0, 1);     \
    v80_87 = vfmaq_laneq_f16(v80_87, vb2, va0, 2);     \
    v88_95 = vfmaq_laneq_f16(v88_95, vb2, va0, 3);     \
    v96_103 = vfmaq_laneq_f16(v96_103, vb2, va0, 4);   \
    v104_111 = vfmaq_laneq_f16(v104_111, vb2, va0, 5); \
    v112_119 = vfmaq_laneq_f16(v112_119, vb2, va0, 6); \
    v120_127 = vfmaq_laneq_f16(v120_127, vb2, va0, 7); \
    va0 = vld1q_f16(a + 56);                           \
    vb1 = vld1q_f16(b + 112);                          \
    vb2 = vld1q_f16(b + 120);                          \
    v0_7 = vfmaq_laneq_f16(v0_7, vb1, va0, 0);         \
    v8_15 = vfmaq_laneq_f16(v8_15, vb1, va0, 1);       \
    v16_23 = vfmaq_laneq_f16(v16_23, vb1, va0, 2);     \
    v24_31 = vfmaq_laneq_f16(v24_31, vb1, va0, 3);     \
    v32_39 = vfmaq_laneq_f16(v32_39, vb1, va0, 4);     \
    v40_47 = vfmaq_laneq_f16(v40_47, vb1, va0, 5);     \
    v48_55 = vfmaq_laneq_f16(v48_55, vb1, va0, 6);     \
    v56_63 = vfmaq_laneq_f16(v56_63, vb1, va0, 7);     \
    v64_71 = vfmaq_laneq_f16(v64_71, vb2, va0, 0);     \
    v72_79 = vfmaq_laneq_f16(v72_79, vb2, va0, 1);     \
    v80_87 = vfmaq_laneq_f16(v80_87, vb2, va0, 2);     \
    v88_95 = vfmaq_laneq_f16(v88_95, vb2, va0, 3);     \
    v96_103 = vfmaq_laneq_f16(v96_103, vb2, va0, 4);   \
    v104_111 = vfmaq_laneq_f16(v104_111, vb2, va0, 5); \
    v112_119 = vfmaq_laneq_f16(v112_119, vb2, va0, 6); \
    v120_127 = vfmaq_laneq_f16(v120_127, vb2, va0, 7); \
    l += 8;                                            \
    __builtin_prefetch(b + 128, 0, 3);                 \
    __builtin_prefetch(a + 64, 0, 3);                  \
    b += 16 * 8;                                       \
    a += 8 * 8;                                        \
  } while (0)

// 3. Partial sum 512 digits
#define KERNEL_8x16_ACC4()                             \
  do {                                                 \
    va0 = vld1q_f16(a);                                \
    vb1 = vld1q_f16(b);                                \
    vb2 = vld1q_f16(b + 8);                            \
    v0_7 = vfmaq_laneq_f16(v0_7, vb1, va0, 0);         \
    v8_15 = vfmaq_laneq_f16(v8_15, vb1, va0, 1);       \
    v16_23 = vfmaq_laneq_f16(v16_23, vb1, va0, 2);     \
    v24_31 = vfmaq_laneq_f16(v24_31, vb1, va0, 3);     \
    v32_39 = vfmaq_laneq_f16(v32_39, vb1, va0, 4);     \
    v40_47 = vfmaq_laneq_f16(v40_47, vb1, va0, 5);     \
    v48_55 = vfmaq_laneq_f16(v48_55, vb1, va0, 6);     \
    v56_63 = vfmaq_laneq_f16(v56_63, vb1, va0, 7);     \
    v64_71 = vfmaq_laneq_f16(v64_71, vb2, va0, 0);     \
    v72_79 = vfmaq_laneq_f16(v72_79, vb2, va0, 1);     \
    v80_87 = vfmaq_laneq_f16(v80_87, vb2, va0, 2);     \
    v88_95 = vfmaq_laneq_f16(v88_95, vb2, va0, 3);     \
    v96_103 = vfmaq_laneq_f16(v96_103, vb2, va0, 4);   \
    v104_111 = vfmaq_laneq_f16(v104_111, vb2, va0, 5); \
    v112_119 = vfmaq_laneq_f16(v112_119, vb2, va0, 6); \
    v120_127 = vfmaq_laneq_f16(v120_127, vb2, va0, 7); \
    va0 = vld1q_f16(a + 8);                            \
    vb1 = vld1q_f16(b + 16);                           \
    vb2 = vld1q_f16(b + 24);                           \
    v0_7 = vfmaq_laneq_f16(v0_7, vb1, va0, 0);         \
    v8_15 = vfmaq_laneq_f16(v8_15, vb1, va0, 1);       \
    v16_23 = vfmaq_laneq_f16(v16_23, vb1, va0, 2);     \
    v24_31 = vfmaq_laneq_f16(v24_31, vb1, va0, 3);     \
    v32_39 = vfmaq_laneq_f16(v32_39, vb1, va0, 4);     \
    v40_47 = vfmaq_laneq_f16(v40_47, vb1, va0, 5);     \
    v48_55 = vfmaq_laneq_f16(v48_55, vb1, va0, 6);     \
    v56_63 = vfmaq_laneq_f16(v56_63, vb1, va0, 7);     \
    v64_71 = vfmaq_laneq_f16(v64_71, vb2, va0, 0);     \
    v72_79 = vfmaq_laneq_f16(v72_79, vb2, va0, 1);     \
    v80_87 = vfmaq_laneq_f16(v80_87, vb2, va0, 2);     \
    v88_95 = vfmaq_laneq_f16(v88_95, vb2, va0, 3);     \
    v96_103 = vfmaq_laneq_f16(v96_103, vb2, va0, 4);   \
    v104_111 = vfmaq_laneq_f16(v104_111, vb2, va0, 5); \
    v112_119 = vfmaq_laneq_f16(v112_119, vb2, va0, 6); \
    v120_127 = vfmaq_laneq_f16(v120_127, vb2, va0, 7); \
    va0 = vld1q_f16(a + 16);                           \
    vb1 = vld1q_f16(b + 32);                           \
    vb2 = vld1q_f16(b + 40);                           \
    v0_7 = vfmaq_laneq_f16(v0_7, vb1, va0, 0);         \
    v8_15 = vfmaq_laneq_f16(v8_15, vb1, va0, 1);       \
    v16_23 = vfmaq_laneq_f16(v16_23, vb1, va0, 2);     \
    v24_31 = vfmaq_laneq_f16(v24_31, vb1, va0, 3);     \
    v32_39 = vfmaq_laneq_f16(v32_39, vb1, va0, 4);     \
    v40_47 = vfmaq_laneq_f16(v40_47, vb1, va0, 5);     \
    v48_55 = vfmaq_laneq_f16(v48_55, vb1, va0, 6);     \
    v56_63 = vfmaq_laneq_f16(v56_63, vb1, va0, 7);     \
    v64_71 = vfmaq_laneq_f16(v64_71, vb2, va0, 0);     \
    v72_79 = vfmaq_laneq_f16(v72_79, vb2, va0, 1);     \
    v80_87 = vfmaq_laneq_f16(v80_87, vb2, va0, 2);     \
    v88_95 = vfmaq_laneq_f16(v88_95, vb2, va0, 3);     \
    v96_103 = vfmaq_laneq_f16(v96_103, vb2, va0, 4);   \
    v104_111 = vfmaq_laneq_f16(v104_111, vb2, va0, 5); \
    v112_119 = vfmaq_laneq_f16(v112_119, vb2, va0, 6); \
    v120_127 = vfmaq_laneq_f16(v120_127, vb2, va0, 7); \
    va0 = vld1q_f16(a + 24);                           \
    vb1 = vld1q_f16(b + 48);                           \
    vb2 = vld1q_f16(b + 56);                           \
    v0_7 = vfmaq_laneq_f16(v0_7, vb1, va0, 0);         \
    v8_15 = vfmaq_laneq_f16(v8_15, vb1, va0, 1);       \
    v16_23 = vfmaq_laneq_f16(v16_23, vb1, va0, 2);     \
    v24_31 = vfmaq_laneq_f16(v24_31, vb1, va0, 3);     \
    v32_39 = vfmaq_laneq_f16(v32_39, vb1, va0, 4);     \
    v40_47 = vfmaq_laneq_f16(v40_47, vb1, va0, 5);     \
    v48_55 = vfmaq_laneq_f16(v48_55, vb1, va0, 6);     \
    v56_63 = vfmaq_laneq_f16(v56_63, vb1, va0, 7);     \
    v64_71 = vfmaq_laneq_f16(v64_71, vb2, va0, 0);     \
    v72_79 = vfmaq_laneq_f16(v72_79, vb2, va0, 1);     \
    v80_87 = vfmaq_laneq_f16(v80_87, vb2, va0, 2);     \
    v88_95 = vfmaq_laneq_f16(v88_95, vb2, va0, 3);     \
    v96_103 = vfmaq_laneq_f16(v96_103, vb2, va0, 4);   \
    v104_111 = vfmaq_laneq_f16(v104_111, vb2, va0, 5); \
    v112_119 = vfmaq_laneq_f16(v112_119, vb2, va0, 6); \
    v120_127 = vfmaq_laneq_f16(v120_127, vb2, va0, 7); \
    l += 4;                                            \
    __builtin_prefetch(b + 64, 0, 3);                  \
    __builtin_prefetch(a + 32, 0, 3);                  \
    b += 16 * 4;                                       \
    a += 8 * 4;                                        \
  } while (0)

// 4. Partial sum 128 digits
#define KERNEL_8x16_ACC1()                             \
  do {                                                 \
    va0 = vld1q_f16(a);                                \
    vb1 = vld1q_f16(b);                                \
    vb2 = vld1q_f16(b + 8);                            \
    v0_7 = vfmaq_laneq_f16(v0_7, vb1, va0, 0);         \
    v8_15 = vfmaq_laneq_f16(v8_15, vb1, va0, 1);       \
    v16_23 = vfmaq_laneq_f16(v16_23, vb1, va0, 2);     \
    v24_31 = vfmaq_laneq_f16(v24_31, vb1, va0, 3);     \
    v32_39 = vfmaq_laneq_f16(v32_39, vb1, va0, 4);     \
    v40_47 = vfmaq_laneq_f16(v40_47, vb1, va0, 5);     \
    v48_55 = vfmaq_laneq_f16(v48_55, vb1, va0, 6);     \
    v56_63 = vfmaq_laneq_f16(v56_63, vb1, va0, 7);     \
    v64_71 = vfmaq_laneq_f16(v64_71, vb2, va0, 0);     \
    v72_79 = vfmaq_laneq_f16(v72_79, vb2, va0, 1);     \
    v80_87 = vfmaq_laneq_f16(v80_87, vb2, va0, 2);     \
    v88_95 = vfmaq_laneq_f16(v88_95, vb2, va0, 3);     \
    v96_103 = vfmaq_laneq_f16(v96_103, vb2, va0, 4);   \
    v104_111 = vfmaq_laneq_f16(v104_111, vb2, va0, 5); \
    v112_119 = vfmaq_laneq_f16(v112_119, vb2, va0, 6); \
    v120_127 = vfmaq_laneq_f16(v120_127, vb2, va0, 7); \
    l += 1;                                            \
    __builtin_prefetch(b + 16, 0, 3);                  \
    __builtin_prefetch(a + 8, 0, 3);                   \
    b += 16 * 1;                                       \
    a += 8 * 1;                                        \
  } while (0)

#define SAVE_KERNEL_8X16_F16_F32()                                             \
  do {                                                                         \
    vst1q_f32(c, vaddq_f32(vld1q_f32(c), vcvt_f32_f16(vget_low_f16(v0_7))));   \
    vst1q_f32(c + 4,                                                           \
              vaddq_f32(vld1q_f32(c + 4), vcvt_f32_f16(vget_high_f16(v0_7)))); \
                                                                               \
    vst1q_f32(                                                                 \
      c + 8, vaddq_f32(vld1q_f32(c + 8), vcvt_f32_f16(vget_low_f16(v64_71)))); \
    vst1q_f32(c + 8 + 4, vaddq_f32(vld1q_f32(c + 8 + 4),                       \
                                   vcvt_f32_f16(vget_high_f16(v64_71))));      \
                                                                               \
    vst1q_f32(c + ldc, vaddq_f32(vld1q_f32(c + ldc),                           \
                                 vcvt_f32_f16(vget_low_f16(v8_15))));          \
    vst1q_f32(c + ldc + 4, vaddq_f32(vld1q_f32(c + ldc + 4),                   \
                                     vcvt_f32_f16(vget_high_f16(v8_15))));     \
                                                                               \
    vst1q_f32(c + ldc + 8, vaddq_f32(vld1q_f32(c + ldc + 8),                   \
                                     vcvt_f32_f16(vget_low_f16(v72_79))));     \
    vst1q_f32(c + ldc + 8 + 4,                                                 \
              vaddq_f32(vld1q_f32(c + ldc + 8 + 4),                            \
                        vcvt_f32_f16(vget_high_f16(v72_79))));                 \
                                                                               \
    vst1q_f32(c + 2 * ldc, vaddq_f32(vld1q_f32(c + 2 * ldc),                   \
                                     vcvt_f32_f16(vget_low_f16(v16_23))));     \
    vst1q_f32(c + 2 * ldc + 4,                                                 \
              vaddq_f32(vld1q_f32(c + 2 * ldc + 4),                            \
                        vcvt_f32_f16(vget_high_f16(v16_23))));                 \
                                                                               \
    vst1q_f32(c + 2 * ldc + 8, vaddq_f32(vld1q_f32(c + 2 * ldc + 8),           \
                                         vcvt_f32_f16(vget_low_f16(v80_87)))); \
    vst1q_f32(c + 2 * ldc + 8 + 4,                                             \
              vaddq_f32(vld1q_f32(c + 2 * ldc + 8 + 4),                        \
                        vcvt_f32_f16(vget_high_f16(v80_87))));                 \
                                                                               \
    vst1q_f32(c + 3 * ldc, vaddq_f32(vld1q_f32(c + 3 * ldc),                   \
                                     vcvt_f32_f16(vget_low_f16(v24_31))));     \
    vst1q_f32(c + 3 * ldc + 4,                                                 \
              vaddq_f32(vld1q_f32(c + 3 * ldc + 4),                            \
                        vcvt_f32_f16(vget_high_f16(v24_31))));                 \
                                                                               \
    vst1q_f32(c + 3 * ldc + 8, vaddq_f32(vld1q_f32(c + 3 * ldc + 8),           \
                                         vcvt_f32_f16(vget_low_f16(v88_95)))); \
    vst1q_f32(c + 3 * ldc + 8 + 4,                                             \
              vaddq_f32(vld1q_f32(c + 3 * ldc + 8 + 4),                        \
                        vcvt_f32_f16(vget_high_f16(v88_95))));                 \
                                                                               \
    vst1q_f32(c + 4 * ldc, vaddq_f32(vld1q_f32(c + 4 * ldc),                   \
                                     vcvt_f32_f16(vget_low_f16(v32_39))));     \
    vst1q_f32(c + 4 * ldc + 4,                                                 \
              vaddq_f32(vld1q_f32(c + 4 * ldc + 4),                            \
                        vcvt_f32_f16(vget_high_f16(v32_39))));                 \
                                                                               \
    vst1q_f32(c + 4 * ldc + 8,                                                 \
              vaddq_f32(vld1q_f32(c + 4 * ldc + 8),                            \
                        vcvt_f32_f16(vget_low_f16(v96_103))));                 \
    vst1q_f32(c + 4 * ldc + 8 + 4,                                             \
              vaddq_f32(vld1q_f32(c + 4 * ldc + 8 + 4),                        \
                        vcvt_f32_f16(vget_high_f16(v96_103))));                \
                                                                               \
    vst1q_f32(c + 5 * ldc, vaddq_f32(vld1q_f32(c + 5 * ldc),                   \
                                     vcvt_f32_f16(vget_low_f16(v40_47))));     \
    vst1q_f32(c + 5 * ldc + 4,                                                 \
              vaddq_f32(vld1q_f32(c + 5 * ldc + 4),                            \
                        vcvt_f32_f16(vget_high_f16(v40_47))));                 \
    vst1q_f32(c + 5 * ldc + 8,                                                 \
              vaddq_f32(vld1q_f32(c + 5 * ldc + 8),                            \
                        vcvt_f32_f16(vget_low_f16(v104_111))));                \
    vst1q_f32(c + 5 * ldc + 8 + 4,                                             \
              vaddq_f32(vld1q_f32(c + 5 * ldc + 8 + 4),                        \
                        vcvt_f32_f16(vget_high_f16(v104_111))));               \
                                                                               \
    vst1q_f32(c + 6 * ldc, vaddq_f32(vld1q_f32(c + 6 * ldc),                   \
                                     vcvt_f32_f16(vget_low_f16(v48_55))));     \
    vst1q_f32(c + 6 * ldc + 4,                                                 \
              vaddq_f32(vld1q_f32(c + 6 * ldc + 4),                            \
                        vcvt_f32_f16(vget_high_f16(v48_55))));                 \
                                                                               \
    vst1q_f32(c + 6 * ldc + 8,                                                 \
              vaddq_f32(vld1q_f32(c + 6 * ldc + 8),                            \
                        vcvt_f32_f16(vget_low_f16(v112_119))));                \
    vst1q_f32(c + 6 * ldc + 8 + 4,                                             \
              vaddq_f32(vld1q_f32(c + 6 * ldc + 8 + 4),                        \
                        vcvt_f32_f16(vget_high_f16(v112_119))));               \
                                                                               \
    vst1q_f32(c + 7 * ldc, vaddq_f32(vld1q_f32(c + 7 * ldc),                   \
                                     vcvt_f32_f16(vget_low_f16(v56_63))));     \
    vst1q_f32(c + 7 * ldc + 4,                                                 \
              vaddq_f32(vld1q_f32(c + 7 * ldc + 4),                            \
                        vcvt_f32_f16(vget_high_f16(v56_63))));                 \
                                                                               \
    vst1q_f32(c + 7 * ldc + 8,                                                 \
              vaddq_f32(vld1q_f32(c + 7 * ldc + 8),                            \
                        vcvt_f32_f16(vget_low_f16(v120_127))));                \
    vst1q_f32(c + 7 * ldc + 8 + 4,                                             \
              vaddq_f32(vld1q_f32(c + 7 * ldc + 8 + 4),                        \
                        vcvt_f32_f16(vget_high_f16(v120_127))));               \
  } while (0)

void hgemm_kernel_8x16(unsigned int M, unsigned int N, unsigned int K,
                       __fp16 *sa, __fp16 *sb, __fp16 *sc, unsigned int ldc) {
  assert(M > 0 && N > 0 && K > 0);
  assert(M % 8 == 0 && N % 16 == 0 && K % 8 == 0);

  __fp16 *a = sa, *b = sb, *c = sc;
  unsigned int i, j, l;
  for (i = 0; i < M; i += 8) {
    for (j = 0; j < N; j += 16) {
      __builtin_prefetch(b, 0, 3);
      __builtin_prefetch(a, 0, 3);
      // 8x16
      float16x8_t v0_7, v8_15;
      float16x8_t v16_23, v24_31;
      float16x8_t v32_39, v40_47;
      float16x8_t v48_55, v56_63;
      float16x8_t v64_71, v72_79;
      float16x8_t v80_87, v88_95;
      float16x8_t v96_103, v104_111;
      float16x8_t v112_119, v120_127;
      float16x8_t vb1, vb2;
      float16x8_t va0;

      INIT_KERNEL_8X16();
      l = 0;
      for (; l < K;) {
        KERNEL_8x16_ACC1();
      }
      vst1q_f16(c, vaddq_f16(vld1q_f16(c), v0_7));
      vst1q_f16(c + 8, vaddq_f16(vld1q_f16(c + 8), v64_71));
      vst1q_f16(c + ldc, vaddq_f16(vld1q_f16(c + ldc), v8_15));
      vst1q_f16(c + ldc + 8, vaddq_f16(vld1q_f16(c + ldc + 8), v72_79));
      vst1q_f16(c + 2 * ldc, vaddq_f16(vld1q_f16(c + 2 * ldc), v16_23));
      vst1q_f16(c + 2 * ldc + 8, vaddq_f16(vld1q_f16(c + 2 * ldc + 8), v80_87));
      vst1q_f16(c + 3 * ldc, vaddq_f16(vld1q_f16(c + 3 * ldc), v24_31));
      vst1q_f16(c + 3 * ldc + 8, vaddq_f16(vld1q_f16(c + 3 * ldc + 8), v88_95));
      vst1q_f16(c + 4 * ldc, vaddq_f16(vld1q_f16(c + 4 * ldc), v32_39));
      vst1q_f16(c + 4 * ldc + 8,
                vaddq_f16(vld1q_f16(c + 4 * ldc + 8), v96_103));
      vst1q_f16(c + 5 * ldc, vaddq_f16(vld1q_f16(c + 5 * ldc), v40_47));
      vst1q_f16(c + 5 * ldc + 8,
                vaddq_f16(vld1q_f16(c + 5 * ldc + 8), v104_111));
      vst1q_f16(c + 6 * ldc, vaddq_f16(vld1q_f16(c + 6 * ldc), v48_55));
      vst1q_f16(c + 6 * ldc + 8,
                vaddq_f16(vld1q_f16(c + 6 * ldc + 8), v112_119));
      vst1q_f16(c + 7 * ldc, vaddq_f16(vld1q_f16(c + 7 * ldc), v56_63));
      vst1q_f16(c + 7 * ldc + 8,
                vaddq_f16(vld1q_f16(c + 7 * ldc + 8), v120_127));
      c += 16;
      a -= 8 * K;
    }
    sc += ldc * 8;
    c = sc;
    a += 8 * K;
    b = sb;
  }
}

void hgemm_kernel_8x16(unsigned int M, unsigned int N, unsigned int K,
                       __fp16 *sa, __fp16 *sb, float *sc, unsigned int ldc) {
  assert(M > 0 && N > 0 && K > 0);
  assert(M % 8 == 0 && N % 16 == 0 && K % 4 == 0);

  __fp16 *a = sa, *b = sb;
  float *c = sc;
  unsigned int i, j, l;
  unsigned int K4 = (K >> 2) << 2;
  unsigned int K8 = (K >> 3) << 3;
  unsigned int K16 = (K >> 4) << 4;
  for (i = 0; i < M; i += 8) {
    for (j = 0; j < N; j += 16) {
      __builtin_prefetch(b, 0, 3);
      __builtin_prefetch(a, 0, 3);
      float16x8_t v0_7, v8_15;
      float16x8_t v16_23, v24_31;
      float16x8_t v32_39, v40_47;
      float16x8_t v48_55, v56_63;
      float16x8_t v64_71, v72_79;
      float16x8_t v80_87, v88_95;
      float16x8_t v96_103, v104_111;
      float16x8_t v112_119, v120_127;
      float16x8_t va0;
      float16x8_t vb1, vb2;
      l = 0;
      for (; l < K16;) {
        INIT_KERNEL_8X16();
        KERNEL_8x16_ACC16();
        SAVE_KERNEL_8X16_F16_F32();
      }
      for (; l < K8;) {
        INIT_KERNEL_8X16();
        KERNEL_8x16_ACC8();
        SAVE_KERNEL_8X16_F16_F32();
      }
      for (; l < K4;) {
        INIT_KERNEL_8X16();
        KERNEL_8x16_ACC4();
        SAVE_KERNEL_8X16_F16_F32();
      }
      for (; l < K;) {
        INIT_KERNEL_8X16();
        KERNEL_8x16_ACC1();
        SAVE_KERNEL_8X16_F16_F32();
      }
      c += 16;
      a -= 8 * K;
    }
    sc += ldc * 8;
    c = sc;
    a += 8 * K;
    b = sb;
  }
}
