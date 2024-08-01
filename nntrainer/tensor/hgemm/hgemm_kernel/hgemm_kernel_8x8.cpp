// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file   hgemm_kernel_8x8.cpp
 * @date   01 April 2024
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is half-precision GEMM 8x8 kernel
 *
 */

#include <arm_neon.h>
#include <assert.h>
#include <hgemm_kernel.h>
#include <stdlib.h>

#define INIT_KERNEL_8x8()   \
  do {                      \
    v24 = vdupq_n_f16(0.F); \
    v25 = vdupq_n_f16(0.F); \
    v26 = vdupq_n_f16(0.F); \
    v27 = vdupq_n_f16(0.F); \
    v28 = vdupq_n_f16(0.F); \
    v29 = vdupq_n_f16(0.F); \
    v30 = vdupq_n_f16(0.F); \
    v31 = vdupq_n_f16(0.F); \
  } while (0)

// 1. Partial sum 1024 digits
#define KERNEL_8x8_ACC16()                   \
  do {                                       \
    va0 = vld1q_f16(a);                      \
    v16 = vld1q_f16(b);                      \
    v24 = vfmaq_laneq_f16(v24, v16, va0, 0); \
    v25 = vfmaq_laneq_f16(v25, v16, va0, 1); \
    v26 = vfmaq_laneq_f16(v26, v16, va0, 2); \
    v27 = vfmaq_laneq_f16(v27, v16, va0, 3); \
    v28 = vfmaq_laneq_f16(v28, v16, va0, 4); \
    v29 = vfmaq_laneq_f16(v29, v16, va0, 5); \
    v30 = vfmaq_laneq_f16(v30, v16, va0, 6); \
    v31 = vfmaq_laneq_f16(v31, v16, va0, 7); \
    va0 = vld1q_f16(a + 8);                  \
    v17 = vld1q_f16(b + 8);                  \
    v24 = vfmaq_laneq_f16(v24, v17, va0, 0); \
    v25 = vfmaq_laneq_f16(v25, v17, va0, 1); \
    v26 = vfmaq_laneq_f16(v26, v17, va0, 2); \
    v27 = vfmaq_laneq_f16(v27, v17, va0, 3); \
    v28 = vfmaq_laneq_f16(v28, v17, va0, 4); \
    v29 = vfmaq_laneq_f16(v29, v17, va0, 5); \
    v30 = vfmaq_laneq_f16(v30, v17, va0, 6); \
    v31 = vfmaq_laneq_f16(v31, v17, va0, 7); \
    va0 = vld1q_f16(a + 8 * 2);              \
    v18 = vld1q_f16(b + 8 * 2);              \
    v24 = vfmaq_laneq_f16(v24, v18, va0, 0); \
    v25 = vfmaq_laneq_f16(v25, v18, va0, 1); \
    v26 = vfmaq_laneq_f16(v26, v18, va0, 2); \
    v27 = vfmaq_laneq_f16(v27, v18, va0, 3); \
    v28 = vfmaq_laneq_f16(v28, v18, va0, 4); \
    v29 = vfmaq_laneq_f16(v29, v18, va0, 5); \
    v30 = vfmaq_laneq_f16(v30, v18, va0, 6); \
    v31 = vfmaq_laneq_f16(v31, v18, va0, 7); \
    va0 = vld1q_f16(a + 8 * 3);              \
    v19 = vld1q_f16(b + 8 * 3);              \
    v24 = vfmaq_laneq_f16(v24, v19, va0, 0); \
    v25 = vfmaq_laneq_f16(v25, v19, va0, 1); \
    v26 = vfmaq_laneq_f16(v26, v19, va0, 2); \
    v27 = vfmaq_laneq_f16(v27, v19, va0, 3); \
    v28 = vfmaq_laneq_f16(v28, v19, va0, 4); \
    v29 = vfmaq_laneq_f16(v29, v19, va0, 5); \
    v30 = vfmaq_laneq_f16(v30, v19, va0, 6); \
    v31 = vfmaq_laneq_f16(v31, v19, va0, 7); \
    va0 = vld1q_f16(a + 8 * 4);              \
    v20 = vld1q_f16(b + 8 * 4);              \
    v24 = vfmaq_laneq_f16(v24, v20, va0, 0); \
    v25 = vfmaq_laneq_f16(v25, v20, va0, 1); \
    v26 = vfmaq_laneq_f16(v26, v20, va0, 2); \
    v27 = vfmaq_laneq_f16(v27, v20, va0, 3); \
    v28 = vfmaq_laneq_f16(v28, v20, va0, 4); \
    v29 = vfmaq_laneq_f16(v29, v20, va0, 5); \
    v30 = vfmaq_laneq_f16(v30, v20, va0, 6); \
    v31 = vfmaq_laneq_f16(v31, v20, va0, 7); \
    va0 = vld1q_f16(a + 8 * 5);              \
    v21 = vld1q_f16(b + 8 * 5);              \
    v24 = vfmaq_laneq_f16(v24, v21, va0, 0); \
    v25 = vfmaq_laneq_f16(v25, v21, va0, 1); \
    v26 = vfmaq_laneq_f16(v26, v21, va0, 2); \
    v27 = vfmaq_laneq_f16(v27, v21, va0, 3); \
    v28 = vfmaq_laneq_f16(v28, v21, va0, 4); \
    v29 = vfmaq_laneq_f16(v29, v21, va0, 5); \
    v30 = vfmaq_laneq_f16(v30, v21, va0, 6); \
    v31 = vfmaq_laneq_f16(v31, v21, va0, 7); \
    va0 = vld1q_f16(a + 8 * 6);              \
    v22 = vld1q_f16(b + 8 * 6);              \
    v24 = vfmaq_laneq_f16(v24, v22, va0, 0); \
    v25 = vfmaq_laneq_f16(v25, v22, va0, 1); \
    v26 = vfmaq_laneq_f16(v26, v22, va0, 2); \
    v27 = vfmaq_laneq_f16(v27, v22, va0, 3); \
    v28 = vfmaq_laneq_f16(v28, v22, va0, 4); \
    v29 = vfmaq_laneq_f16(v29, v22, va0, 5); \
    v30 = vfmaq_laneq_f16(v30, v22, va0, 6); \
    v31 = vfmaq_laneq_f16(v31, v22, va0, 7); \
    va0 = vld1q_f16(a + 8 * 7);              \
    v23 = vld1q_f16(b + 8 * 7);              \
    v24 = vfmaq_laneq_f16(v24, v23, va0, 0); \
    v25 = vfmaq_laneq_f16(v25, v23, va0, 1); \
    v26 = vfmaq_laneq_f16(v26, v23, va0, 2); \
    v27 = vfmaq_laneq_f16(v27, v23, va0, 3); \
    v28 = vfmaq_laneq_f16(v28, v23, va0, 4); \
    v29 = vfmaq_laneq_f16(v29, v23, va0, 5); \
    v30 = vfmaq_laneq_f16(v30, v23, va0, 6); \
    v31 = vfmaq_laneq_f16(v31, v23, va0, 7); \
    va0 = vld1q_f16(a + 8 * 8);              \
    v23 = vld1q_f16(b + 8 * 8);              \
    v24 = vfmaq_laneq_f16(v24, v23, va0, 0); \
    v25 = vfmaq_laneq_f16(v25, v23, va0, 1); \
    v26 = vfmaq_laneq_f16(v26, v23, va0, 2); \
    v27 = vfmaq_laneq_f16(v27, v23, va0, 3); \
    v28 = vfmaq_laneq_f16(v28, v23, va0, 4); \
    v29 = vfmaq_laneq_f16(v29, v23, va0, 5); \
    v30 = vfmaq_laneq_f16(v30, v23, va0, 6); \
    v31 = vfmaq_laneq_f16(v31, v23, va0, 7); \
    va0 = vld1q_f16(a + 8 * 9);              \
    v23 = vld1q_f16(b + 8 * 9);              \
    v24 = vfmaq_laneq_f16(v24, v23, va0, 0); \
    v25 = vfmaq_laneq_f16(v25, v23, va0, 1); \
    v26 = vfmaq_laneq_f16(v26, v23, va0, 2); \
    v27 = vfmaq_laneq_f16(v27, v23, va0, 3); \
    v28 = vfmaq_laneq_f16(v28, v23, va0, 4); \
    v29 = vfmaq_laneq_f16(v29, v23, va0, 5); \
    v30 = vfmaq_laneq_f16(v30, v23, va0, 6); \
    v31 = vfmaq_laneq_f16(v31, v23, va0, 7); \
    va0 = vld1q_f16(a + 8 * 10);             \
    v23 = vld1q_f16(b + 8 * 10);             \
    v24 = vfmaq_laneq_f16(v24, v23, va0, 0); \
    v25 = vfmaq_laneq_f16(v25, v23, va0, 1); \
    v26 = vfmaq_laneq_f16(v26, v23, va0, 2); \
    v27 = vfmaq_laneq_f16(v27, v23, va0, 3); \
    v28 = vfmaq_laneq_f16(v28, v23, va0, 4); \
    v29 = vfmaq_laneq_f16(v29, v23, va0, 5); \
    v30 = vfmaq_laneq_f16(v30, v23, va0, 6); \
    v31 = vfmaq_laneq_f16(v31, v23, va0, 7); \
    va0 = vld1q_f16(a + 8 * 11);             \
    v23 = vld1q_f16(b + 8 * 11);             \
    v24 = vfmaq_laneq_f16(v24, v23, va0, 0); \
    v25 = vfmaq_laneq_f16(v25, v23, va0, 1); \
    v26 = vfmaq_laneq_f16(v26, v23, va0, 2); \
    v27 = vfmaq_laneq_f16(v27, v23, va0, 3); \
    v28 = vfmaq_laneq_f16(v28, v23, va0, 4); \
    v29 = vfmaq_laneq_f16(v29, v23, va0, 5); \
    v30 = vfmaq_laneq_f16(v30, v23, va0, 6); \
    v31 = vfmaq_laneq_f16(v31, v23, va0, 7); \
    va0 = vld1q_f16(a + 8 * 12);             \
    v23 = vld1q_f16(b + 8 * 12);             \
    v24 = vfmaq_laneq_f16(v24, v23, va0, 0); \
    v25 = vfmaq_laneq_f16(v25, v23, va0, 1); \
    v26 = vfmaq_laneq_f16(v26, v23, va0, 2); \
    v27 = vfmaq_laneq_f16(v27, v23, va0, 3); \
    v28 = vfmaq_laneq_f16(v28, v23, va0, 4); \
    v29 = vfmaq_laneq_f16(v29, v23, va0, 5); \
    v30 = vfmaq_laneq_f16(v30, v23, va0, 6); \
    v31 = vfmaq_laneq_f16(v31, v23, va0, 7); \
    va0 = vld1q_f16(a + 8 * 13);             \
    v23 = vld1q_f16(b + 8 * 13);             \
    v24 = vfmaq_laneq_f16(v24, v23, va0, 0); \
    v25 = vfmaq_laneq_f16(v25, v23, va0, 1); \
    v26 = vfmaq_laneq_f16(v26, v23, va0, 2); \
    v27 = vfmaq_laneq_f16(v27, v23, va0, 3); \
    v28 = vfmaq_laneq_f16(v28, v23, va0, 4); \
    v29 = vfmaq_laneq_f16(v29, v23, va0, 5); \
    v30 = vfmaq_laneq_f16(v30, v23, va0, 6); \
    v31 = vfmaq_laneq_f16(v31, v23, va0, 7); \
    va0 = vld1q_f16(a + 8 * 14);             \
    v23 = vld1q_f16(b + 8 * 14);             \
    v24 = vfmaq_laneq_f16(v24, v23, va0, 0); \
    v25 = vfmaq_laneq_f16(v25, v23, va0, 1); \
    v26 = vfmaq_laneq_f16(v26, v23, va0, 2); \
    v27 = vfmaq_laneq_f16(v27, v23, va0, 3); \
    v28 = vfmaq_laneq_f16(v28, v23, va0, 4); \
    v29 = vfmaq_laneq_f16(v29, v23, va0, 5); \
    v30 = vfmaq_laneq_f16(v30, v23, va0, 6); \
    v31 = vfmaq_laneq_f16(v31, v23, va0, 7); \
    va0 = vld1q_f16(a + 8 * 15);             \
    v23 = vld1q_f16(b + 8 * 15);             \
    v24 = vfmaq_laneq_f16(v24, v23, va0, 0); \
    v25 = vfmaq_laneq_f16(v25, v23, va0, 1); \
    v26 = vfmaq_laneq_f16(v26, v23, va0, 2); \
    v27 = vfmaq_laneq_f16(v27, v23, va0, 3); \
    v28 = vfmaq_laneq_f16(v28, v23, va0, 4); \
    v29 = vfmaq_laneq_f16(v29, v23, va0, 5); \
    v30 = vfmaq_laneq_f16(v30, v23, va0, 6); \
    v31 = vfmaq_laneq_f16(v31, v23, va0, 7); \
    __builtin_prefetch(b + 128, 0, 3);       \
    __builtin_prefetch(a + 128, 0, 3);       \
    l += 16;                                 \
    b += 8 * 16;                             \
    a += 8 * 16;                             \
  } while (0)

// 2. Partial sum 512 digits
#define KERNEL_8x8_ACC8()                    \
  do {                                       \
    va0 = vld1q_f16(a);                      \
    v16 = vld1q_f16(b);                      \
    v24 = vfmaq_laneq_f16(v24, v16, va0, 0); \
    v25 = vfmaq_laneq_f16(v25, v16, va0, 1); \
    v26 = vfmaq_laneq_f16(v26, v16, va0, 2); \
    v27 = vfmaq_laneq_f16(v27, v16, va0, 3); \
    v28 = vfmaq_laneq_f16(v28, v16, va0, 4); \
    v29 = vfmaq_laneq_f16(v29, v16, va0, 5); \
    v30 = vfmaq_laneq_f16(v30, v16, va0, 6); \
    v31 = vfmaq_laneq_f16(v31, v16, va0, 7); \
    va1 = vld1q_f16(a + 8);                  \
    v17 = vld1q_f16(b + 8);                  \
    v24 = vfmaq_laneq_f16(v24, v17, va1, 0); \
    v25 = vfmaq_laneq_f16(v25, v17, va1, 1); \
    v26 = vfmaq_laneq_f16(v26, v17, va1, 2); \
    v27 = vfmaq_laneq_f16(v27, v17, va1, 3); \
    v28 = vfmaq_laneq_f16(v28, v17, va1, 4); \
    v29 = vfmaq_laneq_f16(v29, v17, va1, 5); \
    v30 = vfmaq_laneq_f16(v30, v17, va1, 6); \
    v31 = vfmaq_laneq_f16(v31, v17, va1, 7); \
    va2 = vld1q_f16(a + 16);                 \
    v18 = vld1q_f16(b + 16);                 \
    v24 = vfmaq_laneq_f16(v24, v18, va2, 0); \
    v25 = vfmaq_laneq_f16(v25, v18, va2, 1); \
    v26 = vfmaq_laneq_f16(v26, v18, va2, 2); \
    v27 = vfmaq_laneq_f16(v27, v18, va2, 3); \
    v28 = vfmaq_laneq_f16(v28, v18, va2, 4); \
    v29 = vfmaq_laneq_f16(v29, v18, va2, 5); \
    v30 = vfmaq_laneq_f16(v30, v18, va2, 6); \
    v31 = vfmaq_laneq_f16(v31, v18, va2, 7); \
    va3 = vld1q_f16(a + 24);                 \
    v19 = vld1q_f16(b + 24);                 \
    v24 = vfmaq_laneq_f16(v24, v19, va3, 0); \
    v25 = vfmaq_laneq_f16(v25, v19, va3, 1); \
    v26 = vfmaq_laneq_f16(v26, v19, va3, 2); \
    v27 = vfmaq_laneq_f16(v27, v19, va3, 3); \
    v28 = vfmaq_laneq_f16(v28, v19, va3, 4); \
    v29 = vfmaq_laneq_f16(v29, v19, va3, 5); \
    v30 = vfmaq_laneq_f16(v30, v19, va3, 6); \
    v31 = vfmaq_laneq_f16(v31, v19, va3, 7); \
    va4 = vld1q_f16(a + 32);                 \
    v20 = vld1q_f16(b + 32);                 \
    v24 = vfmaq_laneq_f16(v24, v20, va4, 0); \
    v25 = vfmaq_laneq_f16(v25, v20, va4, 1); \
    v26 = vfmaq_laneq_f16(v26, v20, va4, 2); \
    v27 = vfmaq_laneq_f16(v27, v20, va4, 3); \
    v28 = vfmaq_laneq_f16(v28, v20, va4, 4); \
    v29 = vfmaq_laneq_f16(v29, v20, va4, 5); \
    v30 = vfmaq_laneq_f16(v30, v20, va4, 6); \
    v31 = vfmaq_laneq_f16(v31, v20, va4, 7); \
    va5 = vld1q_f16(a + 40);                 \
    v21 = vld1q_f16(b + 40);                 \
    v24 = vfmaq_laneq_f16(v24, v21, va5, 0); \
    v25 = vfmaq_laneq_f16(v25, v21, va5, 1); \
    v26 = vfmaq_laneq_f16(v26, v21, va5, 2); \
    v27 = vfmaq_laneq_f16(v27, v21, va5, 3); \
    v28 = vfmaq_laneq_f16(v28, v21, va5, 4); \
    v29 = vfmaq_laneq_f16(v29, v21, va5, 5); \
    v30 = vfmaq_laneq_f16(v30, v21, va5, 6); \
    v31 = vfmaq_laneq_f16(v31, v21, va5, 7); \
    va6 = vld1q_f16(a + 48);                 \
    v22 = vld1q_f16(b + 48);                 \
    v24 = vfmaq_laneq_f16(v24, v22, va6, 0); \
    v25 = vfmaq_laneq_f16(v25, v22, va6, 1); \
    v26 = vfmaq_laneq_f16(v26, v22, va6, 2); \
    v27 = vfmaq_laneq_f16(v27, v22, va6, 3); \
    v28 = vfmaq_laneq_f16(v28, v22, va6, 4); \
    v29 = vfmaq_laneq_f16(v29, v22, va6, 5); \
    v30 = vfmaq_laneq_f16(v30, v22, va6, 6); \
    v31 = vfmaq_laneq_f16(v31, v22, va6, 7); \
    va7 = vld1q_f16(a + 56);                 \
    v23 = vld1q_f16(b + 56);                 \
    v24 = vfmaq_laneq_f16(v24, v23, va7, 0); \
    v25 = vfmaq_laneq_f16(v25, v23, va7, 1); \
    v26 = vfmaq_laneq_f16(v26, v23, va7, 2); \
    v27 = vfmaq_laneq_f16(v27, v23, va7, 3); \
    v28 = vfmaq_laneq_f16(v28, v23, va7, 4); \
    v29 = vfmaq_laneq_f16(v29, v23, va7, 5); \
    v30 = vfmaq_laneq_f16(v30, v23, va7, 6); \
    v31 = vfmaq_laneq_f16(v31, v23, va7, 7); \
    __builtin_prefetch(b + 64, 0, 3);        \
    __builtin_prefetch(a + 64, 0, 3);        \
    l += 8;                                  \
    b += 8 * 8;                              \
    a += 8 * 8;                              \
  } while (0)

// 3. Partial sum 256 digits
#define KERNEL_8x8_ACC4()                    \
  do {                                       \
    va0 = vld1q_f16(a);                      \
    v16 = vld1q_f16(b);                      \
    v24 = vfmaq_laneq_f16(v24, v16, va0, 0); \
    v25 = vfmaq_laneq_f16(v25, v16, va0, 1); \
    v26 = vfmaq_laneq_f16(v26, v16, va0, 2); \
    v27 = vfmaq_laneq_f16(v27, v16, va0, 3); \
    v28 = vfmaq_laneq_f16(v28, v16, va0, 4); \
    v29 = vfmaq_laneq_f16(v29, v16, va0, 5); \
    v30 = vfmaq_laneq_f16(v30, v16, va0, 6); \
    v31 = vfmaq_laneq_f16(v31, v16, va0, 7); \
    va1 = vld1q_f16(a + 8);                  \
    v17 = vld1q_f16(b + 8);                  \
    v24 = vfmaq_laneq_f16(v24, v17, va1, 0); \
    v25 = vfmaq_laneq_f16(v25, v17, va1, 1); \
    v26 = vfmaq_laneq_f16(v26, v17, va1, 2); \
    v27 = vfmaq_laneq_f16(v27, v17, va1, 3); \
    v28 = vfmaq_laneq_f16(v28, v17, va1, 4); \
    v29 = vfmaq_laneq_f16(v29, v17, va1, 5); \
    v30 = vfmaq_laneq_f16(v30, v17, va1, 6); \
    v31 = vfmaq_laneq_f16(v31, v17, va1, 7); \
    va2 = vld1q_f16(a + 16);                 \
    v18 = vld1q_f16(b + 16);                 \
    v24 = vfmaq_laneq_f16(v24, v18, va2, 0); \
    v25 = vfmaq_laneq_f16(v25, v18, va2, 1); \
    v26 = vfmaq_laneq_f16(v26, v18, va2, 2); \
    v27 = vfmaq_laneq_f16(v27, v18, va2, 3); \
    v28 = vfmaq_laneq_f16(v28, v18, va2, 4); \
    v29 = vfmaq_laneq_f16(v29, v18, va2, 5); \
    v30 = vfmaq_laneq_f16(v30, v18, va2, 6); \
    v31 = vfmaq_laneq_f16(v31, v18, va2, 7); \
    va3 = vld1q_f16(a + 24);                 \
    v19 = vld1q_f16(b + 24);                 \
    v24 = vfmaq_laneq_f16(v24, v19, va3, 0); \
    v25 = vfmaq_laneq_f16(v25, v19, va3, 1); \
    v26 = vfmaq_laneq_f16(v26, v19, va3, 2); \
    v27 = vfmaq_laneq_f16(v27, v19, va3, 3); \
    v28 = vfmaq_laneq_f16(v28, v19, va3, 4); \
    v29 = vfmaq_laneq_f16(v29, v19, va3, 5); \
    v30 = vfmaq_laneq_f16(v30, v19, va3, 6); \
    v31 = vfmaq_laneq_f16(v31, v19, va3, 7); \
    __builtin_prefetch(b + 32, 0, 3);        \
    __builtin_prefetch(a + 32, 0, 3);        \
    l += 4;                                  \
    b += 8 * 4;                              \
    a += 8 * 4;                              \
  } while (0)

// 4. Partial sum 64 digits
#define KERNEL_8x8_ACC1()                    \
  do {                                       \
    va0 = vld1q_f16(a);                      \
    v16 = vld1q_f16(b);                      \
    v24 = vfmaq_laneq_f16(v24, v16, va0, 0); \
    v25 = vfmaq_laneq_f16(v25, v16, va0, 1); \
    v26 = vfmaq_laneq_f16(v26, v16, va0, 2); \
    v27 = vfmaq_laneq_f16(v27, v16, va0, 3); \
    v28 = vfmaq_laneq_f16(v28, v16, va0, 4); \
    v29 = vfmaq_laneq_f16(v29, v16, va0, 5); \
    v30 = vfmaq_laneq_f16(v30, v16, va0, 6); \
    v31 = vfmaq_laneq_f16(v31, v16, va0, 7); \
    __builtin_prefetch(b + 8, 0, 3);         \
    __builtin_prefetch(a + 8, 0, 3);         \
    l += 1;                                  \
    b += 8 * 1;                              \
    a += 8 * 1;                              \
  } while (0)

#define SAVE_KERNEL_8X8_F16_f32()                                              \
  do {                                                                         \
    vst1q_f32(c, vaddq_f32(vld1q_f32(c), vcvt_f32_f16(vget_low_f16(v24))));    \
    vst1q_f32(c + 4,                                                           \
              vaddq_f32(vld1q_f32(c + 4), vcvt_f32_f16(vget_high_f16(v24))));  \
                                                                               \
    vst1q_f32(c + ldc,                                                         \
              vaddq_f32(vld1q_f32(c + ldc), vcvt_f32_f16(vget_low_f16(v25)))); \
    vst1q_f32(c + 4 + ldc, vaddq_f32(vld1q_f32(c + 4 + ldc),                   \
                                     vcvt_f32_f16(vget_high_f16(v25))));       \
                                                                               \
    vst1q_f32(c + 2 * ldc, vaddq_f32(vld1q_f32(c + 2 * ldc),                   \
                                     vcvt_f32_f16(vget_low_f16(v26))));        \
    vst1q_f32(c + 4 + 2 * ldc, vaddq_f32(vld1q_f32(c + 4 + 2 * ldc),           \
                                         vcvt_f32_f16(vget_high_f16(v26))));   \
                                                                               \
    vst1q_f32(c + 3 * ldc, vaddq_f32(vld1q_f32(c + 3 * ldc),                   \
                                     vcvt_f32_f16(vget_low_f16(v27))));        \
    vst1q_f32(c + 4 + 3 * ldc, vaddq_f32(vld1q_f32(c + 4 + 3 * ldc),           \
                                         vcvt_f32_f16(vget_high_f16(v27))));   \
                                                                               \
    vst1q_f32(c + 4 * ldc, vaddq_f32(vld1q_f32(c + 4 * ldc),                   \
                                     vcvt_f32_f16(vget_low_f16(v28))));        \
    vst1q_f32(c + 4 + 4 * ldc, vaddq_f32(vld1q_f32(c + 4 + 4 * ldc),           \
                                         vcvt_f32_f16(vget_high_f16(v28))));   \
                                                                               \
    vst1q_f32(c + 5 * ldc, vaddq_f32(vld1q_f32(c + 5 * ldc),                   \
                                     vcvt_f32_f16(vget_low_f16(v29))));        \
    vst1q_f32(c + 4 + 5 * ldc, vaddq_f32(vld1q_f32(c + 4 + 5 * ldc),           \
                                         vcvt_f32_f16(vget_high_f16(v29))));   \
                                                                               \
    vst1q_f32(c + 6 * ldc, vaddq_f32(vld1q_f32(c + 6 * ldc),                   \
                                     vcvt_f32_f16(vget_low_f16(v30))));        \
    vst1q_f32(c + 4 + 6 * ldc, vaddq_f32(vld1q_f32(c + 4 + 6 * ldc),           \
                                         vcvt_f32_f16(vget_high_f16(v30))));   \
                                                                               \
    vst1q_f32(c + 7 * ldc, vaddq_f32(vld1q_f32(c + 7 * ldc),                   \
                                     vcvt_f32_f16(vget_low_f16(v31))));        \
    vst1q_f32(c + 4 + 7 * ldc, vaddq_f32(vld1q_f32(c + 4 + 7 * ldc),           \
                                         vcvt_f32_f16(vget_high_f16(v31))));   \
  } while (0)

template <>
void hgemm_kernel_8x8(unsigned int M, unsigned int N, unsigned int K,
                      __fp16 *sa, __fp16 *sb, __fp16 *sc, unsigned int ldc) {
  assert(M > 0 && N > 0 && K > 0);
  assert(M % 8 == 0 && N % 8 == 0 && K % 4 == 0);

  __fp16 *a = sa, *b = sb, *c = sc;
  unsigned int i, j, l;
  for (i = 0; i < M; i += 8) {
    for (j = 0; j < N; j += 8) {
      __builtin_prefetch(b, 0, 3);
      __builtin_prefetch(a, 0, 3);

      float16x8_t v16, v17, v18, v19, v20, v21, v22, v23;
      float16x8_t v24, v25, v26, v27, v28, v29, v30, v31;
      float16x8_t va0, va1, va2, va3, va4, va5, va6, va7;
      INIT_KERNEL_8x8();
      l = 0;
      for (; l < K;) {
        KERNEL_8x8_ACC1();
      }
      vst1q_f16(c, vaddq_f16(vld1q_f16(c), v24));
      vst1q_f16(c + ldc, vaddq_f16(vld1q_f16(c + ldc), v25));
      vst1q_f16(c + 2 * ldc, vaddq_f16(vld1q_f16(c + 2 * ldc), v26));
      vst1q_f16(c + 3 * ldc, vaddq_f16(vld1q_f16(c + 3 * ldc), v27));
      vst1q_f16(c + 4 * ldc, vaddq_f16(vld1q_f16(c + 4 * ldc), v28));
      vst1q_f16(c + 5 * ldc, vaddq_f16(vld1q_f16(c + 5 * ldc), v29));
      vst1q_f16(c + 6 * ldc, vaddq_f16(vld1q_f16(c + 6 * ldc), v30));
      vst1q_f16(c + 7 * ldc, vaddq_f16(vld1q_f16(c + 7 * ldc), v31));
      c += 8;
      a -= 8 * K;
    }
    sc += ldc * 8;
    c = sc;
    a += 8 * K;
    b = sb;
  }
}

template <>
void hgemm_kernel_8x8(unsigned int M, unsigned int N, unsigned int K,
                      __fp16 *sa, __fp16 *sb, float *sc, unsigned int ldc) {
  assert(M > 0 && N > 0 && K > 0);
  assert(M % 8 == 0 && N % 8 == 0 && K % 8 == 0);

  __fp16 *a = sa, *b = sb;
  float *c = sc;
  unsigned int i, j, l;
  unsigned int K4 = (K >> 2) << 2;
  unsigned int K8 = (K >> 3) << 3;
  unsigned int K16 = (K >> 4) << 4;
  for (i = 0; i < M; i += 8) {
    for (j = 0; j < N; j += 8) {
      __builtin_prefetch(b, 0, 3);
      __builtin_prefetch(a, 0, 3);

      float16x8_t v16, v17, v18, v19, v20, v21, v22, v23;
      float16x8_t v24, v25, v26, v27, v28, v29, v30, v31;
      float16x8_t va0, va1, va2, va3, va4, va5, va6, va7;
      l = 0;
      for (; l < K16;) {
        INIT_KERNEL_8x8();
        KERNEL_8x8_ACC16();
        SAVE_KERNEL_8X8_F16_f32();
      }
      for (; l < K8;) {
        INIT_KERNEL_8x8();
        KERNEL_8x8_ACC8();
        SAVE_KERNEL_8X8_F16_f32();
      }
      for (; l < K4;) {
        INIT_KERNEL_8x8();
        KERNEL_8x8_ACC4();
        SAVE_KERNEL_8X8_F16_f32();
      }
      for (; l < K;) {
        INIT_KERNEL_8x8();
        KERNEL_8x8_ACC1();
        SAVE_KERNEL_8X8_F16_f32();
      }

      c += 8;
      a -= 8 * K;
    }
    sc += ldc * 8;
    c = sc;
    a += 8 * K;
    b = sb;
  }
}
