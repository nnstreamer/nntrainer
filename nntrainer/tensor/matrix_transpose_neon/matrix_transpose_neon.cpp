// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file   matrix_transpose_neon.cpp
 * @date   09 May 2024
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is source file for matrix transpose using NEON
 *
 */

#include <arm_neon.h>
#include "./transpose_utils_neon.h"
#include "./matrix_transpose_neon.h"

template <>
void transpose_fallback(
    unsigned int M,
    unsigned int N,
    const __fp16* src,
    unsigned int ld_src,
    __fp16* dst,
    unsigned int ld_dst) {
  for (unsigned int j = 0; j < N; j++) {
    for (unsigned int i = 0; i < M; i++) {
      dst[i + j * ld_dst] = src[i * ld_src + j];
    }
  }
}

template <>
void transpose_neon(
    unsigned int M,
    unsigned int N,
    const __fp16* src,
    unsigned int ld_src,
    __fp16* dst,
    unsigned int ld_dst) {
  unsigned int ib = 0, jb = 0;
  if (N % 8 > 0 && N % 8 < 4) {
    for (ib = 0; ib + 8 <= M; ib += 8) {
      for (jb = 0; jb + 8 <= N; jb += 8) {
        transpose_kernel_8x8_neon(
            &src[ib * ld_src + jb], ld_src, &dst[ib + jb * ld_dst], ld_dst);
      }
      for (unsigned int i = ib; i < ib + 8; i += 4) {
        transpose_kernel_mxn_neon_128<4>(
            N - jb,
            &src[i * ld_src + jb],
            ld_src,
            &dst[i + jb * ld_dst],
            ld_dst);
      }
    }
  } else if (N % 8 == 4) {
    for (ib = 0; ib + 8 <= M; ib += 8) {
      for (jb = 0; jb + 8 <= N; jb += 8) {
        transpose_kernel_8x8_neon(
            &src[ib * ld_src + jb], ld_src, &dst[ib + jb * ld_dst], ld_dst);
      }
      for (unsigned int i = ib; i < ib + 8; i += 4) {
        transpose_kernel_4x4_neon(
            &src[i * ld_src + jb], ld_src, &dst[i + jb * ld_dst], ld_dst);
      }
    }
  } else {
    for (ib = 0; ib + 8 <= M; ib += 8) {
      for (jb = 0; jb + 8 <= N; jb += 8) {
        transpose_kernel_8x8_neon(
            &src[ib * ld_src + jb], ld_src, &dst[ib + jb * ld_dst], ld_dst);
      }
      if (jb < N) {
        transpose_kernel_mxn_neon_256<8>(
            N - jb,
            &src[ib * ld_src + jb],
            ld_src,
            &dst[ib + jb * ld_dst],
            ld_dst);
      }
    }
  }
  switch (M - ib) {
    case 1:
      for (unsigned int j = 0; j < N; ++j) {
        dst[ib + j * ld_dst] = src[ib * ld_src + j];
      }
      break;
    case 2:
      for (jb = 0; jb + 4 <= N; jb += 4) {
        transpose_kernel_mxn_neon_128<2>(
            4, &src[ib * ld_src + jb], ld_src, &dst[ib + jb * ld_dst], ld_dst);
      }
      if (jb < N) {
        transpose_kernel_mxn_neon_128<2>(
            N - jb,
            &src[ib * ld_src + jb],
            ld_src,
            &dst[ib + jb * ld_dst],
            ld_dst);
      }
      break;
    case 3:
      for (jb = 0; jb + 4 <= N; jb += 4) {
        transpose_kernel_mxn_neon_128<3>(
            4, &src[ib * ld_src + jb], ld_src, &dst[ib + jb * ld_dst], ld_dst);
      }
      if (jb < N) {
        transpose_kernel_mxn_neon_128<3>(
            N - jb,
            &src[ib * ld_src + jb],
            ld_src,
            &dst[ib + jb * ld_dst],
            ld_dst);
      }
      break;
    case 4:
      for (jb = 0; jb + 4 <= N; jb += 4) {
        transpose_kernel_4x4_neon(
            &src[ib * ld_src + jb], ld_src, &dst[ib + jb * ld_dst], ld_dst);
      }
      if (jb < N) {
        transpose_kernel_mxn_neon_128<4>(
            N - jb,
            &src[ib * ld_src + jb],
            ld_src,
            &dst[ib + jb * ld_dst],
            ld_dst);
      }
      break;
    case 5:
      for (jb = 0; jb + 8 <= N; jb += 8) {
        transpose_kernel_mxn_neon_256<5>(
            8, &src[ib * ld_src + jb], ld_src, &dst[ib + jb * ld_dst], ld_dst);
      }
      if (jb < N) {
        transpose_kernel_mxn_neon_256<5>(
            N - jb,
            &src[ib * ld_src + jb],
            ld_src,
            &dst[ib + jb * ld_dst],
            ld_dst);
      }
      break;
    case 6:
      for (jb = 0; jb + 8 <= N; jb += 8) {
        transpose_kernel_mxn_neon_256<6>(
            8, &src[ib * ld_src + jb], ld_src, &dst[ib + jb * ld_dst], ld_dst);
      }
      if (jb < N) {
        transpose_kernel_mxn_neon_256<6>(
            N - jb,
            &src[ib * ld_src + jb],
            ld_src,
            &dst[ib + jb * ld_dst],
            ld_dst);
      }
      break;
    case 7:
      for (jb = 0; jb + 8 <= N; jb += 8) {
        transpose_kernel_mxn_neon_256<7>(
            8, &src[ib * ld_src + jb], ld_src, &dst[ib + jb * ld_dst], ld_dst);
      }
      if (jb < N) {
        transpose_kernel_mxn_neon_256<7>(
            N - jb,
            &src[ib * ld_src + jb],
            ld_src,
            &dst[ib + jb * ld_dst],
            ld_dst);
      }
      break;
  }
}
