// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Donghyeon Jeong <dhyeon.jeong@samsung.com>
 *
 * @file   avx2_impl.cpp
 * @date   20 Feb 2024
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Donghyeon Jeong <dhyeon.jeong@samsung.com>
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is a source for AVX implementation
 *
 */

#include <avx2_impl.h>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <immintrin.h>
#include <limits>

namespace nntrainer::avx2 {

bool is_valid(const unsigned int N, const float *input) {
  assert(N != 0);
  assert(input != NULL);

  int temp = 0;
  unsigned int idx = 0;

  const __m256 SIGN_MASK = _mm256_set1_ps(-0.0);
  const __m256 INF = _mm256_set1_ps(std::numeric_limits<float>::infinity());

  // 16 single-precision check : ( X != X )
  for (; N - idx >= 16; idx += 16) {
    __m256 vec0 = _mm256_loadu_ps(input);
    __m256 vec1 = _mm256_loadu_ps(input + 8);
    input += 16;
    __m256 res = _mm256_cmp_ps(vec0, vec0, _CMP_NEQ_UQ);
    temp = temp | _mm256_movemask_ps(res);

    if (temp)
      return false;

    // check infinity in vec0
    vec0 = _mm256_andnot_ps(SIGN_MASK, vec0);
    vec0 = _mm256_cmp_ps(vec0, INF, _CMP_EQ_OQ);

    temp = temp | _mm256_movemask_ps(vec0);
    if (temp)
      return false;

    __m256 res1 = _mm256_cmp_ps(vec1, vec1, _CMP_NEQ_UQ);
    temp = temp | _mm256_movemask_ps(res1);

    if (temp)
      return false;

    // check infinity in vec1
    vec1 = _mm256_andnot_ps(SIGN_MASK, vec1);
    vec1 = _mm256_cmp_ps(vec1, INF, _CMP_EQ_OQ);

    temp = temp | _mm256_movemask_ps(vec1);

    if (temp)
      return false;
  }

  // 8 single-precision check : ( X != X )
  for (; N - idx >= 8; idx += 8) {
    __m256 vec = _mm256_loadu_ps(input);
    input += 8;
    __m256 res = _mm256_cmp_ps(vec, vec, _CMP_NEQ_UQ);
    temp = temp | _mm256_movemask_ps(res);

    if (temp)
      return false;

    // check infinity in vec
    vec = _mm256_andnot_ps(SIGN_MASK, vec);
    vec = _mm256_cmp_ps(vec, INF, _CMP_EQ_OQ);

    temp = temp | _mm256_movemask_ps(vec);

    if (temp)
      return false;
  }

  // remain check : ( X != X )
  while (idx < N) {
    if (*input != *input || *input == std::numeric_limits<float>::infinity()) {
      return false;
    }
    ++input;
    ++idx;
  }

  return true;
}

void custom_scopy(const unsigned int N, const float *X, const int incX,
                  float *Y, const int incY) {
  unsigned int N8 = (N >> 3) << 3;
  for (unsigned int i = 0; i < N8; i += 8) {
#if defined(_WIN32)
    __m256 temp = _mm256_loadu_ps(&X[i]);
    _mm256_storeu_ps(&Y[i], temp);
#else
    __asm__ __volatile__("vmovups (%1), %%ymm0\n\t"
                         "vmovups %%ymm0, (%0)\n\t"
                         :
                         : "r"(&Y[i]), "r"(&X[i])
                         : "ymm0", "memory");
#endif
  }
  for (unsigned int i = N8; i < N; ++i) {
    Y[i] = X[i];
  }
}

void transpose_matrix(const unsigned int M, const unsigned int N,
                      const float *src, unsigned int ld_src, float *dst,
                      unsigned int ld_dst) {
  unsigned int vindexm[8] = {0,          ld_src,     ld_src * 2, ld_src * 3,
                             ld_src * 4, ld_src * 5, ld_src * 6, ld_src * 7};
  __m256i vindex = _mm256_loadu_si256((__m256i *)&vindexm[0]);
  __m256 vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8;

  unsigned int M8 = (M & ~(7));
  unsigned int N8 = (N & ~(7));
  for (unsigned int i = 0; i < M8; i += 8) {
    for (unsigned int j = 0; j < N8; j += 8) {
      // loading from columns
      vec1 = _mm256_i32gather_ps(&src[ld_src * i + j + 0], vindex, 4);
      vec2 = _mm256_i32gather_ps(&src[ld_src * i + j + 1], vindex, 4);
      vec3 = _mm256_i32gather_ps(&src[ld_src * i + j + 2], vindex, 4);
      vec4 = _mm256_i32gather_ps(&src[ld_src * i + j + 3], vindex, 4);
      vec5 = _mm256_i32gather_ps(&src[ld_src * i + j + 4], vindex, 4);
      vec6 = _mm256_i32gather_ps(&src[ld_src * i + j + 5], vindex, 4);
      vec7 = _mm256_i32gather_ps(&src[ld_src * i + j + 6], vindex, 4);
      vec8 = _mm256_i32gather_ps(&src[ld_src * i + j + 7], vindex, 4);

      // storing to the rows
      _mm256_storeu_ps(&dst[(j + 0) * ld_dst + i], vec1);
      _mm256_storeu_ps(&dst[(j + 1) * ld_dst + i], vec2);
      _mm256_storeu_ps(&dst[(j + 2) * ld_dst + i], vec3);
      _mm256_storeu_ps(&dst[(j + 3) * ld_dst + i], vec4);
      _mm256_storeu_ps(&dst[(j + 4) * ld_dst + i], vec5);
      _mm256_storeu_ps(&dst[(j + 5) * ld_dst + i], vec6);
      _mm256_storeu_ps(&dst[(j + 6) * ld_dst + i], vec7);
      _mm256_storeu_ps(&dst[(j + 7) * ld_dst + i], vec8);
    }
  }

  // tailing right
  for (unsigned int i = 0; i < M; i++) {
    for (unsigned int j = N8; j < N; j++) {
      dst[i + j * ld_dst] = src[i * ld_src + j];
    }
  }

  // tailing bottom
  for (unsigned int i = M8; i < M; i++) {
    for (unsigned int j = 0; j < N; j++) {
      dst[i + j * ld_dst] = src[i * ld_src + j];
    }
  }
}

} // namespace nntrainer::avx2
