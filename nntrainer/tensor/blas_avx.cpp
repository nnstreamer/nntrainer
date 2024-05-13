// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Donghyeon Jeong <dhyeon.jeong@samsung.com>
 *
 * @file   blas_avx.cpp
 * @date   20 Feb 2024
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Donghyeon Jeong <dhyeon.jeong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is a source for AVX implementation
 *
 */

#include <cassert>
#include <chrono>
#include <cstdint>
#include <immintrin.h>

#include <blas_avx.h>

namespace nntrainer::avx {

#ifdef ENABLE_FP16
void vcvt_f16_f32(size_t N, const void *input, float *output) {
  assert(N != 0);
  assert(input != NULL);
  assert(output != NULL);

  unsigned int idx = 0;
  const _Float16 *data = (const _Float16 *)input;

  // 16 half-precision floating point values to single-precision values
  for (; N - idx >= 16; idx += 16) {
    const __m256 vec0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)data));
    const __m256 vec1 =
      _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(data + 8)));
    data += 16;

    _mm256_storeu_ps(output, vec0);
    _mm256_storeu_ps(output + 8, vec1);
    output += 16;
  }
  // 8 half-precision floating point values to single-precision values
  for (; N - idx >= 8; idx += 8) {
    const __m256 vec = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)data));
    data += 8;

    _mm256_storeu_ps(output, vec);
    output += 8;
  }
  // remaining half-precision floating point values to single-precision values
  if (N - idx > 0 && N - idx < 8) {
    const __m256 vec = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)data));
    __m128 s_vec = _mm256_castps256_ps128(vec);
    if ((N - idx) & 4) {
      _mm_storeu_ps(output, s_vec);
      s_vec = _mm256_extractf128_ps(vec, 1);
      output += 4;
    }
    if ((N - idx) & 2) {
      _mm_storel_pi((__m64 *)output, s_vec);
      s_vec = _mm_movehl_ps(s_vec, s_vec);
      output += 2;
    }
    if ((N - idx) & 1) {
      _mm_store_ss(output, s_vec);
    }
  }
}

void vcvt_f32_f16(size_t N, const float *input, void *output) {
  assert(N != 0);
  assert(input != NULL);
  assert(output != NULL);

  unsigned int idx = 0;
  _Float16 *out_data = (_Float16 *)output;

  // 16 single-precision floating point values to half-precision values
  for (; N - idx >= 16; idx += 16) {
    const __m256 vec0 = _mm256_loadu_ps(input);
    const __m256 vec1 = _mm256_loadu_ps(input + 8);
    input += 16;

    _mm_storeu_si128((__m128i *)out_data,
                     _mm256_cvtps_ph(vec0, _MM_FROUND_TO_NEAREST_INT));
    _mm_storeu_si128((__m128i *)(out_data + 8),
                     _mm256_cvtps_ph(vec1, _MM_FROUND_TO_NEAREST_INT));
    out_data += 16;
  }
  // 8 single-precision floating point values to half-precision values
  for (; N - idx >= 8; idx += 8) {
    const __m256 vec = _mm256_loadu_ps(input);
    input += 8;

    _mm_storeu_si128((__m128i *)out_data,
                     _mm256_cvtps_ph(vec, _MM_FROUND_TO_NEAREST_INT));
    out_data += 8;
  }
  // 4 single-precision floating point values to half-precision values
  for (; N - idx >= 4; idx += 4) {
    const __m128 vec = _mm_loadu_ps(input);
    input += 4;

    _mm_storeu_si64((__m128i *)out_data,
                    _mm_cvtps_ph(vec, _MM_FROUND_TO_NEAREST_INT));
    out_data += 4;
  }
  // remaining single-precision floating point values to half-precision values
  while (idx < N) {
    *out_data = static_cast<_Float16>(*input);
    ++out_data;
    ++input;
    ++idx;
  }
}

bool hasNaN(const size_t N, const _Float16 *input) {
  assert(N != 0);
  assert(input != NULL);

  int temp = 0;
  size_t idx = 0;

  // 16 single-precision check : ( X != X )
  for (; N - idx >= 16; idx += 16) {
    const __m256 vec0 =
      _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)input));
    const __m256 vec1 =
      _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(input + 8)));

    input += 16;

    __m256 res = _mm256_cmp_ps(vec0, vec0, _CMP_NEQ_UQ);
    temp = temp | _mm256_movemask_ps(res);

    if (temp)
      return true;

    __m256 res1 = _mm256_cmp_ps(vec1, vec1, _CMP_NEQ_UQ);
    temp = temp | _mm256_movemask_ps(res1);

    if (temp)
      return true;
  }

  // 8 single-precision check : ( X != X )
  for (; N - idx >= 8; idx += 8) {
    const __m256 vec = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)input));
    input += 8;
    __m256 res = _mm256_cmp_ps(vec, vec, _CMP_NEQ_UQ);
    temp = temp | _mm256_movemask_ps(res);

    if (temp)
      return true;
  }

  // remain check : ( X != X )
  while (idx < N) {
    if (*input != *input) {
      return true;
    }
    ++input;
    ++idx;
  }

  return false;
}
#endif

bool hasNaN(const size_t N, const float *input) {
  assert(N != 0);
  assert(input != NULL);

  int temp = 0;
  size_t idx = 0;

  // 16 single-precision check : ( X != X )
  for (; N - idx >= 16; idx += 16) {
    const __m256 vec0 = _mm256_loadu_ps(input);
    const __m256 vec1 = _mm256_loadu_ps(input + 8);
    input += 16;
    __m256 res = _mm256_cmp_ps(vec0, vec0, _CMP_NEQ_UQ);
    temp = temp | _mm256_movemask_ps(res);
    __m256 res1 = _mm256_cmp_ps(vec1, vec1, _CMP_NEQ_UQ);
    temp = temp | _mm256_movemask_ps(res1);

    if (temp)
      return true;
  }

  // 8 single-precision check : ( X != X )
  for (; N - idx >= 8; idx += 8) {
    const __m256 vec = _mm256_loadu_ps(input);
    input += 8;
    __m256 res = _mm256_cmp_ps(vec, vec, _CMP_NEQ_UQ);
    temp = temp | _mm256_movemask_ps(res);

    if (temp)
      return true;
  }

  // remain check : ( X != X )
  while (idx < N) {
    if (*input != *input) {
      return true;
    }
    ++input;
    ++idx;
  }

  return false;
}

} // namespace nntrainer::avx
