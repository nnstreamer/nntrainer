// SPDX-License-Identifier: Apache-2.0
/**
 * @file	util_simd.cpp
 * @date	09 Jan 2024
 * @brief	This is a collection of simd util functions
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Sungsik Kong <ss.kong@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#include <algorithm>
#include <cmath>
#include <util_simd.h>
#ifdef USE_NEON
#include <util_simd_neon.h>
#endif

namespace nntrainer {

void calc_trigonometric_vals_dup(unsigned int N_half, float *angle, float *cos_,
                                 float *sin_, unsigned int alpha) {
#ifdef USE_NEON
  nntrainer::neon::calc_trigonometric_vals_dup(N_half, angle, cos_, sin_,
                                               alpha);
#else
  throw std::invalid_argument(
    "Error: No implementation of rotary embedding layer incremental_forwarding "
    "with SIMD acceleration except for NEON!");
#endif
}

void swiglu(const unsigned int N, float *X, float *Y, float *Z) {
#ifdef USE_NEON
  nntrainer::neon::swiglu(N, X, Y, Z);
#else
  unsigned int i = 0;
  while (i < N) {
    X[i] = (Y[i] / (1.f + std::exp(-Y[i]))) * Z[i];
    ++i;
  }
#endif
}

float max(const unsigned int N, float *X) {
#ifdef USE_NEON
  return nntrainer::neon::max(N, X);
#else
  std::vector<float> v(X, X + N);
  return *std::max_element(v.begin(), v.end());
#endif
}

void softmax(const unsigned int N, float *X, float *Y) {
#ifdef USE_NEON
  nntrainer::neon::softmax(N, X, Y);
#else
  unsigned int i = 0;
  float sum = 0.f;
  float max_x = max(N, X);
  while (i < N) {
    sum += std::exp(X[i] - max_x);
    ++i;
  }
  i = 0;
  while (i < N) {
    Y[i] = std::exp(X[i] - max_x) / sum;
    ++i;
  }
#endif
}

#ifdef ENABLE_FP16

void compute_rotary_embedding_value(unsigned int dim, unsigned int half_,
                                    unsigned int w, _FP16 *in, _FP16 *out,
                                    float *cos_, float *sin_) {
#ifdef USE_NEON
  nntrainer::neon::compute_rotary_embedding_value(dim, half_, w, in, out, cos_,
                                                  sin_);
#else
  throw std::invalid_argument(
    "Error: No implementation of rotary embedding layer incremental_forwarding "
    "with SIMD acceleration except for NEON!");
#endif
}

void swiglu(const unsigned int N, _FP16 *X, _FP16 *Y, _FP16 *Z) {
#ifdef USE_NEON
  nntrainer::neon::swiglu(N, X, Y, Z);
#else
  unsigned int i = 0;
  while (i < N) {
    X[i] =
      (Y[i] / static_cast<_FP16>(1.f + std::exp(static_cast<float>(-Y[i])))) *
      Z[i];
    ++i;
  }
#endif
}

_FP16 max(const unsigned int N, _FP16 *X) {
#ifdef USE_NEON
  return nntrainer::neon::max(N, X);
#else
  std::vector<_FP16> v(X, X + N);
  return *std::max_element(v.begin(), v.end());
#endif
}

void softmax(const unsigned int N, _FP16 *X, _FP16 *Y) {
#ifdef USE_NEON
  nntrainer::neon::softmax(N, X, Y);
#else
  _FP16 max_x = max(N, X);
  unsigned int i = 0;
  float sum = 0.f;
  while (i < N) {
    sum += std::exp(static_cast<float>(X[i] - max_x));
    ++i;
  }
  i = 0;
  while (i < N) {
    Y[i] = static_cast<_FP16>(std::exp(static_cast<float>(X[i] - max_x)) / sum);
    ++i;
  }
#endif
}
#endif

} // namespace nntrainer
