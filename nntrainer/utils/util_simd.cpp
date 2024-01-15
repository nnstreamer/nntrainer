// SPDX-License-Identifier: Apache-2.0
/**
 * @file	util_simd.cpp
 * @date	09 Jan 2024
 * @brief	This is a collection of simd util functions
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Sungsik Kong <ss.kong@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#include <cmath>
#include <util_simd.h>
#ifdef USE_NEON
#include <util_simd_neon.h>
#endif

namespace nntrainer {

void calc_trigonometric_vals_dup(unsigned int N_half, float *angle, float *cos_,
                                 float *sin_, unsigned int alpha) {
#ifdef USE_NEON
  nntrainer::neon::calc_trigonometric_vals_dup_neon(N_half, angle, cos_, sin_,
                                                    alpha);
#else
  throw std::invalid_argument(
    "Error: No implementation of rotary embedding layer incremental_forwarding "
    "with SIMD acceleration except for NEON!");
#endif
}

void swish(const unsigned int N, float *X, float *Y, float *Z) {
#ifdef USE_NEON
  nntrainer::neon::swish_neon(N, X, Y, Z);
#else
  unsigned int i = 0;
  while (i < N) {
    X[i] = (Y[i] / (1.f + std::exp(-Y[i]))) * Z[i];
    ++i;
  }
#endif
}

#ifdef ENABLE_FP16

void compute_rotary_embedding_value(unsigned int dim, unsigned int half_,
                                    unsigned int w, _FP16 *in, _FP16 *out,
                                    float *cos_, float *sin_) {
#ifdef USE_NEON
  nntrainer::neon::compute_rotary_embedding_value_neon(dim, half_, w, in, out,
                                                       cos_, sin_);
#else
  throw std::invalid_argument(
    "Error: No implementation of rotary embedding layer incremental_forwarding "
    "with SIMD acceleration except for NEON!");
#endif
}

void swish(const unsigned int N, _FP16 *X, _FP16 *Y, _FP16 *Z) {
#ifdef USE_NEON
  nntrainer::neon::swish_neon(N, X, Y, Z);
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
#endif

} // namespace nntrainer
