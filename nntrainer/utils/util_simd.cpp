// SPDX-License-Identifier: Apache-2.0
/**
 * @file	util_simd.cpp
 * @date	09 Jan 2024
 * @brief	This is a collection of simd util functions. Aim of this file is to
 * directly call SIMD implemented functions without Tensor.
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Sungsik Kong <ss.kong@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#include <algorithm>
#include <cmath>
#include <cpu_backend.h>
#include <util_simd.h>

namespace nntrainer {

void calc_trigonometric_vals_dup_util(unsigned int N_half, float *angle,
                                      float *cos_, float *sin_,
                                      unsigned int from,
                                      float attention_scaling) {
  calc_trigonometric_vals_dup(N_half, angle, cos_, sin_, from,
                              attention_scaling);
}

void swiglu_util(const unsigned int N, float *X, float *Y, float *Z) {
  swiglu(N, X, Y, Z);
}

float max_util(const unsigned int N, float *X) { return max_val(N, X); }

void softmax_util(const unsigned int N, float *X, float *Y) {
  softmax(N, X, Y);
}

} // namespace nntrainer
