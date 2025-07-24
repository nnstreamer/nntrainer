// SPDX-License-Identifier: Apache-2.0
/**
 * @file	util_simd_fp16.cpp
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

void compute_rotary_embedding_value_util(unsigned int dim, unsigned int half_,
                                         unsigned int w, _FP16 *in, _FP16 *out,
                                         float *cos_, float *sin_) {
  compute_rotary_embedding_value(dim, half_, w, in, out, cos_, sin_);
}

void swiglu_util(const unsigned int N, _FP16 *X, _FP16 *Y, _FP16 *Z) {
  swiglu(N, X, Y, Z);
}

_FP16 max_util(const unsigned int N, _FP16 *X) { return max_val(N, X); }

void softmax_util(const unsigned int N, _FP16 *X, _FP16 *Y) {
  softmax(N, X, Y);
}

} // namespace nntrainer
