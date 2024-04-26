// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   weight.cpp
 * @date   22 September 2020
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is Weight Class for Neural Network
 *
 */

#include <util_func.h>
#include <weight.h>

#include <nntrainer_error.h>

namespace nntrainer {

Weight::Weight(const TensorDim &dim, const Tensor::Initializer init,
               const WeightRegularizer reg, const float reg_const,
               const float decay_const, const float max_norm, bool train,
               bool alloc_now_, std::string name, unsigned int axis,
               float loss_scale_) :
  Var_Grad(dim, init, train, alloc_now_, name),
  regularizer(reg),
  regularizer_constant(reg_const),
  decay(decay_const),
  clip_by_global_norm(max_norm),
  output_axis(axis),
  loss_scale(loss_scale_) {
  if (init == Tensor::Initializer::NONE)
    throw std::invalid_argument("Weight initializer cannot be none");
  if (regularizer == WeightRegularizer::UNKNOWN)
    throw std::invalid_argument("Weight regularizer unknown");
}

Weight::Weight(const TensorDim &dim_v, const TensorDim &dim_g,
               const Tensor::Initializer init, const WeightRegularizer reg,
               const float reg_const, const float decay_const,
               const float max_norm, bool train, bool alloc_now_,
               std::string name, unsigned int axis, float loss_scale_) :
  Var_Grad(dim_v, dim_g, init, train, alloc_now_, name),
  regularizer(reg),
  regularizer_constant(reg_const),
  decay(decay_const),
  clip_by_global_norm(max_norm),
  output_axis(axis),
  loss_scale(loss_scale_) {
  if (init == Tensor::Initializer::NONE)
    throw std::invalid_argument("Weight initializer cannot be none");
  if (regularizer == WeightRegularizer::UNKNOWN)
    throw std::invalid_argument("Weight regularizer unknown");
}

} // namespace nntrainer
