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
               const WeightRegularizer reg, const float reg_const, bool train,
               bool alloc_now_, std::string name) :
  Var_Grad(dim, init, train, alloc_now_, name),
  regularizer(reg),
  regularizer_constant(reg_const) {
  if (init == Tensor::Initializer::NONE)
    throw std::invalid_argument("Weight initializer cannot be none");
  if (regularizer == WeightRegularizer::UNKNOWN)
    throw std::invalid_argument("Weight regularizer unknown");
}

void Weight::initializeGradient(const Tensor &preallocated) {
  // Use self variable to initialize itself
  Var_Grad::initializeGradient(preallocated);
  if (alloc_now)
    allocateOptimizerVariables();
}

void Weight::allocateOptimizerVariables() {
  // Allocate optimizer parameters
  for (auto const &dim : opt_vars_dim) {
    opt_vars.emplace_back(dim);
    opt_vars.back().setZero();
  }
}

} // namespace nntrainer
