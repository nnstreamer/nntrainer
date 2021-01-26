// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file	weight.cpp
 * @date	22 September 2020
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug		No known bugs except for NYI items
 * @brief	This is Weight Class for Neural Network
 *
 */

#include <util_func.h>
#include <weight.h>

namespace nntrainer {

Weight::Weight(const TensorDim &dim, const WeightInitializer init, bool train,
               bool alloc_now_, std::string name) :
  Var_Grad(dim, train, alloc_now_, name),
  initializer(init) {
  if (initializer == WeightInitializer::WEIGHT_UNKNOWN)
    throw std::invalid_argument("Weight initializer unknown");
}

void Weight::initializeWeight(const Tensor &weights_preallocated) {
  Var_Grad::initializeWeight(weights_preallocated);

  if (alloc_now)
    initializeWeight();
}

void Weight::initializeVariable() {
  Tensor &var_ref = getVariableRef();
  const TensorDim dim = var_ref.getDim();

  switch (initializer) {
  case WeightInitializer::WEIGHT_ZEROS:
    var_ref.setZero();
    break;
  case WeightInitializer::WEIGHT_ONES:
    var_ref.setValue(1.0f);
    break;
  case WeightInitializer::WEIGHT_LECUN_NORMAL:
    var_ref.setRandNormal(0.0f, sqrtFloat(1.0f / dim.height()));
    break;
  case WeightInitializer::WEIGHT_XAVIER_NORMAL:
    var_ref.setRandNormal(0.0f, sqrtFloat(2.0f / (dim.width() + dim.height())));
    break;
  case WeightInitializer::WEIGHT_HE_NORMAL:
    var_ref.setRandNormal(0.0f, sqrtFloat(2.0f / (dim.height())));
    break;
  case WeightInitializer::WEIGHT_LECUN_UNIFORM:
    var_ref.setRandUniform(-1.0f * sqrtFloat(1.0f / dim.height()),
                           sqrtFloat(1.0f / dim.height()));
    break;
  case WeightInitializer::WEIGHT_XAVIER_UNIFORM:
    var_ref.setRandUniform(-1.0f *
                             sqrtFloat(6.0f / (dim.height() + dim.width())),
                           sqrtFloat(6.0 / (dim.height() + dim.width())));
    break;
  case WeightInitializer::WEIGHT_HE_UNIFORM:
    var_ref.setRandUniform(-1.0f * sqrtFloat(6.0f / (dim.height())),
                           sqrtFloat(6.0 / (dim.height())));
    break;
  default:
    break;
  }
}

void Weight::initializeGrad(const Tensor &preallocated, bool gtrain) {
  // Use self variable to initialize itself
  Var_Grad::initializeGrad(preallocated, gtrain);
  if (gtrain && alloc_now)
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
