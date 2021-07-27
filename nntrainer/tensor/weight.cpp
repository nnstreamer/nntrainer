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
  Var_Grad(dim, train, alloc_now_, name),
  initializer(init),
  regularizer(reg),
  regularizer_constant(reg_const) {
  if (initializer == Tensor::Initializer::NONE)
    throw std::invalid_argument("Weight initializer unknown");
  if (regularizer == WeightRegularizer::UNKNOWN)
    throw std::invalid_argument("Weight regularizer unknown");
}

void Weight::initializeVariable(const Tensor &preallocated) {
  Var_Grad::initializeVariable(preallocated);

  if (alloc_now)
    runVariableInitializer();
}

void Weight::runVariableInitializer() {
  Tensor &var_ref = getVariableRef();
  const TensorDim dim = var_ref.getDim();

  unsigned int fan_in, fan_out;

  /// @fixme: when unit is equal to one, this does not work, we need to rely on
  /// effective dimension then actual numbers here. For now, some heuristics
  /// added to infer what would be fan_in/fan_out
  if (dim.batch() * dim.channel() * dim.height() == 1) {
    fan_out = fan_in = dim.width();
  } else if (dim.batch() * dim.channel() == 1) { /// fully connected layers
    fan_in = dim.height();
    fan_out = dim.width();
  } else { /// convolution filters, @todo extend this to > 4
    auto field_size = dim.height() * dim.width();

    // this also handles below cases.
    // 1. fan_in = fan_out = 1 as well.
    // 2. batch == 1, channel == 1 and height == 1, theoretical rank of 1
    fan_in = dim.channel() * field_size;
    fan_out = dim.batch() * field_size;
  }

  switch (initializer) {
  case Tensor::Initializer::ZEROS:
    var_ref.setZero();
    break;
  case Tensor::Initializer::ONES:
    var_ref.setValue(1.0f);
    break;
  case Tensor::Initializer::LECUN_NORMAL:
    var_ref.setRandNormal(0.0f, sqrtFloat(1.0f / fan_in));
    break;
  case Tensor::Initializer::XAVIER_NORMAL:
    var_ref.setRandNormal(0.0f, sqrtFloat(2.0f / (fan_in + fan_out)));
    break;
  case Tensor::Initializer::HE_NORMAL:
    var_ref.setRandNormal(0.0f, sqrtFloat(2.0f / (fan_in)));
    break;
  case Tensor::Initializer::LECUN_UNIFORM:
    var_ref.setRandUniform(-1.0f * sqrtFloat(1.0f / fan_in),
                           sqrtFloat(1.0f / fan_in));
    break;
  case Tensor::Initializer::XAVIER_UNIFORM:
    var_ref.setRandUniform(-1.0f * sqrtFloat(6.0f / (fan_in + fan_out)),
                           sqrtFloat(6.0 / (fan_in + fan_out)));
    break;
  case Tensor::Initializer::HE_UNIFORM:
    var_ref.setRandUniform(-1.0f * sqrtFloat(6.0f / (fan_in)),
                           sqrtFloat(6.0 / (fan_in)));
    break;
  default:
    break;
  }
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
