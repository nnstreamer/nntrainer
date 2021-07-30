// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   optimizer_context.h
 * @date   30 July 2021
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is the layer context for each layer
 */

#include <optimizer_context.h>
#include <weight.h>

namespace nntrainer {

/**
 * @brief Get the Weight tensor object
 */
Tensor &RunOptimizerContext::getWeight() const {
  return weight->getVariableRef();
}

/**
 * @brief Get the Weight Gradient tensor object
 */
Tensor &RunOptimizerContext::getGradient() const {
  return weight->getGradientRef();
}

/**
 * @brief Get the optimizer variable associated to this weight
 */
Tensor &RunOptimizerContext::getOptimizerVariable(unsigned int idx) const {
  return weight->getOptimizerVariableRef(idx);
}

/**
 * @brief   Apply the gradient with the given learning rate
 */
void RunOptimizerContext::applyGradient(double lr) const {
  weight->applyGradient(lr);
}
} // namespace nntrainer
