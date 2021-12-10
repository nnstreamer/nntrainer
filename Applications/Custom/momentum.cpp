// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   momentum.h
 * @date   1 June 2021
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is the Momentum optimizer.
 */

#include <cmath>
#include <fstream>

#include <momentum.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>

namespace nntrainer {

const std::string Momentum::type = "momentum";

void Momentum::addOptimizerVariable(std::vector<Weight> &weight_list) {
  for (auto &w : weight_list) {
    w.clearOptimizerVariables();

    if (!w.getTrainable())
      continue;

    w.addOptimizerVariable(w.getDim()); /** Add accumulated_momentum */
  }
}

void Momentum::applyGradient(Weight &weight, double updated_lr, int iteration) {

  Tensor &x_grad = weight.getGradientRef();
  Tensor &accumulated = weight.getOptimizerVariableRef(0);

  accumulated.multiply_i(momentum);
  accumulated.add_i(x_grad);

  x_grad.fill(accumulated);
  weight.applyGradient(updated_lr);
}

void Momentum::setProperty(const std::string &key, const std::string &value) {
  int status = ML_ERROR_NONE;
  if (key == "momentum") {
    status = setDouble(momentum, value);
  } else {
    OptimizerImpl::setProperty(key, value);
  }

  throw_status(status);
}

#ifdef PLUGGABLE

ml::train::Optimizer *create_momentum_optimizer() {
  auto optimizer = new Momentum();
  return optimizer;
}

void destory_momentum_optimizer(ml::train::Optimizer *optimizer) {
  delete optimizer;
}

extern "C" {
nntrainer::OptimizerPluggable ml_train_optimizer_pluggable{
  create_momentum_optimizer, destory_momentum_optimizer};
}

#endif

} // namespace nntrainer
