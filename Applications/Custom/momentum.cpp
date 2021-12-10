// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   momentum.h
 * @date   1 June 2021
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is the Momentum optimizer.
 */
#include <momentum.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <tensor.h>

namespace custom {

Momentum::Momentum() { setProperty({"learning_rate=0.001", "momentum=0.9f"}); }

void Momentum::applyGradient(nntrainer::RunOptimizerContext &context) {
  nntrainer::Tensor &x_grad = context.getGradient();
  nntrainer::Tensor &accumulated = context.getOptimizerVariable(0);

  float momentum = std::get<PropMomentum>(momentum_props);

  accumulated.add_i(x_grad, momentum);

  x_grad.fill(accumulated);
  context.applyGradient(getLearningRate(context.getIteration()));
}

double Momentum::getLearningRate(size_t iteration) const {
  return nntrainer::OptimizerImpl::getLearningRate(iteration);
}

std::vector<ml::train::TensorDim>
Momentum::getOptimizerVariableDim(const ml::train::TensorDim &dim) {
  /// momentum optimizer uses *accumulated
  return {dim};
}

void Momentum::setProperty(const std::vector<std::string> &values) {
  auto left = loadProperties(values, momentum_props);
  OptimizerImpl::setProperty(left);
}

/// if a custom optimizer is compiled as a separate so, this is where you need
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

} // namespace custom
