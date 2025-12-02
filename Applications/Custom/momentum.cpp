// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   momentum.cpp
 * @date   31 March 2023
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is the Momentum optimizer.
 */

#include <cmath>
#include <fstream>

#include <momentum.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>

namespace custom {

Momentum::Momentum() : momentum_props(PropsM()) {
  /** default properties */
  auto &[m] = momentum_props;
  m.set(0.9f);
}

enum MomentumParams { wm };

std::vector<ml::train::TensorDim>
Momentum::getOptimizerVariableDim(const ml::train::TensorDim &dim) {
  return {dim};
}

void Momentum::exportTo(nntrainer::Exporter &exporter,
                        const ml::train::ExportMethods &method) const {
  exporter.saveResult(momentum_props, method, this);
  Optimizer::exportTo(exporter, method);
}

void Momentum::applyGradient(nntrainer::RunOptimizerContext &context) {

  auto &m = std::get<PropsM>(momentum_props).get();

  nntrainer::Tensor &x_grad = context.getGradient();
  nntrainer::Tensor &accumulated =
    context.getOptimizerVariable(MomentumParams::wm);

  accumulated.multiply_i(m);
  accumulated.add_i(x_grad);

  x_grad.fill(accumulated);
  context.applyGradient(context.getLearningRate());
}

void Momentum::setProperty(const std::vector<std::string> &values) {
  auto left = loadProperties(values, momentum_props);
  Optimizer::setProperty(left);
}

#ifdef PLUGGABLE

nntrainer::Optimizer *create_momentum_optimizer() {
  auto optimizer = new Momentum();
  return optimizer;
}

void destory_momentum_optimizer(nntrainer::Optimizer *optimizer) {
  delete optimizer;
}

extern "C" {
nntrainer::OptimizerPluggable ml_train_optimizer_pluggable{
  create_momentum_optimizer, destory_momentum_optimizer};
}

#endif

} // namespace custom
