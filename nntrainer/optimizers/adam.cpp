// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   adam.cpp
 * @date   6 October 2020
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is the Adam optimizer.
 */

#include <cmath>
#include <fstream>

#include <adam.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <util_func.h>

namespace nntrainer {

Adam::Adam() : adam_props(PropsB1(), PropsB2(), PropsEpsilon(), TorchRef()) {
  /** default properties */
  auto &[b1, b2, eps, torch_ref] = adam_props;
  b1.set(0.9f);
  b2.set(0.999f);
  eps.set(1.0e-7f);
  torch_ref.set(false);
}

Adam::~Adam() {}

enum AdamParams { wm, wv };

std::vector<TensorDim> Adam::getOptimizerVariableDim(const TensorDim &dim) {
  return {dim, dim};
}

void Adam::exportTo(Exporter &exporter,
                    const ml::train::ExportMethods &method) const {
  exporter.saveResult(adam_props, method, this);
  Optimizer::exportTo(exporter, method);
}

void Adam::setProperty(const std::vector<std::string> &values) {
  auto left = loadProperties(values, adam_props);
  Optimizer::setProperty(left);
}

double Adam::getUpdatedLearningRate(unsigned int iteration, double ll) const {
  auto &beta1 = std::get<PropsB1>(adam_props).get();
  auto &beta2 = std::get<PropsB2>(adam_props).get();

  std::function<float(double)> biasCorrection = [&](float f) {
    return 1.0f - pow(f, iteration + 1);
  };

  ll *= sqrt(biasCorrection(beta2)) / biasCorrection(beta1);

  return ll;
}

void Adam::applyGradient(RunOptimizerContext &context) {
  Tensor &x_grad = context.getGradient();

  auto &beta1 = std::get<PropsB1>(adam_props).get();
  auto &beta2 = std::get<PropsB2>(adam_props).get();
  auto &epsilon = std::get<PropsEpsilon>(adam_props).get();
  auto &torch_ref = std::get<TorchRef>(adam_props).get();

  // This is implementation of adam from original paper.
  // This is not deleted intentionally.
  unsigned int iteration = context.getIteration();
  float biasCorrection1 = 1 - pow(beta1, iteration + 1);
  float biasCorrection2 = 1 - pow(beta2, iteration + 1);
  Tensor &wm = context.getOptimizerVariable(AdamParams::wm);
  Tensor &wv = context.getOptimizerVariable(AdamParams::wv);

  wm.multiply_i(beta1);
  wm.add_i(x_grad, 1.0f - beta1);

  wv.multiply_i(beta2);
  wv.add_i(x_grad.multiply(x_grad), 1.0f - beta2);

  if (torch_ref) {
    Tensor denom = wv.apply<float>(sqrtFloat);
    denom.divide_i(sqrtFloat(biasCorrection2));
    denom.add_i(epsilon);
    wm.divide(denom, x_grad);

    context.applyGradient(context.getLearningRate() / biasCorrection1);

  } else {
    std::function<double(double)> sqrtEps = [epsilon](double f) {
      return 1 / (sqrtDouble(f) + epsilon);
    };

    x_grad = wv.apply<float>(sqrtEps, x_grad);
    x_grad.multiply_i(wm);
    context.applyGradient(getUpdatedLearningRate(context.getIteration(),
                                                 context.getLearningRate()));
  }
}

} // namespace nntrainer
