// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Daniel Jang <minhyukjang@snu.ac.kr>
 *
 * @file   adamw.cpp
 * @date   3 November 2024
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @author Daniel Jang <minhyukjang@snu.ac.kr>
 * @bug    No known bugs except for NYI items
 * @brief  This is the AdamW Optimizer.
 */

#include <cmath>
#include <fstream>

#include <adamw.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <util_func.h>

namespace nntrainer {

AdamW::AdamW() : adam_props(PropsB1(), PropsB2(), PropsEpsilon(), TorchRef()) {
  /** default properties */
  auto &[b1, b2, eps, torch_ref] = adam_props;
  b1.set(0.9f);
  b2.set(0.999f);
  eps.set(1.0e-7f);
  torch_ref.set(false);
}

AdamW::~AdamW() {}

enum AdamParams { wm, wv };

std::vector<TensorDim> AdamW::getOptimizerVariableDim(const TensorDim &dim) {
  return {dim, dim};
}

void AdamW::exportTo(Exporter &exporter,
                     const ml::train::ExportMethods &method) const {
  exporter.saveResult(adam_props, method, this);
  Optimizer::exportTo(exporter, method);
}

void AdamW::setProperty(const std::vector<std::string> &values) {
  auto left = loadProperties(values, adam_props);
  Optimizer::setProperty(left);
}

void AdamW::applyGradient(RunOptimizerContext &context) {
  Tensor empty_tensor;

  Tensor &x_grad =
    context.getGradient().getDataType() == ml::train::TensorDim::DataType::FP32
      ? context.getGradient()
      : empty_tensor;

  if (x_grad.empty()) {
    x_grad = context.getGradient().clone();
  }

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

  wv.divide_i(sqrtFloat(biasCorrection2));
  std::function<double(double)> sqrtEps = [epsilon](double f) {
    return 1 / (sqrtDouble(f) + epsilon);
  };
  wv.apply<float>(sqrtEps);
  x_grad = wv;
  x_grad.divide_i(biasCorrection1);
  x_grad.multiply_i(wm);

  context.applyGradient(context.getLearningRate());
}

} // namespace nntrainer
