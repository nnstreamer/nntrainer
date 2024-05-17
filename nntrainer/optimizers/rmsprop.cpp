// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Jijoong Moon <jijoong.moo@samsung.com>
 *
 * @file   rmsprop.cpp
 * @date   17 May 2024
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is the RMSProp optimizer.
 */

#include <cmath>
#include <fstream>

#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <rmsprop.h>
#include <util_func.h>

namespace nntrainer {

RMSProp::RMSProp() :
  rmsprop_props(props::Rho(), props::PropsEpsilon(), props::TorchRef()) {
  /** default properties */
  auto &[rho, eps, torch_ref] = rmsprop_props;
  rho.set(0.9f);
  eps.set(1.0e-7f);
  torch_ref.set(false);
}

RMSProp::~RMSProp() {}

enum RMSPropParams { wv };

std::vector<TensorDim> RMSProp::getOptimizerVariableDim(const TensorDim &dim) {
  /**
   * @note We assume the optimizer parameters should be full precsion to
   * maintain the accuracy even in mixed precision training.
   */
  TensorDim wv_dim(dim);
  wv_dim.setDataType(ml::train::TensorDim::DataType::FP32);
  return {wv_dim};
}

void RMSProp::exportTo(Exporter &exporter,
                       const ml::train::ExportMethods &method) const {
  exporter.saveResult(rmsprop_props, method, this);
  Optimizer::exportTo(exporter, method);
}

void RMSProp::setProperty(const std::vector<std::string> &values) {
  auto left = loadProperties(values, rmsprop_props);
  Optimizer::setProperty(left);
}

void RMSProp::applyGradient(RunOptimizerContext &context) {
  Tensor empty_tensor;

  Tensor &x_grad =
    context.getGradient().getDataType() == ml::train::TensorDim::DataType::FP32
      ? context.getGradient()
      : empty_tensor;

  if (x_grad.empty())
    x_grad = context.getGradient().clone(ml::train::TensorDim::DataType::FP32);

  auto &rho = std::get<props::Rho>(rmsprop_props).get();
  auto &epsilon = std::get<props::PropsEpsilon>(rmsprop_props).get();
  auto &torch_ref = std::get<props::TorchRef>(rmsprop_props).get();

  // This is implementation of rmpprop from original paper.
  // This is not deleted intentionally.
  unsigned int iteration = context.getIteration();

  Tensor &wv = context.getOptimizerVariable(RMSPropParams::wv);

  wv.multiply_i(rho);
  wv.add_i(x_grad.multiply(x_grad), 1.0f - rho);

  if (torch_ref) {
    Tensor denom = wv.apply<float>(sqrtFloat<float>);
    denom.add_i(epsilon);

    wv.divide(denom, x_grad);

    context.applyGradient(context.getLearningRate(), x_grad);

  } else {
    std::function<double(double)> sqrtEps = [epsilon](double f) {
      return 1 / (sqrtDouble(f) + epsilon);
    };

    x_grad = wv.apply<float>(sqrtEps, x_grad);
    context.applyGradient(context.getLearningRate(), x_grad);
  }
}

} // namespace nntrainer
