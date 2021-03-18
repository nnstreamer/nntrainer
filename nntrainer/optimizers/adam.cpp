// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file	adam.cpp
 * @date	6 October 2020
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @author	Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug		No known bugs except for NYI items
 * @brief	This is the Adam optimizer.
 */

#include <cmath>
#include <fstream>

#include <adam.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <parse_util.h>
#include <util_func.h>

namespace nntrainer {

const std::string Adam::type = "adam";

enum AdamParams { wm, wv };

void Adam::addOptimizerVariable(std::vector<Weight> &weight_list) {
  for (auto &w : weight_list) {
    w.clearOptimizerVariables();

    // TODO: only trainable weights must be sent to optimizer
    if (!w.getTrainable())
      continue;

    w.addOptimizerVariable(w.getDim()); /** Add wm */
    w.addOptimizerVariable(w.getDim()); /** Add wv */
  }
}

double Adam::getLearningRate(size_t iteration) const {
  double ll = OptimizerImpl::getLearningRate(iteration);

  std::function<float(double)> biasCorrection = [&](float f) {
    return 1.0f - pow(f, iteration + 1);
  };

  ll *= sqrt(biasCorrection(beta2)) / biasCorrection(beta1);

  return ll;
}

void Adam::applyGradient(Weight &weight, double updated_lr, int iteration) {

  Tensor &x_grad = weight.getGradientRef();

  // This is implementation of adam from original paper.
  // This is not deleted intentionally.
  // float biasCorrection1 = 1 - pow(beta1, iteration + 1);
  // float biasCorrection2 = 1 - pow(beta2, iteration + 1);
  // Tensor &wm = weight.getOptimizerVariableRef(AdamParams::wm);
  // Tensor &wv = weight.getOptimizerVariableRef(AdamParams::wv);

  // wm.multiply_i(beta1);
  // wm.add_i(x_grad, 1.0f - beta1);

  // wv.multiply_i(beta2);
  // wv.add_i(x_grad.multiply(x_grad), 1.0f - beta2);

  // Tensor denom = wv.apply(sqrtFloat)
  //                  .divide(sqrtFloat(biasCorrection2))
  //                  .add(epsilon);
  // x.add_i(wm.divide(denom), -ll / biasCorrection1);

  std::function<double(double)> sqrtEps = [&](double f) {
    return 1 / (sqrtDouble(f) + this->epsilon);
  };

  Tensor &wm = weight.getOptimizerVariableRef(AdamParams::wm);
  Tensor &wv = weight.getOptimizerVariableRef(AdamParams::wv);

  wm.multiply_i(beta1);
  wm.add_i(x_grad, 1.0f - beta1);

  wv.multiply_i(beta2);
  wv.add_i(x_grad.multiply(x_grad), 1.0f - beta2);

  x_grad = wv.apply(sqrtEps, x_grad);
  x_grad.multiply_i(wm);
  weight.applyGradient(updated_lr);
}

void Adam::setProperty(const PropertyType type, const std::string &value) {
  int status = ML_ERROR_NONE;

  switch (type) {
  case PropertyType::beta1:
    status = setDouble(beta1, value);
    break;
  case PropertyType::beta2:
    status = setDouble(beta2, value);
    break;
  case PropertyType::epsilon:
    status = setDouble(epsilon, value);
    break;
  default:
    OptimizerImpl::setProperty(type, value);
    status = ML_ERROR_NONE;
    break;
  }

  throw_status(status);
}

} // namespace nntrainer
