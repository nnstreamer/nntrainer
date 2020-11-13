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

int Adam::initialize(std::shared_ptr<Weight> weight_list,
                     unsigned int num_weights, bool set_tensor) {
  int status = ML_ERROR_NONE;
  weight_mv.clear();

  if (set_tensor) {
    for (unsigned int i = 0; i < num_weights; ++i) {
      Weight &w = weight_list.get()[i];

      // TODO: only trainable weights must be sent to optimizer
      if (!w.getTrainable())
        continue;

      Tensor m = Tensor(w.getDim());
      m.setZero();
      Tensor v = Tensor(w.getDim());
      v.setZero();
      std::pair<Tensor, Tensor> p =
        std::pair<Tensor, Tensor>(std::move(m), std::move(v));
      weight_mv.push_back(std::move(p));
    }
  }
  return status;
}

double Adam::getLearningRate(int iteration) {
  double ll = Optimizer::getLearningRate(iteration);

  std::function<float(double)> biasCorrection = [&](float f) {
    return 1.0f - pow(f, iteration + 1);
  };

  ll *= sqrt(biasCorrection(beta2)) / biasCorrection(beta1);

  return ll;
}

void Adam::apply_gradient(Weight &weight, int tensor_idx, double updated_lr,
                          int iteration) {

  Tensor &x = weight.getVariableRef();
  const Tensor &x_grad = weight.getGradientRef();

  // This is implementation of adam from original paper.
  // This is not deleted intentionally.
  // float biasCorrection1 = 1 - pow(beta1, iteration + 1);
  // float biasCorrection2 = 1 - pow(beta2, iteration + 1);
  // Tensor &wm = weight_mv[idx].first;
  // Tensor &wv = weight_mv[idx].second;

  // wm.multiply_i(beta1);
  // wm.add_i(x_grad, 1.0f - beta1);

  // wv.multiply_i(beta2);
  // wv.add_i(x_grad.multiply(x_grad), 1.0f - beta2);

  // Tensor denom = wv.apply(sqrtFloat)
  //                  .divide(sqrtFloat(biasCorrection2))
  //                  .add(epsilon);
  // x.add_i(wm.divide(denom), -ll / biasCorrection1);

  std::function<double(double)> sqrtEps = [&](double f) {
    return sqrtDouble(f) + this->epsilon;
  };

  Tensor &wm = weight_mv[tensor_idx].first;
  Tensor &wv = weight_mv[tensor_idx].second;

  wm.multiply_i(beta1);
  wm.add_i(x_grad, 1.0f - beta1);

  wv.multiply_i(beta2);
  wv.add_i(x_grad.multiply(x_grad), 1.0f - beta2);

  // TODO: combine this operation to reduce from two temp allocations to one
  Tensor divider;
  divider = wv.apply(sqrtEps, divider);
  x.add_i(wm.divide(divider), -updated_lr);
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
    Optimizer::setProperty(type, value);
    status = ML_ERROR_NONE;
    break;
  }

  throw_status(status);
}

void Adam::read(std::ifstream &file) {
  /// @todo need strong exception safety guarantee
  Optimizer::read(file);

  if (continue_train) {
    for (auto iter = weight_mv.begin(); iter != weight_mv.end(); iter++) {
      (*iter).first.read(file);
      (*iter).second.read(file);
    }
  } else {
    size_t total_size = 0;
    for (auto iter = weight_mv.begin(); iter != weight_mv.end(); iter++)
      total_size += (*iter).first.getSize() + (*iter).second.getSize();

    file.seekg(total_size, std::ifstream::cur);
  }
}

void Adam::save(std::ofstream &file) {
  Optimizer::save(file);

  for (auto iter = weight_mv.begin(); iter != weight_mv.end(); iter++) {
    (*iter).first.save(file);
    (*iter).second.save(file);
  }
}

} // namespace nntrainer
