/**
 * Copyright (C) 2020 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *   http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 *
 * @file	bn_layer.cpp
 * @date	14 May 2020
 * @brief	This is Batch Normalization Layer Class for Neural Network
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#include <assert.h>
#include <bn_layer.h>
#include <layer.h>
#include <lazy_tensor.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <parse_util.h>
#include <util_func.h>

namespace nntrainer {

enum class BNParams { mu, var, gamma, beta };

/// @todo add channel wise bn for convolutional layer.
int BatchNormalizationLayer::initialize() {
  int status = ML_ERROR_NONE;

  output_dim = input_dim;
  TensorDim dim = input_dim;
  dim.batch(1);

  Tensor mu = Tensor(dim);
  Tensor var = Tensor(dim);
  Tensor gamma = Tensor(dim);
  Tensor beta = Tensor(dim);

  mu.setZero();
  var.setValue(1);
  gamma.setZero();
  beta.setZero();

  setParamSize(4);
  paramsAt(0) = {std::move(mu), Tensor(), "BN:moving_average"};
  paramsAt(1) = {std::move(var), Tensor(), "BN:moving_variance"};
  paramsAt(2) = {std::move(gamma), Tensor(gamma.getDim()), "BN:gamma"};
  paramsAt(3) = {std::move(beta), Tensor(beta.getDim()), "BN:beta"};

  return status;
}

void BatchNormalizationLayer::setProperty(const PropertyType type,
                                          const std::string &value) {
  int status = ML_ERROR_NONE;
  switch (type) {
  case PropertyType::epsilon:
    if (!value.empty()) {
      status = setFloat(epsilon, value);
      throw_status(status);
    }
    break;
  default:
    Layer::setProperty(type, value);
    break;
  }
}

sharedConstTensor BatchNormalizationLayer::forwarding(sharedConstTensor in) {
  Tensor &mu = paramsAt(static_cast<int>(BNParams::mu)).weight;
  Tensor &var = paramsAt(static_cast<int>(BNParams::var)).weight;
  Tensor &gamma = paramsAt(static_cast<int>(BNParams::gamma)).weight;
  Tensor &beta = paramsAt(static_cast<int>(BNParams::beta)).weight;

  if (trainable) {
    Tensor deviation;
    input = *in;

    ///< current mu */
    Tensor cmu;

    cmu = input.average(0);

    deviation = input.subtract(cmu);

    this->cvar = deviation.chain()
                   .multiply_i(deviation)
                   .sum(0)
                   .multiply_i(1.0f / input_dim.batch())
                   .add_i(epsilon)
                   .run();

    /// @todo replace momentum paramter
    float momentum = 0.9;
    mu.multiply_i(momentum);
    mu.add_i(cmu, 1 - momentum);
    var.multiply_i(momentum);
    var.add_i(cvar, 1 - momentum);

    this->x_normalized = deviation.divide(cvar.apply(sqrtFloat));

    this->hidden = x_normalized.chain().multiply_i(gamma).add_i(beta).run();
  } else {
    /// NYI
    throw std::runtime_error("not_yet_implemented");
  }

  return MAKE_SHARED_TENSOR(hidden);
}

sharedConstTensor
BatchNormalizationLayer::backwarding(sharedConstTensor derivative,
                                     int iteration) {
  Tensor &gamma = paramsAt(static_cast<int>(BNParams::gamma)).weight;
  Tensor &dbeta = paramsAt(static_cast<int>(BNParams::beta)).grad;
  Tensor &dgamma = paramsAt(static_cast<int>(BNParams::beta)).grad;
  Tensor dx_normalized;

  Tensor dx;
  Tensor deriv = *derivative;

  int batch = input_dim.batch();

  dgamma = x_normalized.multiply(deriv).sum(0);
  dbeta = deriv.sum(0);

  dx_normalized = deriv.multiply(gamma);

  dx = dx_normalized.chain()
         .multiply_i(batch)
         .subtract_i(dx_normalized.sum(0))
         .subtract_i(
           x_normalized.multiply(dx_normalized.multiply(x_normalized).sum(0)))
         .divide_i(cvar.multiply(batch))
         .run();

  std::shared_ptr<UpdatableParam> grad_params(params, params.get() + 2);

  opt.apply_gradients(grad_params, param_size - 2, iteration);

  return MAKE_SHARED_TENSOR(std::move(dx));
}

void BatchNormalizationLayer::copy(std::shared_ptr<Layer> l) {
  Layer::copy(l);

  std::shared_ptr<BatchNormalizationLayer> from =
    std::static_pointer_cast<BatchNormalizationLayer>(l);
  this->opt = from->opt;
  this->input_dim = from->input_dim;
  this->output_dim = from->output_dim;
  this->input.copy(from->input);
  this->hidden.copy(from->hidden);
  this->cvar.copy(from->cvar);
}

} /* namespace nntrainer */
