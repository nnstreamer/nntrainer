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
#include <layer_internal.h>
#include <lazy_tensor.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <parse_util.h>
#include <util_func.h>

namespace nntrainer {

enum class BNParams { mu, var, gamma, beta };

/// @todo add multiple axis support
int BatchNormalizationLayer::initialize() {
  int status = ML_ERROR_NONE;

  output_dim = input_dim;
  TensorDim dim;

  /// @note this logic cannot tell channel is actually 1 or it is just not used.
  if (axis == -1)
    axis = input_dim.channel() > 1 ? 1 : 3;

  dim.setTensorDim(axis, input_dim.getTensorDim(axis));

  for (int i = 0; i < 4; ++i) {
    if (axis != i)
      axes_to_reduce.push_back(i);
  }

  setNumWeights(4);
  weightAt(0) =
    std::move(Weight(dim, initializers[static_cast<int>(BNParams::mu)], false,
                     "BN:moving_mean"));
  ///@todo shift var to std to save computation
  weightAt(1) =
    std::move(Weight(dim, initializers[static_cast<int>(BNParams::var)], false,
                     "BN:moving_variance"));
  weightAt(2) = std::move(Weight(
    dim, initializers[static_cast<int>(BNParams::gamma)], true, "BN:gamma"));
  weightAt(3) = std::move(Weight(
    dim, initializers[static_cast<int>(BNParams::beta)], true, "BN:beta"));

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
  case PropertyType::moving_mean_initializer:
    if (!value.empty()) {
      initializers[static_cast<int>(BNParams::mu)] =
        (WeightInitializer)parseType(value, TOKEN_WEIGHT_INIT);
    }
    break;
  case PropertyType::moving_variance_initializer:
    if (!value.empty()) {
      initializers[static_cast<int>(BNParams::var)] =
        (WeightInitializer)parseType(value, TOKEN_WEIGHT_INIT);
    }
    break;
  case PropertyType::beta_initializer:
    if (!value.empty()) {
      initializers[static_cast<int>(BNParams::beta)] =
        (WeightInitializer)parseType(value, TOKEN_WEIGHT_INIT);
    }
    break;
  case PropertyType::gamma_initializer:
    if (!value.empty()) {
      initializers[static_cast<int>(BNParams::gamma)] =
        (WeightInitializer)parseType(value, TOKEN_WEIGHT_INIT);
    }
    break;
  case PropertyType::momentum:
    if (!value.empty()) {
      status = setFloat(momentum, value);
      throw_status(status);
    }
    break;
  default:
    Layer::setProperty(type, value);
    break;
  }
}

sharedConstTensors BatchNormalizationLayer::forwarding(sharedConstTensors in) {
  Tensor &mu = weightAt(static_cast<int>(BNParams::mu)).getVariableRef();
  Tensor &var = weightAt(static_cast<int>(BNParams::var)).getVariableRef();
  Tensor &gamma = weightAt(static_cast<int>(BNParams::gamma)).getVariableRef();
  Tensor &beta = weightAt(static_cast<int>(BNParams::beta)).getVariableRef();

  input = *in[0];
  /// @todo change trainable #524
  if (trainable) {
    Tensor cmu = input.average(axes_to_reduce);
    deviation = input.subtract(cmu);

    cvar = deviation.pow(2.0f).average(axes_to_reduce);
    cvar.add_i(epsilon);

    mu.multiply_i(momentum);
    mu.add_i(cmu, 1 - momentum);
    var.multiply_i(momentum);
    var.add_i(cvar, 1 - momentum);

    invstd = cvar.pow(-0.5f);
    this->x_normalized = deviation.multiply(invstd);
    this->hidden = x_normalized.multiply(gamma);
    this->hidden.add_i(beta);
  } else {
    deviation = input.subtract(mu);
    this->x_normalized = deviation.divide(var.pow(0.5f));
    this->hidden = x_normalized.multiply(gamma);
    this->hidden.add(beta);
  }

  return {MAKE_SHARED_TENSOR(hidden)};
}

sharedConstTensors
BatchNormalizationLayer::backwarding(sharedConstTensors derivative,
                                     int iteration) {
  Tensor &gamma = weightAt(static_cast<int>(BNParams::gamma)).getVariableRef();
  Tensor &dgamma = weightAt(static_cast<int>(BNParams::gamma)).getGradientRef();
  Tensor &dbeta = weightAt(static_cast<int>(BNParams::beta)).getGradientRef();
  Tensor dx_normalized;

  Tensor deriv = *derivative[0];

  int N = 1;

  for (auto &axis : axes_to_reduce) {
    N *= input_dim.getTensorDim(axis);
  }

  dbeta = deriv.sum(axes_to_reduce);
  dgamma = deviation.multiply(invstd).multiply(deriv).sum(axes_to_reduce);

  Tensor dx_1 = gamma.multiply(invstd);
  Tensor dx_2 = deriv.multiply(N);
  dx_2.subtract_i(deriv.sum(axes_to_reduce));
  dx_2.subtract_i(deviation.divide(cvar).multiply(
    deviation.multiply(deriv).sum(axes_to_reduce)));

  Tensor dx = dx_2.multiply(dx_1);
  dx.divide_i(N);

  opt->apply_gradients(weight_list, num_weights, iteration);

  return {MAKE_SHARED_TENSOR(std::move(dx))};
}

void BatchNormalizationLayer::copy(std::shared_ptr<Layer> l) {
  Layer::copy(l);

  std::shared_ptr<BatchNormalizationLayer> from =
    std::static_pointer_cast<BatchNormalizationLayer>(l);
  this->cvar.copy(from->cvar);
}

} /* namespace nntrainer */
