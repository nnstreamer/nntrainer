// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   activation_layer.cpp
 * @date   17 June 2020
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is Activation Layer Class for Neural Network
 *
 */

#include <algorithm>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <vector>

#include <activation_layer.h>
#include <blas_interface.h>
#include <layer_internal.h>
#include <lazy_tensor.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <parse_util.h>
#include <tensor.h>
#include <util_func.h>

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

void ActivationLayer::finalize(InitLayerContext &context) {
  context.setOutputDimensions(context.getInputDimensions());
}

void ActivationLayer::forwarding(RunLayerContext &context, bool training) {
  Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);
  Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);
  acti_func.run_fn(input_, hidden_);
}

void ActivationLayer::calcDerivative(RunLayerContext &context) {
  Tensor &deriv = context.getIncomingDerivative(SINGLE_INOUT_IDX);
  Tensor &ret = context.getOutgoingDerivative(SINGLE_INOUT_IDX);
  Tensor &out = context.getOutput(SINGLE_INOUT_IDX);

  acti_func.run_prime_fn(out, ret, deriv);
}

void ActivationLayer::setProperty(const std::vector<std::string> &values) {
  /// @todo: deprecate this in favor of loadProperties
  for (unsigned int i = 0; i < values.size(); ++i) {
    std::string key;
    std::string value;
    std::stringstream ss;

    if (getKeyValue(values[i], key, value) != ML_ERROR_NONE) {
      throw std::invalid_argument("Error parsing the property: " + values[i]);
    }

    if (value.empty()) {
      ss << "value is empty: key: " << key << ", value: " << value;
      throw std::invalid_argument(ss.str());
    }

    /// @note this calls derived setProperty if available
    setProperty(key, value);
  }
}

void ActivationLayer::setProperty(const std::string &type_str,
                                  const std::string &value) {
  using PropertyType = LayerV1::PropertyType;
  LayerV1::PropertyType type =
    static_cast<LayerV1::PropertyType>(parseLayerProperty(type_str));

  switch (type) {
  case PropertyType::activation: {
    acti_func.setActiFunc((ActivationType)parseType(value, TOKEN_ACTI));
  } break;
  default:
    std::string msg =
      "[Layer] Unknown Layer Property Key for value " + std::string(value);
    throw exception::not_supported(msg);
  }
}

int ActivationLayer::setActivation(
  std::function<Tensor &(Tensor const &, Tensor &)> const &activation_fn,
  std::function<Tensor &(Tensor &, Tensor &, Tensor const &)> const
    &activation_prime_fn) {
  acti_func.setActivation(activation_fn, activation_prime_fn);

  return ML_ERROR_NONE;
}

int ActivationLayer::setActivation(
  std::function<Tensor &(Tensor const &, Tensor &)> const &activation_fn,
  std::function<Tensor &(Tensor &, Tensor &)> const &activation_prime_fn) {

  acti_func.setActivation(activation_fn, activation_prime_fn);

  return ML_ERROR_NONE;
}

int ActivationLayer::setActivation(
  std::function<float(float const)> const &activation_fn,
  std::function<float(float const)> const &activation_prime_fn) {

  acti_func.setActivation(activation_fn, activation_prime_fn);

  return ML_ERROR_NONE;
}

}; // namespace nntrainer
