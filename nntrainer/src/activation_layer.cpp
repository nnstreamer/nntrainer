/**
 * Copyright (C) 2020 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * SPDX-License-Identifier: Apache-2.0-only
 *
 * @file   activation_layer.cpp
 * @date   17 June 2020
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is Activation Layer Class for Neural Network
 *
 */

#include <activation_layer.h>
#include <fstream>
#include <iostream>
#include <layer.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <optimizer.h>
#include <parse_util.h>
#include <tensor.h>
#include <util_func.h>
#include <vector>

namespace nntrainer {

/**
 * @brief     Initialize the layer
 *
 * @param[in] last last layer
 * @retval #ML_ERROR_NONE Successful.
 * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
 */
int ActivationLayer::initialize(bool last) {
  if (input_dim.batch() <= 0 || input_dim.height() <= 0 ||
      input_dim.width() <= 0 || input_dim.channel() <= 0) {
    ml_loge("Error: Dimension must be greater than 0");
    return ML_ERROR_INVALID_PARAMETER;
  }

  this->last_layer = last;

  dim = input_dim;
  output_dim = dim;

  return ML_ERROR_NONE;
}

Tensor ActivationLayer::forwarding(Tensor in, int &status) {
  status = ML_ERROR_NONE;

  input = in;
  hidden = _act_fn(in);

  return hidden;
}

Tensor ActivationLayer::backwarding(Tensor derivative, int iteration) {
  if (activation_type == ActiType::ACT_SOFTMAX)
    return derivative.multiply(_act_prime_fn(hidden));
  else
    return derivative.multiply(_act_prime_fn(input));
}

/**
 * @brief     copy layer
 * @param[in] l layer to copy
 */
void ActivationLayer::copy(std::shared_ptr<Layer> l) {
  std::shared_ptr<ActivationLayer> from =
    std::static_pointer_cast<ActivationLayer>(l);
  this->input.copy(from->input);
  this->hidden.copy(from->hidden);
  this->activation_type = from->activation_type;
};

int ActivationLayer::setActivation(
  std::function<Tensor(Tensor const &)> const &activation_fn,
  std::function<Tensor(Tensor const &)> const &activation_prime_fn) {
  _act_fn = activation_fn;
  _act_prime_fn = activation_prime_fn;

  return ML_ERROR_NONE;
}

int ActivationLayer::setActivation(
  std::function<float(float const)> const &activation_fn,
  std::function<float(float const)> const &activation_prime_fn) {
  _act_fn = [activation_fn](Tensor const &t) { return t.apply(activation_fn); };
  _act_prime_fn = [activation_prime_fn](Tensor const &t) {
    return t.apply(activation_prime_fn);
  };

  return ML_ERROR_NONE;
}

/**
 * @brief setActivation by preset actiType
 *
 * @param[in] ActiType actiType actiType to be set
 */
void ActivationLayer::setActivation(ActiType acti_type) {
  switch (acti_type) {
  case ActiType::ACT_TANH:
    this->setActivation(tanhFloat, tanhPrime);
    break;
  case ActiType::ACT_SIGMOID:
    this->setActivation(sigmoid, sigmoidePrime);
    break;
  case ActiType::ACT_SOFTMAX:
    this->setActivation(softmax, softmaxPrime);
    break;
  case ActiType::ACT_RELU:
    this->setActivation(relu, reluPrime);
    break;
  case ActiType::ACT_NONE:
    this->setActivation(no_op, no_op);
    break;
  case ActiType::ACT_UNKNOWN:
  default:
    throw std::runtime_error("Error: Not Supported Activation Type");
  }
  this->activation_type = acti_type;
}

/**
 * @brief     set Property of layer
 * @param[in] values values of property
 * @retval #ML_ERROR_NONE Successful.
 * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
 */
int ActivationLayer::setProperty(std::vector<std::string> values) {
  int status = ML_ERROR_NONE;

  if (values.size() != 1) {
    return ML_ERROR_INVALID_PARAMETER;
  }

  std::string key;
  std::string value;

  status = getKeyValue(values[0], key, value);
  NN_RETURN_STATUS();

  if (parseLayerProperty(key) != ACTIVATION_PROPERTY) {
    return ML_ERROR_INVALID_PARAMETER;
  }

  try {
    this->setActivation((ActiType)parseType(value, TOKEN_ACTI));
  } catch (const std::exception &ex) {
    ml_loge("Error: Not supported Data");
    return ML_ERROR_INVALID_PARAMETER;
  }
  return ML_ERROR_NONE;
}

}; // namespace nntrainer
