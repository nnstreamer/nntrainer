// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file	lstm.cpp
 * @date	17 March 2021
 * @brief	This is Long Short-Term Memory Layer Class of Neural Network
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#include <cmath>
#include <layer_internal.h>
#include <lazy_tensor.h>
#include <lstm.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <parse_util.h>
#include <util_func.h>

namespace nntrainer {

const std::string LSTMLayer::type = "lstm";

enum LSTMParams { weight_xh, weight_hh, bias_h };

int LSTMLayer::initialize(Manager &manager) {
  int status = ML_ERROR_NONE;
  if (getNumInputs() != 1) {
    throw std::invalid_argument("LSTM layer takes only one input");
  }

  return status;
}

void LSTMLayer::setProperty(const PropertyType type, const std::string &value) {
  int status = ML_ERROR_NONE;
  // TODO : Add return_state property & api to get the hidden input
  switch (type) {
  case PropertyType::unit: {
    if (!value.empty()) {
      status = setUint(unit, value);
      throw_status(status);
      output_dim[0].width(unit);
    }
    break;
  case PropertyType::activation:
    if (!value.empty()) {
      ActivationType acti_type = (ActivationType)parseType(value, TOKEN_ACTI);
      Layer::activation_type = acti_type;
      acti_func.setActiFunc(acti_type);
    }
    break;
  case PropertyType::recurrent_activation:
    if (!value.empty()) {
      ActivationType acti_type = (ActivationType)parseType(value, TOKEN_ACTI);
      recurrent_activation_type = acti_type;
      recurrent_acti_func.setActiFunc(acti_type);
    }
    break;
  default:
    Layer::setProperty(type, value);
    break;
  }
  }
}

void LSTMLayer::setRecurrentActivation(ActivationType activation) {
  if (activation == ActivationType::ACT_UNKNOWN) {
    throw std::invalid_argument("Error: have to specify activation function");
  }
  recurrent_activation_type = activation;
}

void LSTMLayer::forwarding(bool training) {
  // NYI
}

void LSTMLayer::copy(std::shared_ptr<Layer> l) {
  Layer::copy(l);

  std::shared_ptr<LSTMLayer> from = std::static_pointer_cast<LSTMLayer>(l);
  this->unit = from->unit;
  this->acti_func = from->acti_func;
  this->recurrent_activation_type = from->recurrent_activation_type;
  this->recurrent_acti_func = from->recurrent_acti_func;
}

void LSTMLayer::calcDerivative() {
  // NYI
}

void LSTMLayer::calcGradient() {
  // NYI
}

} // namespace nntrainer
