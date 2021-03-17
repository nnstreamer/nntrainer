// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file	rnn.cpp
 * @date	17 March 2021
 * @brief	This is Recurrent Layer Class of Neural Network
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#include <layer_internal.h>
#include <lazy_tensor.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <parse_util.h>
#include <rnn.h>
#include <util_func.h>

namespace nntrainer {

const std::string RNNLayer::type = "rnn";

enum RNNParams { weight_xh, weight_hh, bias_h };

int RNNLayer::initialize(Manager &manager) {
  int status = ML_ERROR_NONE;
  if (getNumInputs() != 1) {
    throw std::invalid_argument("RNN layer takes only one input");
  }

  // NYI

  return status;
}

void RNNLayer::setProperty(const PropertyType type, const std::string &value) {
  int status = ML_ERROR_NONE;
  switch (type) {
  case PropertyType::unit: {
    if (!value.empty()) {
      status = setUint(unit, value);
      throw_status(status);
      output_dim[0].width(unit);
    }
    break;
  default:
    Layer::setProperty(type, value);
    break;
  }
  }
}

void RNNLayer::forwarding(bool training) {
  // NYI
}

void RNNLayer::copy(std::shared_ptr<Layer> l) {
  Layer::copy(l);

  std::shared_ptr<RNNLayer> from = std::static_pointer_cast<RNNLayer>(l);
  this->unit = from->unit;
}

void RNNLayer::calcDerivative() {
  //  NYI
}

void RNNLayer::calcGradient() {
  // NYI
}

} // namespace nntrainer
