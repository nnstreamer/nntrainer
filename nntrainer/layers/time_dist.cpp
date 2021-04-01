// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file	time_dist.cpp
 * @date	01 April 2021
 * @brief	This is Time Distributed Layer Class of Neural Network
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
#include <time_dist.h>
#include <util_func.h>

namespace nntrainer {

const std::string TimeDistLayer::type = "time_dist";

int TimeDistLayer::initialize(Manager &manager) {
  int status = ML_ERROR_NONE;

  if (getNumInputs() != 1) {
    throw std::invalid_argument("Time distributed layer takes only one input");
  }

  return status;
}

void TimeDistLayer::setProperty(const PropertyType type,
                                const std::string &value) {
  int status = ML_ERROR_NONE;
  switch (type) {
  case PropertyType::dist_layer: {
    if (!value.empty()) {
      iter_name = value;
    } else {
      status = ML_ERROR_INVALID_PARAMETER;
      throw_status(status);
    }
  }
  default:
    Layer::setProperty(type, value);
    break;
  }
}

void TimeDistLayer::forwarding(bool training) {
  // NYI
}

void TimeDistLayer::copy(std::shared_ptr<Layer> l) {
  Layer::copy(l);

  std::shared_ptr<TimeDistLayer> from =
    std::static_pointer_cast<TimeDistLayer>(l);
  this->iter = from->iter;
}

void TimeDistLayer::calcDerivative() {
  // NYI
}

void TimeDistLayer::calcGradient() {
  // NYI
}

} /* namespace nntrainer */
