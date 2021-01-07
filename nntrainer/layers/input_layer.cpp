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
 * @file	input_layer.cpp
 * @date	14 May 2020
 * @brief	This is Input Layer Class for Neural Network
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#include <input_layer.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <parse_util.h>

namespace nntrainer {

const std::string InputLayer::type = "input";

void InputLayer::setProperty(const PropertyType type,
                             const std::string &value) {
  int status = ML_ERROR_NONE;

  switch (type) {
  case PropertyType::normalization:
    if (!value.empty()) {
      status = setBoolean(normalization, value);
      throw_status(status);
    }
    break;
  case PropertyType::standardization:
    if (!value.empty()) {
      status = setBoolean(standardization, value);
      throw_status(status);
    }
    break;
  default:
    Layer::setProperty(type, value);
    break;
  }
}

void InputLayer::forwarding(bool training) {
  Tensor &hidden_ = net_hidden[0]->getVariableRef();
  hidden_ = net_input[0]->getVariableRef();

  if (normalization)
    hidden_.normalization_i();
  if (standardization)
    hidden_.standardization_i();
}

void InputLayer::calcDerivative() {
  throw exception::not_supported(
    "calcDerivative for input layer is not supported");
}

int InputLayer::initialize(Manager &manager) {
  int status = ML_ERROR_NONE;
  output_dim = input_dim;

  return status;
}

void InputLayer::setTrainable(bool train) {
  if (train)
    throw exception::not_supported("Input layer does not support training");

  Layer::setTrainable(false);
}

} /* namespace nntrainer */
