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
#include <layer_internal.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <parse_util.h>

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

void InputLayer::setProperty(const std::vector<std::string> &values) {
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

void InputLayer::setProperty(const std::string &type_str,
                             const std::string &value) {
  using PropertyType = LayerV1::PropertyType;
  int status = ML_ERROR_NONE;
  LayerV1::PropertyType type =
    static_cast<LayerV1::PropertyType>(parseLayerProperty(type_str));

  switch (type) {
  case PropertyType::normalization: {
    status = setBoolean(normalization, value);
    throw_status(status);
  } break;
  case PropertyType::standardization: {
    status = setBoolean(standardization, value);
    throw_status(status);
  } break;
  default:
    std::string msg =
      "[InputLayer] Unknown Layer Property Key for value " + std::string(value);
    throw exception::not_supported(msg);
  }
}

void InputLayer::forwarding(RunLayerContext &context, bool training) {
  Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);
  hidden_ = context.getInput(SINGLE_INOUT_IDX);

  if (normalization)
    hidden_.normalization_i();
  if (standardization)
    hidden_.standardization_i();
}

void InputLayer::calcDerivative(RunLayerContext &context) {
  throw exception::not_supported(
    "calcDerivative for input layer is not supported");
}

void InputLayer::finalize(InitLayerContext &context) {
  context.setOutputDimensions(context.getInputDimensions());
}

} /* namespace nntrainer */
