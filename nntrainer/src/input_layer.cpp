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

int InputLayer::setOptimizer(Optimizer &opt) {
  this->opt.setType(opt.getType());
  this->opt.setOptParam(opt.getOptParam());

  return this->opt.initialize(dim, false);
}

int InputLayer::setProperty(std::vector<std::string> values) {
  int status = ML_ERROR_NONE;
  for (unsigned int i = 0; i < values.size(); ++i) {
    std::string key;
    std::string value;

    status = getKeyValue(values[i], key, value);
    NN_RETURN_STATUS();

    unsigned int type = parseLayerProperty(key.c_str());

    switch (static_cast<PropertyType>(type)) {
    case PropertyType::input_shape:
      status = dim.setTensorDim(value.c_str());
      NN_RETURN_STATUS();
      break;
    case PropertyType::bias_zero:
      status = setBoolean(init_zero, value);
      NN_RETURN_STATUS();
      break;
    case PropertyType::normalization:
      status = setBoolean(normalization, value);
      NN_RETURN_STATUS();
      break;
    case PropertyType::standardization:
      status = setBoolean(standardization, value);
      NN_RETURN_STATUS();
      break;
    default:
      ml_loge("Error: Unknown Layer Property Key : %s", key.c_str());
      status = ML_ERROR_INVALID_PARAMETER;
      break;
    }
  }
  return status;
}

void InputLayer::copy(std::shared_ptr<Layer> l) {
  std::shared_ptr<InputLayer> from = std::static_pointer_cast<InputLayer>(l);
  this->opt = from->opt;
  this->last_layer = from->last_layer;
  this->dim = from->dim;
  this->input.copy(from->input);
  this->hidden.copy(from->hidden);
}

Tensor InputLayer::forwarding(Tensor in, int &status) {
  input = in;
  if (normalization)
    input = input.normalization();
  return input;
}

int InputLayer::initialize(bool last) {
  int status = ML_ERROR_NONE;
  if (dim.batch() <= 0 || dim.channel() <= 0 || dim.height() <= 0 ||
      dim.width() <= 0) {
    ml_loge("Error: Dimension must be greater than 0");
    return ML_ERROR_INVALID_PARAMETER;
  }

  return status;
}

int InputLayer::initialize(int b, int c, int h, int w, bool last,
                           bool init_zero) {
  int status = ML_ERROR_NONE;

  this->dim.batch(b);
  this->dim.channel(c);
  this->dim.width(w);
  this->dim.height(h);

  status = initialize(last);

  return status;
}
} /* namespace nntrainer */
