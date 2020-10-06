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
 * @file	optimizer.cpp
 * @date	08 April 2020
 * @brief	This is Implementation of Optimizer class
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#include <cmath>
#include <fstream>
#include <iostream>

#include <lazy_tensor.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <optimizer_internal.h>
#include <parse_util.h>
#include <util_func.h>

namespace nntrainer {

int Optimizer::initialize(std::shared_ptr<Weight> weight_list,
                          unsigned int num_weights, bool set_tensor) {
  return ML_ERROR_NONE;
}

double Optimizer::getLearningRate(int iteration) {
  double ll = learning_rate;

  if (decay_steps != -1) {
    ll = ll * pow(decay_rate, (iteration / decay_steps));
  }

  return ll;
}

void Optimizer::apply_gradients(std::shared_ptr<Weight> weight_list,
                                unsigned int num_weights, int iteration) {

  double ll = getLearningRate(iteration);

  int idx = 0;
  for (unsigned int i = 0; i < num_weights; ++i) {
    Weight &weight = weight_list.get()[i];

    if (!weight.getTrainable())
      continue;

    apply_gradient(weight, idx, ll, iteration);
    idx += 1;
  }
}

int Optimizer::setProperty(std::vector<std::string> values) {
  int status = ML_ERROR_NONE;

  for (unsigned int i = 0; i < values.size(); ++i) {
    std::string key;
    std::string value;

    status = getKeyValue(values[i], key, value);
    NN_RETURN_STATUS();

    unsigned int type = parseOptProperty(key);

    if (value.empty()) {
      return ML_ERROR_INVALID_PARAMETER;
    }

    try {
      /// @note this calls derived setProperty if available
      setProperty(static_cast<PropertyType>(type), value);
    } catch (...) {
      return ML_ERROR_INVALID_PARAMETER;
    }
  }

  try {
    checkValidation();
  } catch (...) {
    return ML_ERROR_INVALID_PARAMETER;
  }
  return status;
}

void Optimizer::checkValidation() {
  if (learning_rate <= 0.0f)
    throw std::invalid_argument("Learning rate must be positive");
}

void Optimizer::setProperty(const PropertyType type, const std::string &value) {
  int status = ML_ERROR_NONE;

  switch (type) {
  case PropertyType::learning_rate:
    status = setFloat(learning_rate, value);
    break;
  case PropertyType::decay_steps:
    status = setFloat(decay_steps, value);
    break;
  case PropertyType::decay_rate:
    status = setFloat(decay_rate, value);
    break;
  case PropertyType::continue_train:
    status = setBoolean(continue_train, value);
    break;
  default:
    ml_loge("Error: Unknown Optimizer Property Key");
    status = ML_ERROR_INVALID_PARAMETER;
    break;
  }

  throw_status(status);
}

void Optimizer::read(std::ifstream &file) {
  OptType loaded_type;
  file.read((char *)&loaded_type, sizeof(OptType));

  if (loaded_type >= OptType::UNKNOWN)
    throw std::runtime_error("Saved file has unknown optimizer");
}

void Optimizer::save(std::ofstream &file) {
  if (type >= OptType::UNKNOWN)
    throw std::runtime_error("Cannot save unknown optimizer");

  file.write((char *)&type, sizeof(OptType));
}
} // namespace nntrainer
