// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   optimizer_impl.cpp
 * @date   18 March 2021
 * @brief  This is base Optimizer implementation class
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#include <fstream>
#include <iostream>

#include <cmath>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <optimizer_impl.h>
#include <parse_util.h>
#include <util_func.h>

namespace nntrainer {

void OptimizerImpl::setProperty(const std::vector<std::string> &values) {
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

void OptimizerImpl::setProperty(const std::string &key,
                                const std::string &value) {
  int status = ML_ERROR_NONE;
  PropertyType type = static_cast<PropertyType>(parseOptProperty(key));

  switch (type) {
  case PropertyType::learning_rate:
    status = setFloat(learning_rate, value);
    break;
  case PropertyType::decay_steps:
    status = setUint(decay_steps, value);
    break;
  case PropertyType::decay_rate:
    status = setFloat(decay_rate, value);
    break;
  case PropertyType::continue_train:
    status = setBoolean(continue_train, value);
    break;
  default:
    status = ML_ERROR_INVALID_PARAMETER;
    break;
  }

  throw_status(status);
}

double OptimizerImpl::getLearningRate(size_t iteration) const {
  double ll = learning_rate;

  if (decay_steps != 0) {
    ll = ll * pow(decay_rate, (iteration / (float)decay_steps));
  }

  return ll;
}

} // namespace nntrainer
