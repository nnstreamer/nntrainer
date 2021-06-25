// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   optimizer_devel.cpp
 * @date   08 April 2020
 * @brief  This is Optimizer internal interface class
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#include <fstream>
#include <iostream>

#include <nntrainer_log.h>
#include <optimizer_devel.h>
#include <parse_util.h>
#include <util_func.h>

namespace nntrainer {

void Optimizer::applyGradients(std::vector<Weight> &weight_list,
                               int iteration) {

  if (weight_list.empty())
    return;

  double ll = getLearningRate(iteration);

  for (auto &weight : weight_list) {
    if (!weight.hasGradient())
      continue;

    /** calculate regularization gradient before applying the gradient */
    weight.calcRegularizationGradient();

    applyGradient(weight, ll, iteration);
  }
}

int Optimizer::setProperty(std::vector<std::string> values) {
  int status = ML_ERROR_NONE;

  for (unsigned int i = 0; i < values.size(); ++i) {
    std::string key;
    std::string value;

    status = getKeyValue(values[i], key, value);
    NN_RETURN_STATUS();

    if (value.empty()) {
      return ML_ERROR_INVALID_PARAMETER;
    }

    try {
      /// @note this calls derived setProperty if available
      setProperty(key, value);
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

void Optimizer::checkValidation() const {
  if (getLearningRate() <= 0.0f)
    throw std::invalid_argument("Learning rate must be positive");
}

void Optimizer::setProperty(const std::string &key, const std::string &value) {
  int status = ML_ERROR_NONE;
  unsigned int type = parseOptProperty(key);

  switch (type) {
  default:
    ml_loge("Error: Unknown Optimizer Property Key");
    status = ML_ERROR_INVALID_PARAMETER;
    break;
  }

  throw_status(status);
}

void Optimizer::read(std::ifstream &file) {
  std::string loaded_type = readString(file);

  if (loaded_type != getType()) {
    throw std::runtime_error(
      "[Optimizer::read] written type unmatches with set type");
  }
}

void Optimizer::save(std::ofstream &file) { writeString(file, getType()); }

} // namespace nntrainer
