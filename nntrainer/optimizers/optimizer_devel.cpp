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

#include <nntrainer_error.h>
#include <optimizer_devel.h>
#include <util_func.h>

namespace nntrainer {

void Optimizer::setProperty(const std::vector<std::string> &values) {
  if (!values.empty()) {
    std::string msg = "[OptimizerDevel] Unknown properties count " +
                      std::to_string(values.size());
    throw exception::not_supported(msg);
  }
}

void Optimizer::read(std::ifstream &file) {
  std::string loaded_type;
  unsigned int opt_type = ml::train::OptimizerType::UNKNOWN;
  checkedRead(file, (char *)&opt_type, sizeof(opt_type));
  switch (opt_type) {
  case ml::train::OptimizerType::ADAM:
    loaded_type = "adam";
    break;
  case ml::train::OptimizerType::SGD:
    loaded_type = "sgd";
    break;
  default:
    break;
  }

  if (loaded_type != getType()) {
    throw std::runtime_error(
      "[Optimizer::read] written type unmatches with set type");
  }
}

void Optimizer::save(std::ofstream &file) {
  unsigned int opt_type = ml::train::OptimizerType::UNKNOWN;
  if (istrequal(getType(), "adam"))
    opt_type = ml::train::OptimizerType::ADAM;
  if (istrequal(getType(), "sgd"))
    opt_type = ml::train::OptimizerType::SGD;
  file.write((char *)&opt_type, sizeof(opt_type));
}

} // namespace nntrainer
