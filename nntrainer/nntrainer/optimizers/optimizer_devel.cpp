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
  std::string loaded_type = readString(file);

  if (loaded_type != getType()) {
    throw std::runtime_error(
      "[Optimizer::read] written type unmatches with set type");
  }
}

void Optimizer::save(std::ofstream &file) { writeString(file, getType()); }

} // namespace nntrainer
