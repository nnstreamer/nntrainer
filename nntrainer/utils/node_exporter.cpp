// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file node_exporter.cpp
 * @date 09 April 2021
 * @brief NNTrainer Node exporter
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug No known bugs except for NYI items
 */

#include <node_exporter.h>

namespace nntrainer {

template <>
const std::vector<std::pair<std::string, std::string>> &
Exporter::get_result<ExportMethods::METHOD_STRINGVECTOR>() {
  if (!is_exported) {
    throw std::invalid_argument("This exporter has not exported anything yet");
  }

  return stored_result;
}

} // namespace nntrainer
