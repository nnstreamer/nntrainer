// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Eunju Yang <ej.yang@samsung.com>
 *
 * @file    subgraph.cpp
 * @date    07 Jan 2025
 * @see     https://github.com/nnstreamer/nntrainer
 * @author  Eunju Yang <ej.yang@samsung.com>
 * @bug     No known bugs except for NYI items
 */

#include <memory>
#include <subgraph.h>

namespace nntrainer {

SubGraphNode createSubGraph(const std::vector<std::string> &properties) {

  if (getComputeEngine(properties) == ml::train::LayerComputeEngine::CPU) {
    const auto &sg_node = SGNODE(std::make_shared<SubGraphCpu>());
    sg_node->setProperty(properties);
    return sg_node;
  } else {
    throw std::runtime_error(
      "Subgraph for the computing engine is not yet supported");
  }
};

} // namespace nntrainer
