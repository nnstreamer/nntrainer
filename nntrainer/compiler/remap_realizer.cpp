// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file remap_realizer.h
 * @date 12 October 2021
 * @brief NNTrainer graph realizer which realizes identifier to a new identifier
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <remap_realizer.h>

#include <layer_node.h>
namespace nntrainer {

RemapRealizer::RemapRealizer(
  std::function<void(std::string &, unsigned &)> remap_connection_function) :
  remap_fn(nullptr),
  remap_connection_fn(remap_connection_function) {
  if (!remap_connection_fn) {
    throw std::invalid_argument("remap function is not given!");
  }
}

RemapRealizer::RemapRealizer(
  std::function<void(std::string &)> remap_function) :
  remap_fn(remap_function),
  remap_connection_fn(nullptr) {
  if (!remap_fn) {
    throw std::invalid_argument("remap function is not given!");
  }
}

RemapRealizer::~RemapRealizer() {}

GraphRepresentation
RemapRealizer::realize(const GraphRepresentation &reference) {
  GraphRepresentation processed(reference.begin(), reference.end());

  for (auto &node : processed) {
    /// @note while remap realization, the graph is invalid.
    remap_connection_fn ? node->remapConnections(remap_connection_fn)
                        : node->remapIdentifiers(remap_fn);
  }
  return processed;
}

} // namespace nntrainer
