// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file remap_realizer.h
 * @date 12 October 2021
 * @brief NNTrainer graph realizer which realizes identifer to a new identifier
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <remap_realizer.h>

#include <layer_node.h>
namespace nntrainer {

RemapRealizer::RemapRealizer(
  std::function<void(std::string &)> remap_function) :
  remap_fn(remap_function) {
  if (!remap_fn) {
    throw std::invalid_argument("remap function is not given!");
  }
}

RemapRealizer::~RemapRealizer() {}

GraphRepresentation
RemapRealizer::realize(const GraphRepresentation &reference) {
  GraphRepresentation processed(reference.begin(), reference.end());
  for (auto &node : processed) {
    node->remapIdentifiers(remap_fn);
  }
  return processed;
}

} // namespace nntrainer
