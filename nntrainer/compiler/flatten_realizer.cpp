// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file flatten_realizer.h
 * @date 09 October 2021
 * @brief NNTrainer graph realizer which realizes flatten=true to actual node
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <flatten_realizer.h>

#include <flatten_layer.h>
#include <layer_node.h>

namespace nntrainer {

FlattenRealizer::~FlattenRealizer() {}

GraphRepresentation
FlattenRealizer::realize(const GraphRepresentation &reference) {
  GraphRepresentation processed;
  processed.reserve(reference.size());

  for (auto &node : reference) {
    /// @note: [node] type=flatten; flatten=true; is awkward but allowed.
    /// There is no reason to prohibit this.
    processed.push_back(node);
    if (node->getFlatten() && !node->getDistribute()) {
      auto layer_name = node->getName();
      auto flatten_node =
        createLayerNode(FlattenLayer::type, {"name=" + layer_name});
      node->setProperty({"flatten=false"});
      node->setProperty({"name=" + layer_name + "/flatten_realized"});
      flatten_node->setProperty({"input_layers=" + node->getName()});
      processed.push_back(std::move(flatten_node));
    }
  }

  return processed;
}
} // namespace nntrainer
