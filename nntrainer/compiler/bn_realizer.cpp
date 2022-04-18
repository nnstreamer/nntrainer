// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2022 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file bn_realizer.cpp
 * @date 13 April 2022
 * @brief NNTrainer graph realizer which remove batch normalization layer for
 * inference
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <bn_realizer.h>
#include <connection.h>
#include <layer_node.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>

#include <algorithm>
#include <stdexcept>
#include <unordered_map>

namespace nntrainer {
BnRealizer::BnRealizer() {}

BnRealizer::~BnRealizer() {}

GraphRepresentation BnRealizer::realize(const GraphRepresentation &reference) {
  std::unordered_map<std::string, LayerNode *> existing_nodes;

  std::transform(
    reference.begin(), reference.end(),
    std::inserter(existing_nodes, existing_nodes.end()),
    [](auto &node) { return std::pair(node->getName(), node.get()); });

  // NYI

  return reference;
}

} // namespace nntrainer
