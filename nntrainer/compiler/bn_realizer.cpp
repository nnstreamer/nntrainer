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

static constexpr size_t SINGLE_INOUT_IDX = 0;

BnRealizer::BnRealizer() {}

BnRealizer::~BnRealizer() {}

GraphRepresentation BnRealizer::realize(const GraphRepresentation &reference) {
  std::unordered_map<std::string, LayerNode *> existing_nodes;
  std::vector<LayerNode *> bn_layers;

  std::transform(
    reference.begin(), reference.end(),
    std::inserter(existing_nodes, existing_nodes.end()),
    [](auto &node) { return std::pair(node->getName(), node.get()); });

  for (auto &node : reference) {
    if (istrequal(node->getType(), "batch_normalization")) {
      bn_layers.push_back(node.get());
    }
  }

  for (auto iter = bn_layers.begin(); iter != bn_layers.end(); ++iter) {
    auto node = (*iter);
    auto &input_name = node->getInputConnectionName(SINGLE_INOUT_IDX);
    auto input_node = existing_nodes.at(input_name);

    for (unsigned int i = 0; i < input_node->getNumOutputConnections(); ++i) {
      if (istrequal(node->getName(),
                    input_node->getOutputConnection(i)->getName())) {
        input_node->setOutputConnection(
          i, node->getOutputConnection(SINGLE_INOUT_IDX)->getName(),
          SINGLE_INOUT_IDX);
      }
    }

    auto &output_name = node->getOutputConnection(SINGLE_INOUT_IDX)->getName();
    auto output_node = existing_nodes.at(output_name);

    for (unsigned int i = 0; i < output_node->getNumInputConnections(); ++i) {
      if (istrequal(node->getName(), output_node->getInputConnectionName(i))) {
        output_node->setInputConnectionName(
          i, node->getInputConnectionName(SINGLE_INOUT_IDX));
      }
    }
  }

  GraphRepresentation processed;
  for (auto &node : reference) {
    if (!istrequal(node->getType(), "batch_normalization")) {
      processed.push_back(node);
    }
  }

  return processed;
}

} // namespace nntrainer
