// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2022 seongwoo <mhs4670go@naver.com>
 *
 * @file tflite_export_realizer.cpp
 * @date 18 July 2025
 * @brief NNTrainer graph realizer which remove loss layer for inference
 * @see	https://github.com/nnstreamer/nntrainer
 * @author seongwoo <mhs4670go@naver.com>
 * @author donghak park <donghak.park@samsung.com>
 * @bug No known bugs except for NYI items
 */

#include <algorithm>
#include <cassert>
#include <connection.h>
#include <layer_node.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <set>
#include <stdexcept>
#include <string>
#include <tflite_export_realizer.h>
#include <unordered_map>

namespace nntrainer {

static constexpr size_t SINGLE_IN_IDX = 0;

GraphRepresentation
TfliteExportRealizer::realize(const GraphRepresentation &reference) {
  /// @todo support more loss layers
  /// @note Some layers need to consider not removing all semantics.
  /// For example, When CrossEntropySigmoidLossLayer needs to be removed,
  /// sigmoid computation shouldn't be removed.
  static const std::set<std::string> loss_type = {"mse"};
  std::unordered_map<std::string, LayerNode *> existing_nodes;
  std::vector<LayerNode *> loss_layers;

  std::transform(
    reference.begin(), reference.end(),
    std::inserter(existing_nodes, existing_nodes.end()),
    [](auto &node) { return std::pair(node->getName(), node.get()); });

  for (auto &node : reference) {
    if (loss_type.find(node->getType()) != loss_type.end()) {
      loss_layers.push_back(node.get());
    }
  }

  for (auto iter = loss_layers.begin(); iter != loss_layers.end(); ++iter) {
    auto loss_node = (*iter);
    assert(loss_node->getNumInputConnections() == 1);
    auto &input_name = loss_node->getInputConnectionName(SINGLE_IN_IDX);
    auto input_node = existing_nodes.at(input_name);
    for (unsigned int i = 0; i < input_node->getNumOutputConnections(); ++i) {
      if (istrequal(loss_node->getName(),
                    input_node->getOutputConnection(i)->getName())) {
        /// Assume that loss layers don't have output connections
        assert(loss_node->getOutputConnections().size() == 0);
        input_node->setOutputLayers({});
      }
    }
  }

  GraphRepresentation processed;
  for (auto &node : reference) {
    if (loss_type.find(node->getType()) == loss_type.end()) {
      processed.push_back(node);
    }
  }

  return processed;
}

GraphRepresentation
TfliteExportRealizer::realize_dropout(const GraphRepresentation &reference) {
  static const std::set<std::string> dropout_type = {"dropout"};
  std::unordered_map<std::string, LayerNode *> existing_nodes;
  std::vector<LayerNode *> dropout_layers;

  std::transform(
    reference.begin(), reference.end(),
    std::inserter(existing_nodes, existing_nodes.end()),
    [](auto &node) { return std::pair(node->getName(), node.get()); });

  // find dropout layer and push to vector
  for (auto &node : reference) {
    if (dropout_type.find(node->getType()) != dropout_type.end()) {
      dropout_layers.push_back(node.get());
    }
  }

  for (auto iter = dropout_layers.begin(); iter != dropout_layers.end();
       ++iter) {
    auto node = (*iter);
    auto &input_name = node->getInputConnectionName(SINGLE_IN_IDX);
    auto input_node = existing_nodes.at(input_name);

    for (unsigned int i = 0; i < input_node->getNumOutputConnections(); ++i) {
      if (istrequal(node->getName(),
                    input_node->getOutputConnection(i)->getName())) {
        input_node->setOutputConnection(
          i, node->getOutputConnection(i)->getName(), SINGLE_IN_IDX);
      }
      input_node->getOutput(SINGLE_IN_IDX)
        .setData(node->getOutput(SINGLE_IN_IDX).getMemoryData());
    }

    auto &output_name = node->getOutputConnection(SINGLE_IN_IDX)->getName();
    auto output_node = existing_nodes.at(output_name);

    for (unsigned int i = 0; i < output_node->getNumInputConnections(); ++i) {
      if (istrequal(node->getName(), output_node->getInputConnectionName(i))) {
        output_node->setInputConnectionName(i, node->getInputConnectionName(i));
      }
    }
  }

  GraphRepresentation processed;
  for (auto &node : reference) {
    if (dropout_type.find(node->getType()) == dropout_type.end()) {
      processed.push_back(node);
    }
  }

  return processed;
}
} // namespace nntrainer
