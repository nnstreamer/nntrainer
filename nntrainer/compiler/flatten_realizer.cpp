// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file flatten_realizer.cpp
 * @date 09 October 2021
 * @brief NNTrainer graph realizer which realizes flatten=true to actual node
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <flatten_realizer.h>
#include <remap_realizer.h>
#include <unordered_map>

#include <flatten_layer.h>
#include <layer_node.h>

namespace nntrainer {

FlattenRealizer::~FlattenRealizer() {}

GraphRepresentation
FlattenRealizer::realize(const GraphRepresentation &reference) {
  GraphRepresentation processed;
  processed.reserve(reference.size());

  std::unordered_map<std::string /**< layer_name */,
                     std::string /**< flatten_layer_name */>
    remap_table;
  std::vector<LayerNode *> flatten_nodes;
  std::unordered_map<std::string /**< temp_layer_name */,
                     std::string /**< layer_name */>
    recovery_table;

  for (auto &node : reference) {
    /// @note: [node] type=flatten; flatten=true; is awkward but allowed.
    /// There is no reason to prohibit this.
    processed.push_back(node);
    if (node->getFlatten() && !node->getDistribute()) {
      node->setProperty({"flatten=false"});

      auto layer_name = node->getName();

      auto flatten_name = layer_name + "/flatten_realized";
      auto temp_name = flatten_name + "/temp";

      remap_table.insert({layer_name, flatten_name});
      recovery_table.insert({temp_name, layer_name});

      auto flatten_node =
        createLayerNode(FlattenLayer::type, {"name=" + flatten_name});
      flatten_node->setProperty({"input_layers=" + temp_name});
      processed.push_back(std::move(flatten_node));
    }
  }
  processed =
    RemapRealizer([&remap_table](std::string &name, unsigned &idx) {
      if (auto iter = remap_table.find(name); iter != remap_table.end()) {
        name = iter->second;
      }
    })
      .realize(processed);
  processed =
    RemapRealizer([&recovery_table](std::string &name, unsigned &idx) {
      if (auto iter = recovery_table.find(name); iter != recovery_table.end()) {
        name = iter->second;
      }
    })
      .realize(processed);

  return processed;
}
} // namespace nntrainer
