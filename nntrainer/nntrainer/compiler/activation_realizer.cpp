// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file activation_realizer.cpp
 * @date 23 November 2021
 * @brief NNTrainer graph realizer which realizes activation!=none to actual
 * activation node
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include "nntrainer_error.h"
#include <activation_realizer.h>
#include <remap_realizer.h>

#include <activation_layer.h>
#include <layer_node.h>
#include <stdexcept>
#include <unordered_map>

namespace nntrainer {

ActivationRealizer::~ActivationRealizer() {}

GraphRepresentation
ActivationRealizer::realize(const GraphRepresentation &reference) {
  GraphRepresentation processed;
  processed.reserve(reference.size());

  std::unordered_map<std::string /**< layer_name */,
                     std::string /**< act_layer_name */>
    remap_table;
  std::unordered_map<std::string /**< temp_layer_name */,
                     std::string /**< layer_name */>
    recovery_table;
  std::vector<LayerNode *> act_nodes;

  for (auto &node : reference) {
    processed.push_back(node);
    if (node->getType() == ActivationLayer::type) {
      /// realizing activation type not allowed because there is no good way to
      /// tell between 1. it is activation layer, 2. activation layer wants
      /// realization for now. later, we should have relu, sigmoid kind as layer
      /// type to resolve this matter, this is checked
      /// node->getActivationToBeRealized() but explicitly stated in order to
      /// make it robust, plus we might want to change the behavior
      continue;
    }
    if (auto act = node->getActivationToBeRealized();
        act != ActivationType::ACT_NONE) {
      NNTR_THROW_IF(act == ActivationType::ACT_UNKNOWN, std::invalid_argument)
        << "unknown activation type for layer: " << node->getName();

      auto layer_name = node->getName();
      props::Activation act_prop;
      act_prop.set(act);
      node->setProperty({"activation=none"});

      auto act_name = layer_name + "/activation_realized";
      auto temp_name = act_name + "/temp";
      remap_table.insert({layer_name, act_name});
      recovery_table.insert({temp_name, layer_name});
      auto act_node =
        createLayerNode("activation", {"name=" + act_name,
                                       "activation=" + to_string(act_prop)});
      act_node->setProperty({"input_layers=" + temp_name});
      processed.push_back(std::move(act_node));
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
