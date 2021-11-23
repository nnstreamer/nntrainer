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

#include <activation_layer.h>
#include <layer_node.h>
#include <stdexcept>

namespace nntrainer {

ActivationRealizer::~ActivationRealizer() {}

GraphRepresentation
ActivationRealizer::realize(const GraphRepresentation &reference) {
  GraphRepresentation processed;
  processed.reserve(reference.size());

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
      node->setProperty({"name=" + layer_name + "/activation_realized"});

      auto act_node =
        createLayerNode("activation", {"name=" + layer_name,
                                       "activation=" + to_string(act_prop)});
      act_node->setProperty({"input_layers=" + node->getName()});
      processed.push_back(std::move(act_node));
    }
  }

  return processed;
}
} // namespace nntrainer
