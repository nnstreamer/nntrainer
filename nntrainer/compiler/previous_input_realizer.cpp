// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file previous_input_realizer.cpp
 * @date 18 November 2021
 * @brief NNTrainer graph realizer which connects input to previous one if empty
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <algorithm>
#include <compiler_fwd.h>
#include <memory>
#include <stdexcept>
#include <vector>

#include <layer_node.h>
#include <nntrainer_log.h>
#include <previous_input_realizer.h>

namespace nntrainer {

PreviousInputRealizer::PreviousInputRealizer(
  const std::vector<std::string> &identified_inputs_) :
  identified_inputs(identified_inputs_) {}

PreviousInputRealizer::~PreviousInputRealizer() {}

GraphRepresentation
PreviousInputRealizer::realize(const GraphRepresentation &reference) {
  GraphRepresentation processed(reference.begin(), reference.end());

  /**
   * @brief for node has input connection, below function determines if the node
   * should be input node or add input_layers from previous layer
   *
   */
  auto is_actually_an_input_node = [this](const LayerNode &node) {
    return node.hasInputShapeProperty() or
           std::any_of(identified_inputs.begin(), identified_inputs.end(),
                       [&node](auto &name) { return node.getName() == name; });
  };

  for (auto iter = processed.begin(); iter != processed.end(); ++iter) {
    auto &node = *iter;
    if (node->getNumInputConnections() != 0) {
      continue;
    }

    if (is_actually_an_input_node(*node)) {
      continue;
    }

    NNTR_THROW_IF(iter == processed.begin(), std::invalid_argument)
      << "First node must be identified as an input if it is qualified to be "
         "input, name: "
      << node->getName();

    auto &prev_node = *(iter - 1);
    ml_logi(
      "%s is identified as a non-input node and default input layer(%s) is "
      "being set ",
      node->getName().c_str(), prev_node->getName().c_str());

    node->setInputLayers({prev_node->getName()});
  }

  return processed;
}

} // namespace nntrainer
