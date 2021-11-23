// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file inputremap_realizer.cpp
 * @date 14 October 2021
 * @brief NNTrainer graph realizer which remaps input to the external graph
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <input_realizer.h>
#include <layer_node.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>

#include <algorithm>
#include <unordered_map>

namespace nntrainer {
InputRealizer::InputRealizer(const std::vector<std::string> &start_layers,
                             const std::vector<std::string> &input_layers) :
  start_layers(start_layers),
  input_layers(input_layers) {}

InputRealizer::~InputRealizer() {}

GraphRepresentation
InputRealizer::realize(const GraphRepresentation &reference) {
  std::unordered_map<std::string, LayerNode *> existing_nodes;

  std::transform(
    reference.begin(), reference.end(),
    std::inserter(existing_nodes, existing_nodes.end()),
    [](auto &node) { return std::pair(node->getName(), node.get()); });

  /// if start_layer is empty, it's not a hard error but likely to be wrong if
  /// there is two inputs
  ml_logw("trying to realize without start_layer specified, if there is more "
          "than two inputs, sort order make setting graph not determinated");

  auto get_next_input_ref = [input_ref_iter = input_layers.begin(),
                             this]() mutable {
    NNTR_THROW_IF(input_ref_iter == input_layers.end(), std::invalid_argument)
      << "there is no more input layers";
    return input_ref_iter++;
  };

  for (auto &start_name : start_layers) {
    auto node = existing_nodes.at(start_name);

    auto num_input = node->getNumInputConnections();

    if (num_input == 0) {
      // case1. There is no input layers presented -> push single input
      node->setProperty({"input_layers=" + *get_next_input_ref()});
    } else {
      /// case2. There is multiple input layers -> substitute orphaned node
      /// Orphaned node probably is being created from slicing or it is also a
      /// possible scenario that the graph in the first place is designed to
      /// have a orphaned node. In the latter case, the graph was non-compilable
      /// from the first time.
      for (auto i = 0u; i < num_input; ++i) {
        auto name = node->getInputConnectionName(i);
        if (!existing_nodes.count(name)) {
          node->setInputConnectionName(i, *get_next_input_ref());
        }
      }
    }
  }

  return reference;
}

} // namespace nntrainer
