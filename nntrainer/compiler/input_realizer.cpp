// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file input_realizer.cpp
 * @date 14 October 2021
 * @brief NNTrainer graph realizer which remaps input to the external graph
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <connection.h>
#include <input_realizer.h>
#include <layer_node.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>

#include <algorithm>
#include <stdexcept>
#include <unordered_map>

namespace nntrainer {
InputRealizer::InputRealizer(const std::vector<Connection> &start_conns,
                             const std::vector<Connection> &input_conns) :
  start_conns(start_conns),
  input_conns(input_conns) {
  NNTR_THROW_IF(start_conns.size() != input_conns.size(), std::invalid_argument)
    << "start connection size is not same input_conns size";
}

InputRealizer::~InputRealizer() {}

GraphRepresentation
InputRealizer::realize(const GraphRepresentation &reference) {
  std::unordered_map<std::string, LayerNode *> existing_nodes;

  std::transform(
    reference.begin(), reference.end(),
    std::inserter(existing_nodes, existing_nodes.end()),
    [](auto &node) { return std::pair(node->getName(), node.get()); });

  for (unsigned i = 0u, sz = start_conns.size(); i < sz; ++i) {
    const auto &sc = start_conns[i];
    const auto &ic = input_conns[i];
    auto node = existing_nodes.at(sc.getName());

    auto num_connection = node->getNumInputConnections();
    if (num_connection == 0) {
      NNTR_THROW_IF(sc.getIndex() != 0, std::invalid_argument)
        << "start connection: " << sc.toString()
        << " not defined and num connection of that node is empty, although "
           "start connection of index zero is allowed";
      node->setProperty({"input_layers=" + ic.toString()});
    } else {
      NNTR_THROW_IF(sc.getIndex() >= num_connection, std::invalid_argument)
        << "start connection: " << sc.toString()
        << " not defined, num connection: " << num_connection;
      node->setInputConnectionName(sc.getIndex(), ic.getName());
      node->setInputConnectionIndex(sc.getIndex(), ic.getIndex());
    }
  }

  return reference;
}

} // namespace nntrainer
