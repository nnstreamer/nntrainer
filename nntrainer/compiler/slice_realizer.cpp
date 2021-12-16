// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file slice_realizer.cpp
 * @date 14 October 2021
 * @brief NNTrainer graph realizer which slice the graph representation
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug No known bugs except for NYI items
 */

#include <connection.h>
#include <iterator>
#include <layer_node.h>
#include <slice_realizer.h>

#include <unordered_map>

namespace nntrainer {

SliceRealizer::SliceRealizer(const std::vector<Connection> &start_layers,
                             const std::vector<Connection> &end_layers) {
  /// discard index information as it is not needed as it is not really needed
  this->start_layers.reserve(start_layers.size());

  std::transform(start_layers.begin(), start_layers.end(),
                 std::back_inserter(this->start_layers),
                 [](const Connection &c) { return c.getName(); });

  std::transform(end_layers.begin(), end_layers.end(),
                 std::inserter(this->end_layers, this->end_layers.begin()),
                 [](const Connection &c) { return c.getName(); });
}

SliceRealizer::~SliceRealizer() {}

GraphRepresentation
SliceRealizer::realize(const GraphRepresentation &reference) {
  struct NodeInfo {
    NodeInfo() : NodeInfo(nullptr) {}
    NodeInfo(std::shared_ptr<LayerNode> node) :
      node(node),
      is_visited(false),
      to_be_added(false) {}
    std::shared_ptr<LayerNode> node; /**< set this if not visited */
    bool is_visited;                 /**< set this if visited */
    bool to_be_added;                /**< set this if it is to be added */
    std::vector<std::string> children;

    LayerNode *operator->() { return node.get(); }
  };

  /** @note mp has to be ordered map to keep the ordering of the nodes in the
   * graph */
  std::unordered_map<std::string, NodeInfo> mp; /// map point

  std::transform(
    reference.begin(), reference.end(), std::inserter(mp, mp.end()),
    [](std::shared_ptr<LayerNode> node) {
      return std::pair<std::string, NodeInfo>(node->getName(), node);
    });

  auto cur_start_layers = start_layers;
  auto cur_end_layers = end_layers;

  /** setup children before filling in the end layers */
  std::for_each(reference.begin(), reference.end(),
                [&mp](std::shared_ptr<LayerNode> node) {
                  auto node_name = node->getName();

                  for (auto i = 0u, num_node = node->getNumInputConnections();
                       i < num_node; ++i) {
                    const auto &parent = node->getInputConnectionName(i);
                    mp.at(parent).children.push_back(node_name);
                  };
                });

  if (cur_start_layers.empty()) {
    for (auto &node : mp) {
      if (node.second.node->getNumInputConnections() == 0) {
        cur_start_layers.push_back(node.second.node->getName());
      }
    }
  }

  if (cur_end_layers.empty()) {
    for (auto &node : mp) {
      if (node.second.children.size() == 0) {
        cur_end_layers.insert(node.first);
      }
    }
  }

  if (cur_start_layers.empty()) {
    throw std::runtime_error("No start layer is found, graph has a loop.");
  }

  if (cur_end_layers.empty()) {
    throw std::runtime_error("No end layer is found, graph has a loop.");
  }

  std::vector<std::string> dfs_stack;

  /** if the give node is the end node in the graph */
  auto is_end_node = [&cur_end_layers](const std::string &name) {
    auto iter = cur_end_layers.find(name);
    return iter != cur_end_layers.end();
  };

  /** add node to be included to subgraph */
  auto update_processed = [&mp](const std::string &name) {
    auto &node_info = mp.at(name);
    node_info.to_be_added = true;
  };

  /** dfs function to perform depth-first search recursively with tracking */
  std::function<void(const std::string &name)> dfs =
    [&dfs, &mp, &dfs_stack, &is_end_node,
     &update_processed](const std::string &name) {
      auto &node_info = mp.at(name);
      /** if node already added or end node, add the current stack to be added
       * to the subgraph */
      if (node_info.to_be_added || is_end_node(name)) {
        std::for_each(dfs_stack.begin(), dfs_stack.end(), update_processed);
        update_processed(name);
      }

      /** if node is visited, return */
      if (node_info.is_visited) {
        return;
      }

      node_info.is_visited = true;
      dfs_stack.push_back(name);
      /** run dfs on all the children */
      for (auto const &child : node_info.children) {
        dfs(child);
      }
      dfs_stack.pop_back();
    };

  /** run dfs from all the starting layers */
  for (auto &name : cur_start_layers) {
    dfs(name);
  }

  /** created the subgraph */
  GraphRepresentation subgraph;
  /** @note: iterate over reference than over mp to ensure the correct ordering
   * of layers */
  for (auto &node : reference) {
    if (mp[node->getName()].to_be_added) {
      subgraph.push_back(node);
    }
  }

  NNTR_THROW_IF(subgraph.empty(), std::invalid_argument)
    << "After slice, there is no node left, please check if configuration is "
       "correct";

  return subgraph;
}

} // namespace nntrainer
