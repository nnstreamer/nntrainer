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
#include <layer_node.h>
#include <slice_realizer.h>

#include <unordered_map>

namespace nntrainer {

SliceRealizer::SliceRealizer(const std::vector<std::string> &start_layers,
                             const std::vector<std::string> &end_layers) :
  start_layers(start_layers),
  end_layers(end_layers.begin(), end_layers.end()) {}

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
    bool to_be_added;                   /**< set this if it is to be added */
    std::vector<std::string> children;
    std::vector<std::string> path;
    /**< path is the tracing result from start to current node
      eg) if traversal has started from a -> b -> c -> d.
      The path has {"a", "b", "c", "d"} */

    LayerNode *operator->() { return node.get(); }
  };

  /** @note mp has to be ordered map to keep the ordering of the nodes in the
   * graph */
  std::map<std::string, NodeInfo> mp; /// map point

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
                  for (auto &parent : node->getInputLayers()) {
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
    throw std::runtime_error("No start layer is found, graph has a loop.");
  }

  std::vector<std::string> dfs_stack(cur_start_layers.rbegin(),
                                     cur_start_layers.rend());

  auto is_end_node = [&cur_end_layers](const std::string &name) {
    auto iter = cur_end_layers.find(name);
    return iter != cur_end_layers.end();
  };

  auto update_processed = [&mp](const std::string &name) {
    auto &node_info = mp.at(name);
    node_info.to_be_added = true;
  };

  while (!dfs_stack.empty()) {
    auto &node_info = mp.at(dfs_stack.back());
    dfs_stack.pop_back();

    /** if end node or added, add the current stack */
    if (node_info.to_be_added || is_end_node(node_info->getName())) {
      node_info.to_be_added = true;
      std::for_each(node_info.path.begin(), node_info.path.end(),
                    update_processed);
    }

    /** if already visited, skip */
    if (node_info.is_visited) {
      continue;
    }

    auto &path = node_info.path;
    node_info.is_visited = true;
    path.push_back(node_info->getName());

    auto &children = node_info.children;
    std::for_each(children.begin(), children.end(),
                  [&path, &mp](const auto &name) { mp.at(name).path = path; });

    dfs_stack.insert(dfs_stack.end(), children.begin(), children.end());
  }

  GraphRepresentation processed;
  for (auto &node : mp) {
    if (node.second.to_be_added) {
      processed.push_back(node.second.node);
    }
  }

  NNTR_THROW_IF(processed.empty(), std::invalid_argument)
    << "After slice, there is no node left, please check if configuration is "
       "correct";

  return processed;
}

} // namespace nntrainer
