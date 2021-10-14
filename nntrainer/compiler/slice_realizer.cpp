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
      is_added(false) {}
    std::shared_ptr<LayerNode> node; /**< set this if not visited */
    bool is_visited;                 /**< set this if visited */
    bool is_added;                   /**< set this if added */
    std::vector<std::string> children;
    std::vector<std::string> path;
    /**< path is the tracing result from start to current node
      eg) if traversal has started from a -> b -> c -> d.
      The path has {"a", "b", "c", "d"} */

    LayerNode *operator->() { return node.get(); }
  };

  std::unordered_map<std::string, NodeInfo> mp; /// map point
  std::transform(
    reference.begin(), reference.end(), std::inserter(mp, mp.end()),
    [](std::shared_ptr<LayerNode> node) {
      return std::pair<std::string, NodeInfo>(node->getName(), node);
    });

  auto cur_start_layers = start_layers;
  auto cur_end_layers = end_layers;

  if (start_layers.empty()) {
    for (auto &node : reference) {
      if (node->getNumInputConnections() == 0) {
        cur_start_layers.push_back(node->getName());
      }
    }
  }

  if (end_layers.empty()) {
    for (auto &node : mp) {
      if (node.second.children.size() == 0) {
        cur_end_layers.insert(node.first);
      }
    }
  }

  std::for_each(reference.begin(), reference.end(),
                [&mp](std::shared_ptr<LayerNode> node) {
                  auto node_name = node->getName();
                  for (auto &parent : node->getInputLayers()) {
                    mp.at(parent).children.push_back(node_name);
                  };
                });

  GraphRepresentation processed;

  auto update_processed = [&processed, &mp](const std::string &name) {
    auto &node_info = mp.at(name);
    if (!node_info.is_added) {
      processed.push_back(node_info.node);
      node_info.is_added = true;
    }
  };

  std::vector<std::string> dfs_stack(cur_start_layers.rbegin(),
                                     cur_start_layers.rend());

  auto is_end_node = [&cur_end_layers](const std::string &name) {
    auto iter = cur_end_layers.find(name);
    return iter != cur_end_layers.end();
  };
  while (!dfs_stack.empty()) {
    auto &node_info = mp.at(dfs_stack.back());
    auto &path = node_info.path;
    path.push_back(node_info->getName());
    if (is_end_node(node_info->getName())) {
      std::for_each(path.begin(), path.end(), update_processed);
    }

    dfs_stack.pop_back();
    node_info.is_visited = true;

    auto &children = node_info.children;
    std::for_each(children.begin(), children.end(),
                  [&path, &mp](const auto &name) { mp.at(name).path = path; });

    /// @todo: stop inserting to the dfs stack if children->isAdded == true
    dfs_stack.insert(dfs_stack.end(), children.rbegin(), children.rend());
  }

  NNTR_THROW_IF(processed.empty(), std::invalid_argument)
    << "After slice, there is no node left, please check if configuration is "
       "correct";

  return processed;
}

} // namespace nntrainer
