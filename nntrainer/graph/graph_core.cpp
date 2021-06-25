// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file    network_graph.h
 * @date    12 May 2020
 * @see     https://github.com/nnstreamer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @author  Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   This is Graph Core Class for Neural Network
 *
 */

#include <algorithm>
#include <sstream>

#include <graph_core.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <parse_util.h>

namespace nntrainer {

void GraphCore::addGraphNode(std::shared_ptr<GraphNode> node) {
  node->setIndex(node_list.size());
  node_list.push_back(node);
}

const std::shared_ptr<GraphNode> &GraphCore::getNode(unsigned int ith) const {
  if (ith >= size())
    throw std::invalid_argument("Exceed total number of nodes");

  if (node_list[ith]->getIndex() != ith)
    throw std::runtime_error("Graph internal index mismatch");

  return node_list[ith];
}

const std::shared_ptr<GraphNode> &
GraphCore::getSortedNode(unsigned int ith) const {
  if (ith >= Sorted.size())
    throw std::invalid_argument("Exceed total number of nodes");

  return Sorted[ith];
}

void GraphCore::makeAdjacencyList(
  std::vector<std::list<std::shared_ptr<GraphNode>>> &adj) {
  /** initialize the adj list */
  for (auto &node : node_list) {
    adj.push_back(std::list<std::shared_ptr<GraphNode>>({node}));
  }

  /** make the connections */
  for (auto &node : node_list) {
    for (auto const &in_conn : node->getInputConnections()) {
      unsigned int to_node_id = getNode(in_conn)->getIndex();
      adj[to_node_id].push_back(node);
    }
  }
}

void GraphCore::topologicalSortUtil(
  std::vector<std::list<std::shared_ptr<GraphNode>>> &adj, unsigned int ith,
  std::vector<bool> &visited,
  std::stack<std::shared_ptr<GraphNode>> &dfs_stack) {
  visited[ith] = true;

  std::list<std::shared_ptr<GraphNode>>::iterator i;
  for (i = adj[ith].begin(); i != adj[ith].end(); ++i) {
    auto index = (*i)->getIndex();
    if (!visited[index])
      topologicalSortUtil(adj, index, visited, dfs_stack);
  }

  dfs_stack.push(getNode(ith));
}

void GraphCore::topologicalSort() {
  std::vector<std::list<std::shared_ptr<GraphNode>>> adj;
  std::stack<std::shared_ptr<GraphNode>> dfs_stack;
  std::vector<bool> visited(node_list.size(), false);

  makeAdjacencyList(adj);
  Sorted.clear();

  // Quite likely this is not needed - verify this
  // TODO : After make node list of graph, we have to find root. (That means it
  // should be the only one input for now.). Need to support multiple input and
  // support search.

  for (unsigned int i = 0; i < adj.size(); ++i) {
    if (visited[i] == false) {
      topologicalSortUtil(adj, i, visited, dfs_stack);
    }
  }

  while (dfs_stack.empty() == false) {
    Sorted.push_back(dfs_stack.top());
    dfs_stack.pop();
  }

  if (Sorted.size() != node_list.size())
    throw std::runtime_error("Internal error in topologicalSort");
}

const std::shared_ptr<GraphNode> &
GraphCore::getNode(const std::string &name) const {
  for (auto &lnode : node_list) {
    if (istrequal(lnode->getName(), name))
      return lnode;
  }

  std::stringstream ss;
  ss << "Cannot find graph node: " << name;
  throw std::invalid_argument(ss.str());
}

void GraphCore::addNode(std::shared_ptr<GraphNode> node, bool ensure_name) {
  /** Ensure that the node has a name and is unique */
  if (ensure_name)
    ensureName(*node);

  /** Insert the node to the graph */
  addGraphNode(node);
}

void GraphCore::ensureName(GraphNode &node, const std::string &prefix,
                           const std::string &postfix, bool force_rename) {
  std::string orig_name = node.getName();
  bool orig_name_empty = orig_name.empty();
  /** If node already has name which is unique and valid, and force is
   * disabled, then nothing to do.
   */
  if (!orig_name_empty && !force_rename && !verifyNode(orig_name)) {
    node_names.insert(orig_name);
    return;
  }

  /** If just prefix with node name makes it unique - directly set the name */
  if (!orig_name_empty) {
    std::string direct_name = prefix + orig_name + postfix;
    if (!verifyNode(direct_name)) {
      node.setName(direct_name);
      node_names.insert(direct_name);
      return;
    }
  }

  std::unordered_set<std::string>::iterator iter;
  std::string name;
  if (orig_name_empty) {
    orig_name = node.getType();
  }

  std::string direct_name = prefix + orig_name + postfix;

  do {
    name = direct_name + std::to_string(def_name_count++);
    iter = node_names.find(name);
  } while (iter != node_names.end());

  node.setName(name);
  node_names.insert(name);
}

void GraphCore::replaceNode(std::shared_ptr<GraphNode> from,
                            std::shared_ptr<GraphNode> to) {
  unsigned int idx = from->getIndex();
  to->setIndex(idx);
  node_list[idx] = to;
}

void GraphCore::realizeInputOutputNode() {
  for (auto iter = cbegin(); iter != cend(); ++iter) {
    if (iter->getInputConnections().size() == 0) {
      input_list.push_back(*iter);
    }
    if (iter->getOutputConnections().size() == 0) {
      output_list.push_back(*iter);
    }
  }
}

} /* namespace nntrainer */
