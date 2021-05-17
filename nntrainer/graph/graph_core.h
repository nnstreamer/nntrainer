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

#ifndef __GRAPH_CORE_H__
#define __GRAPH_CORE_H__
#ifdef __cplusplus

#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <set>
#include <stack>
#include <vector>

#include <graph_node.h>

namespace nntrainer {

/**
 * @class   Graph Core Class
 * @brief   Graph Core Class which provides core graph functionalities
 */
class GraphCore {

public:
  /**
   * @brief     Constructor of Graph Core Class
   */
  GraphCore() : def_name_count(0) {}

  /**
   * @brief Add the given node into Graph
   * @param[in] node shared_ptr of node
   */
  void addNode(std::shared_ptr<GraphNode> node, bool ensure_name = true);

  /**
   * @brief getter of number of nodes
   * @param[out] number of nodes
   */
  unsigned int size() const {
    if (Sorted.empty())
      return adj.size();
    else
      return Sorted.size();
  }

  /**
   * @brief get if the graph is empty
   * @param[out] true if empty, else false
   */
  bool empty() const {
    if (Sorted.empty())
      return adj.empty();
    else
      return Sorted.empty();
  }

  /**
   * @brief     Swap function for the class
   */
  friend void swap(GraphCore &lhs, GraphCore &rhs) {
    using std::swap;

    swap(lhs.adj, rhs.adj);
    swap(lhs.node_list, rhs.node_list);
    swap(lhs.Sorted, rhs.Sorted);
    swap(lhs.node_names, rhs.node_names);
    swap(lhs.def_name_count, rhs.def_name_count);
  }

  /**
   * @brief     reset the graph
   */
  void reset() {
    adj.clear();
    Sorted.clear();
    node_names.clear();
    def_name_count = 0;
  }

  /**
   * @brief getter of GraphNode with index number
   * @param[in] index
   * @ret GraphNode
   */
  const std::shared_ptr<GraphNode> &getNode(unsigned int ith) const;

  /**
   * @brief getter of Sorted GraphNode with index number
   * @param[in] index
   * @ret GraphNode
   */
  const std::shared_ptr<GraphNode> &getSortedNode(unsigned int ith) const;

  /**
   * @brief getter of GraphNode with node name
   * @param[in] node name
   * @retval GraphNode
   */
  const std::shared_ptr<GraphNode> &getNode(const std::string &name) const;

  /**
   * @brief getter all the node nodes in the model
   * @retval node nodes
   * @note these node nodes will be in sorted order if the model is compiled,
   * otherwise the order is the order of addition of node nodes in the model.
   * TODO: deprecate this
   */
  std::vector<std::shared_ptr<GraphNode>> getNodes() const;

  /**
   * @brief     join passed graph into the existing graph model
   * @param[in] graph graph to be added/to extend
   * @param[in] prefix prefix added to names of nodes from this graph
   * @note It is assumed that this model is valid by itself
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   *
   * @todo rename to addnodes
   */
  void extendGraph(std::vector<std::shared_ptr<GraphNode>> graph,
                   std::string &prefix);

  /**
   * @brief     getter of ordered graph
   * @retval    ordered GraphNode list
   * TODO: deprecate this
   */
  const std::vector<std::shared_ptr<GraphNode>> &getSorted() const;

  /**
   * @brief     getter of ordered graph
   * @retval    ordered GraphNode list
   * TODO: deprecate this
   */
  std::vector<std::shared_ptr<GraphNode>> &getSorted();

  /**
   * @brief     get begin iterator for the forwarding
   * @retval    const iterator marking the begin of forwarding
   */
  template <
    typename T = GraphNode,
    std::enable_if_t<std::is_base_of<GraphNode, T>::value, T> * = nullptr>
  inline graph_iterator<T const> cbegin() const {
    return graph_iterator<T const>(&(*Sorted.cbegin()));
  }

  /**
   * @brief     get end iterator for the forwarding
   * @retval    const iterator marking the emd of forwarding
   */
  template <
    typename T = GraphNode,
    std::enable_if_t<std::is_base_of<GraphNode, T>::value, T> * = nullptr>
  inline graph_iterator<T const> cend() const {
    return graph_iterator<T const>(&(*Sorted.cend()));
  }

  /**
   * @brief     get begin iterator for the backwarding
   * @retval    const reverse iterator marking the begin of backwarding
   */
  template <
    typename T = GraphNode,
    std::enable_if_t<std::is_base_of<GraphNode, T>::value, T> * = nullptr>
  inline graph_reverse_iterator<T const> crbegin() const {
    return graph_reverse_iterator<T const>(cend<T>());
  }

  /**
   * @brief     get end iterator for the backwarding
   * @retval    const reverse iterator marking the end of backwarding
   */
  template <
    typename T = GraphNode,
    std::enable_if_t<std::is_base_of<GraphNode, T>::value, T> * = nullptr>
  inline graph_reverse_iterator<T const> crend() const {
    return graph_reverse_iterator<T const>(cbegin<T>());
  }

  /**
   * @brief Sorting and Define order to calculate : Depth First Search
   */
  void topologicalSort();

  /**
   * @brief     Copy the graph
   * @param[in] from Graph Object to copy
   * @retval    Graph Object copyed
   */
  GraphCore &copy(GraphCore &from) {
    if (this != &from) {
      // FIXME: this assumes elements already in nodes/adj, solve that
      // for (unsigned int i = 0; i < adj.size(); i++)
      //   adj[i].front()->getObject()->copy(from.adj[i].front()->getObject());
    }
    return *this;
  }

  /**
   * @brief add Edge between graph nodes
   * @param[in] ith Node index : From
   * @param[in] node GraphNode object to be added : To
   */
  void addEdge(unsigned int ith, const std::shared_ptr<GraphNode> &node);

  /**
   * @brief     make connection between nodes
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int connectGraph();

  /**
   * @brief     remove all the edges from the graph
   *
   */
  void removeEdges();

  /**
   * @brief     Ensure that node has a name.
   * @param[in] node GraphNode whose name is to be ensured to be valid
   * @param[in] prefix Prefix to be attached to the node name
   * @param[in] postfix Postfix to be attached to the node name
   * @param[in] force_rename If the node must be forcefully rename
   * @details   Ensures that the node has a unique and a valid name. A valid
   * name pre-assigned to the node can be changed if force_rename is enabled.
   */
  void ensureName(GraphNode &node, const std::string &prefix = "",
                  const std::string &postfix = "", bool force_rename = false);

  void remove_last_node() {
    auto last_node = Sorted.back();
    Sorted.pop_back();
    adj.erase(adj.begin() + last_node->getIndex());

    /**
     * Remove all the connections for the current lasy layer as it will now only
     */
    last_node = Sorted.back();
    adj[last_node->getIndex()].resize(1);
  }

  void addLossToSorted() { Sorted.push_back(adj.back().front()); }

  bool verifyNode(const std::string &name) {
    if (node_names.find(name) == node_names.end())
      return false;
    return true;
  }

private:
  std::vector<std::list<std::shared_ptr<GraphNode>>>
    adj; /**< adjacency list for graph */
  std::vector<std::shared_ptr<GraphNode>> node_list; /**< Ordered Node List  */

  std::vector<std::shared_ptr<GraphNode>> Sorted; /**< Ordered Node List  */
  bool sorted; /** if the node_list is sorted */

  std::set<std::string>
    node_names;       /**< Set containing all the names of nodes in the model */
  int def_name_count; /**< Count assigned to node names declared by default */

  /**
   * @brief     topological sort
   * @param[in] ith index of GraphNode
   * @param[in] visited temp list
   * @param[in] stack for Node list to visit.
   */
  void topologicalSortUtil(unsigned int ith, std::vector<bool> &visited,
                           std::stack<std::shared_ptr<GraphNode>> &Stack);

  /**
   * @brief     make connection for the given node idx
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  void connectGraph(unsigned int adj_idx);

  /**
   * @brief     set output connections for all the nodes
   */
  void setOutputLayers();

  /**
   * @brief Add given GraphNode to the Graph
   * @param[in] node shared_ptr of GraphNode
   */
  void addGraphNode(std::shared_ptr<GraphNode> node);
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __NETWORK_GRAPH_H__ */
