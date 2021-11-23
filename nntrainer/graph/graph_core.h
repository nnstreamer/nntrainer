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

#include <list>
#include <map>
#include <memory>
#include <stack>
#include <unordered_map>
#include <unordered_set>
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
  GraphCore() : sorted(false), def_name_count(0) {}

  /**
   * @brief     Destructor of Graph Core Class
   *
   */
  ~GraphCore() = default;

  /**
   * @brief Add the given node into Graph
   * @param[in] node shared_ptr of node
   */
  void addNode(std::shared_ptr<GraphNode> node, bool ensure_name = true);

  /**
   * @brief getter of number of nodes
   * @param[out] number of nodes
   */
  unsigned int size() const { return node_list.size(); }

  /**
   * @brief get if the graph is empty
   * @param[out] true if empty, else false
   */
  bool empty() const { return node_list.empty(); }

  /**
   * @brief     Swap function for the class
   */
  friend void swap(GraphCore &lhs, GraphCore &rhs) {
    using std::swap;

    swap(lhs.node_list, rhs.node_list);
    swap(lhs.node_map, rhs.node_map);
    swap(lhs.Sorted, rhs.Sorted);
    swap(lhs.node_names, rhs.node_names);
    swap(lhs.def_name_count, rhs.def_name_count);
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
   * @brief     get begin iterator for the forwarding
   * @retval    const iterator marking the begin of forwarding
   */
  template <
    typename T = GraphNode,
    std::enable_if_t<std::is_base_of<GraphNode, T>::value, T> * = nullptr>
  inline graph_const_iterator<T> cbegin() const {
    if (Sorted.empty())
      return graph_const_iterator<T>(&(*node_list.cbegin()));
    else
      return graph_const_iterator<T>(&(*Sorted.cbegin()));
  }

  /**
   * @brief     get end iterator for the forwarding
   * @retval    const iterator marking the emd of forwarding
   */
  template <
    typename T = GraphNode,
    std::enable_if_t<std::is_base_of<GraphNode, T>::value, T> * = nullptr>
  inline graph_const_iterator<T> cend() const {
    if (Sorted.empty())
      return graph_const_iterator<T>(&(*node_list.cend()));
    else
      return graph_const_iterator<T>(&(*Sorted.cend()));
  }

  /**
   * @brief     get begin iterator for the backwarding
   * @retval    const reverse iterator marking the begin of backwarding
   */
  template <
    typename T = GraphNode,
    std::enable_if_t<std::is_base_of<GraphNode, T>::value, T> * = nullptr>
  inline graph_const_reverse_iterator<T> crbegin() const {
    return graph_const_reverse_iterator<T>(cend<T>());
  }

  /**
   * @brief     get end iterator for the backwarding
   * @retval    const reverse iterator marking the end of backwarding
   */
  template <
    typename T = GraphNode,
    std::enable_if_t<std::is_base_of<GraphNode, T>::value, T> * = nullptr>
  inline graph_const_reverse_iterator<T> crend() const {
    return graph_const_reverse_iterator<T>(cbegin<T>());
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
    node_list.resize(from.node_list.size());
    if (this != &from) {
      // or (unsigned int i = 0; i < node_list.size(); i++)
      //  node_list[i]->copy(from.node_list[i]);
    }
    return *this;
  }

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

  /**
   * @brief   Replace graph node in node_list
   * @param   from Graph node to be replaced
   * @param   to Graph node to replace
   */
  void replaceNode(std::shared_ptr<GraphNode> from,
                   std::shared_ptr<GraphNode> to);

  /**
   * @brief   getter of graph input nodes with index number
   * @param   idx
   * @return  graph node of input node
   */
  const std::shared_ptr<GraphNode> &getInputNode(unsigned int idx) const {
    return input_list[idx];
  }

  /**
   * @brief   getter of number of input nodes
   * @return  number of input nodes
   */
  unsigned int getNumInputNodes() const { return input_list.size(); }

  /**
   * @brief   getter of graph output nodes with index number
   * @param   idx
   * @return  graph node of output node
   */
  const std::shared_ptr<GraphNode> &getOutputNode(unsigned int idx) const {
    return output_list[idx];
  }

  /**
   * @brief   getter of number of output nodes
   * @return  number of output nodes
   */
  unsigned int getNumOutputNodes() const { return output_list.size(); }

  /**
   * @brief       replace output node
   * @param idx   output node index to be replaced
   * @param node  graph node shared pointer to replace
   */
  void replaceOutputNode(unsigned int idx, std::shared_ptr<GraphNode> node) {
    output_list[idx] = node;
  }

  /**
   * @brief find which node is a input or output node in graph
   */
  void realizeInputOutputNode();

  /**
   * @brief     Verify if the node exists
   */
  inline bool verifyNode(const std::string &name) {
    if (node_names.find(name) == node_names.end())
      return false;
    return true;
  }

private:
  std::vector<std::shared_ptr<GraphNode>> input_list;
  std::vector<std::shared_ptr<GraphNode>> output_list;
  std::vector<std::shared_ptr<GraphNode>>
    node_list;                                    /**< Unordered Node List  */
  std::unordered_map<std::string, int> node_map;  /**< Unordered Node map  */
  std::vector<std::shared_ptr<GraphNode>> Sorted; /**< Ordered Node List  */
  bool sorted; /** if the node_list is sorted */

  std::unordered_set<std::string>
    node_names;       /**< Set containing all the names of nodes in the model */
  int def_name_count; /**< Count assigned to node names declared by default */

  /**
   * @brief     topological sort
   * @param[in] ith index of GraphNode
   * @param[in] visited temp list
   * @param[in] stack for Node list to visit.
   */
  void
  topologicalSortUtil(std::vector<std::list<std::shared_ptr<GraphNode>>> &adj,
                      unsigned int ith, std::vector<bool> &visited,
                      std::stack<std::shared_ptr<GraphNode>> &Stack);

  /**
   * @brief Add given GraphNode to the Graph
   * @param[in] node shared_ptr of GraphNode
   */
  void addGraphNode(std::shared_ptr<GraphNode> node);

  /**
   * @brief     make adjancency list for the current graph
   */
  void
  makeAdjacencyList(std::vector<std::list<std::shared_ptr<GraphNode>>> &adj);

  /**
   * @brief     Get index of the node with given name
   *
   * @param     name Name of the node
   * @return    internal index of the node
   */
  unsigned int getNodeIdx(const std::string &name);
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __NETWORK_GRAPH_H__ */
