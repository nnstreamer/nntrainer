// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    network_graph.h
 * @date    19 Oct 2020
 * @see     https://github.com/nnstreamer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   This is Network Graph Class for Neural Network
 *
 */

#ifndef __NETWORK_GRAPH_H_
#define __NETWORK_GRAPH_H_
#ifdef __cplusplus

#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <stack>
#include <vector>

#include <layer_internal.h>

namespace nntrainer {

/**
 * @brief     Graph Node Type
 */
struct LayerNode {
  std::shared_ptr<Layer> layer;
  unsigned int index;
#ifdef PROFILE
  int event_key;
#endif
};

/**
 * @class   NeuralNetwork Graph Class
 * @brief   NeuralNetwork Graph Class which manage layers
 */
class NetworkGraph {

public:
  /**
   * @brief     Constructor of NeuralNetwork Graph Class
   */
  NetworkGraph() :
    num_node(0),
    def_name_count(0),
    skip_non_trainable_layers(0) {}

  /**
   * @brief add Edges between graph nodes
   * @param[in] ith Node index : From
   * @param[in] node LayerNode object to be added : To
   */
  void addEdge(unsigned int ith, LayerNode node);

  /**
   * @brief Sorting and Define order to calculate : Depth First Search
   */
  void topologicalSort();

  /**
   * @brief Create new LayerNode and add into Graph
   * @param[in] layer shared_ptr of Layer
   */
  void addLayerNode(std::shared_ptr<Layer> layer);

  /**
   * @brief Create new LayerNode and add into Graph
   * @param[in] layer shared_ptr of Layer
   */
  void addLayer(std::shared_ptr<Layer> layer);

  /**
   * @brief get current flat graph from the model
   * @note graph contains pointer to the actual nodes, which is not deeply
   * copied.
   * @retval current flat graph
   */
  std::vector<std::shared_ptr<Layer>> getGraph(const std::string &input_layer,
                                               const std::string &output_layer);

  /**
   * @brief getter of number of nodes
   * @param[out] number of nodes
   */
  unsigned int size() { return num_node; }

  /**
   * @brief get if the graph is empty
   * @param[out] true if empty, else false
   */
  bool empty() { return layers.empty(); }

  /**
   * @brief     Swap function for the class
   */
  friend void swap(NetworkGraph &lhs, NetworkGraph &rhs) {
    using std::swap;

    swap(lhs.num_node, rhs.num_node);
    swap(lhs.layers, rhs.layers);
    swap(lhs.adj, rhs.adj);
    swap(lhs.Sorted, rhs.Sorted);
    swap(lhs.layer_names, rhs.layer_names);
    swap(lhs.netBuffers, rhs.netBuffers);
    swap(lhs.def_name_count, rhs.def_name_count);
    swap(lhs.skip_non_trainable_layers, rhs.skip_non_trainable_layers);
  }

  /**
   * @brief     reset the graph
   */
  void reset() {
    layers.clear();
    adj.clear();
    Sorted.clear();
    layer_names.clear();
    netBuffers.clear();
    def_name_count = 0;
    skip_non_trainable_layers = 0;
  }

  /**
   * @brief getter of LayerNode with index number
   * @param[in] index
   * @ret LayerNode
   */
  LayerNode &getLayerNode(unsigned int ith);

  /**
   * @brief getter of Sorted LayerNode with index number
   * @param[in] index
   * @ret LayerNode
   */
  LayerNode &getSortedLayerNode(unsigned int ith);

  /**
   * @brief getter of LayerNode with layer name
   * @param[in] layer name
   * @retval LayerNode
   */
  LayerNode &getLayerNode(const std::string &layer_name);

  /**
   * @brief getter of Layer with layer name
   * @param[in] layer name
   * @retval Layer
   */
  std::shared_ptr<Layer> getLayer(const std::string &layer_name) {
    for (auto iter = layers.begin(); iter != layers.end(); ++iter) {
      if ((*iter)->getName() == layer_name) {
        return *iter;
      }
    }

    return nullptr;
  }

  /**
   * @brief getter of Layer with layer name
   * @param[in] layer name
   * @retval Layer
   */
  std::vector<std::shared_ptr<Layer>> &getLayers() { return layers; }

  /**
   * @brief     join passed graph into the existing graph model
   * @param[in] graph graph to be added/to extend
   * @param[in] prefix prefix added to names of layers from this graph
   * @note It is assumed that this model is valid by itself
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  void extendGraph(std::vector<std::shared_ptr<Layer>> graph,
                   std::string prefix);

  /**
   * @brief     Ensure that layer has a name
   */
  void ensureName(std::shared_ptr<Layer> layer, const std::string &prefix = "",
                  bool force_rename = false);

  /**
   * @brief     set Multi Output Layer
   */
  void setOutputLayers();

  /**
   * @brief     Build Graph Nodes
   * @param[in] loss_type loss type
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int setGraphNode(const LossType loss_type);

  /**
   * @brief     check and add Multi Input Layer : addition or concat Layer
   * @param[in] current layer
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int realizeMultiInputType(Layer &current);

  /**
   * @brief     check and add Multi output Layer : output Layer
   * @param[in] current layer
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int realizeMultiOutputType(Layer &current);

  /**
   * @brief     Realize act type to layer and insert it to layers
   * @param[in] current layer
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int realizeActivationType(Layer &current);

  /**
   * @brief     Realize flatten type to layer and insert it to layers
   * @param[in] current layer
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int realizeFlattenType(Layer &current);

  /**
   * @brief     adding loss layer at last position
   * @param[in] loss_type loss type
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int addLossLayer(const LossType loss_type);

  /**
   * @brief     make connection between nodes
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int setEdge();

  /**
   * @brief     set batch size
   * @param[in] batch size
   */
  void setBatchSize(unsigned int batch_size);

  /**
   * @brief     forwarding network graph
   * @param[in] input data
   * @param[in] training true if forwarding is on training
   * @retval output tensors
   */
  sharedConstTensors forwarding(bool training = false);

  /**
   * @brief     getter of ordered graph
   * @retval    ordered LayerNode list
   */
  const std::vector<LayerNode> &getSorted() { return Sorted; }

  /**
   * @brief     get begin iterator for the backwarding
   * @retval    const reverse iterator marking the begin of backwarding
   */
  std::vector<LayerNode>::const_reverse_iterator getBackwardingBeginIter() {
    return Sorted.crbegin();
  }

  /**
   * @brief     get end iterator for the backwarding
   * @retval    const reverse iterator marking the end of backwarding
   */
  std::vector<LayerNode>::const_reverse_iterator getBackwardingEndIter() {
    return Sorted.crend() - skip_non_trainable_layers;
  }

  /**
   * @brief     getter of output dimension of graph
   * @retval    output tensor dim list
   */
  std::vector<TensorDim> getOutputDimension();

  /**
   * @brief     getter of input dimension of graph
   * @retval    input tensor dim list
   */
  std::vector<TensorDim> getInputDimension();

  /**
   * @brief     Optimize the graph memory utilization for in-place operations
   */
  void inPlaceOptimize(Manager &manager);

  /**
   * @brief     Copy the graph
   * @param[in] from Graph Object to copy
   * @retval    Graph Object copyed
   */
  NetworkGraph &copy(NetworkGraph &from) {
    if (this != &from) {
      for (unsigned int i = 0; i < layers.size(); i++)
        layers[i]->copy(from.layers[i]);
    }
    return *this;
  }

  /**
   * @brief     check if graph is ready to compile.
   * @retval #ML_ERROR_NONE graph is ready to compile
   * @retval #ML_ERROR_INVALID_PARAMETER not ready to compile.
   */
  int isCompilable();

private:
  /**
   * @brief     topological sort
   * @param[in] ith index of LayerNode
   * @param[in] visited temp list
   * @param[in] stack for Node list to visit.
   */
  void topologicalSortUtil(unsigned int ith, std::vector<bool> &visited,
                           std::stack<LayerNode> &Stack);

  void updateNameInLayers(const std::string &cname, const std::string &name);

  std::map<std::string, std::string> sub_in_out; /** This is map to identify
                   input and output layer name of subgraph */
  std::vector<std::shared_ptr<Layer>>
    layers;                              /**< vector for store layer pointers */
  unsigned int num_node;                 /**< Total Number of Graph Nodes */
  std::vector<std::list<LayerNode>> adj; /**< Graph Structure */
  std::vector<LayerNode> Sorted;         /**< Ordered Graph Node List  */
  std::set<std::string>
    layer_names; /**< Set containing all the names of layers in the model */
  std::vector<std::shared_ptr<Var_Grad>>
    netBuffers;       /**< List of Buffers used to calculate layer */
  int def_name_count; /**< Count assigned to layer names declared by default */
  unsigned int
    skip_non_trainable_layers; /**< denotes the number of non-trainable layers
                                  at the start of the graph */
  /**
   * @brief Calculate the number of non-trainable layers at the start
   */
  void countNonTrainableLayersAtBegin();

  /**
   * @brief Update graph to remove redundant memory for in-place layer
   * @param layer_type Type of the layer which will work in-place
   * @note This optimization has no performance overhead.
   */
  void inPlaceOptimize(const std::string &layer_type, Manager &manager);
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __NETWORK_GRAPH_H__ */
