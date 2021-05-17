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

#ifndef __NETWORK_GRAPH_H__
#define __NETWORK_GRAPH_H__
#ifdef __cplusplus

#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <stack>
#include <vector>

#include <graph_core.h>
#include <layer_internal.h>
#include <layer_node.h>
#include <loss_layer.h>

namespace nntrainer {

/**
 * @class   NeuralNetwork Graph Class
 * @brief   NeuralNetwork Graph Class which manage layers
 */
class NetworkGraph {

public:
  /**
   * @brief     Constructor of NeuralNetwork Graph Class
   */
  NetworkGraph() : graph(), skip_non_trainable_layers(0), compiled(false) {}

  /**
   * @brief     Compile the graph
   * @param[in] loss_type loss for the graph
   * returns ML_ERROR_NONE on success, error on failure
   */
  int compile(const LossType loss_type);

  /**
   * @brief Create new LayerNode and add into Graph
   * @param[in] layer shared_ptr of Layer
   */
  void addLayer(std::shared_ptr<LayerNode> layer);

  /**
   * @brief get current flat graph from the model before sorting
   * @note graph contains pointer to the actual nodes, which is not deeply
   * copied.
   * @retval current flat graph
   *
   * @todo remove getting unsorted layers from model loader, compile model
   * loader
   */
  std::vector<std::shared_ptr<LayerNode>>
  getUnsortedLayers(const std::string &input_layer,
                    const std::string &output_layer) const;

  /**
   * @brief getter of number of nodes
   * @param[out] number of nodes
   */
  unsigned int size() const { return graph.size(); }

  /**
   * @brief get if the graph is empty
   * @param[out] true if empty, else false
   */
  bool empty() const { return graph.empty(); }

  /**
   * @brief     Swap function for the class
   */
  friend void swap(NetworkGraph &lhs, NetworkGraph &rhs) {
    using std::swap;

    swap(lhs.graph, rhs.graph);
    swap(lhs.skip_non_trainable_layers, rhs.skip_non_trainable_layers);
  }

  /**
   * @brief     reset the graph
   */
  void reset() {

    graph.reset();
    skip_non_trainable_layers = 0;
  }

  /**
   * @brief getter of LayerNode with index number
   * @param[in] index
   * @ret LayerNode
   */
  std::shared_ptr<LayerNode> getLayerNode(unsigned int ith) {
    return std::static_pointer_cast<LayerNode>(graph.getNode(ith));
  }

  /**
   * @brief getter of Sorted LayerNode with index number
   * @param[in] index
   * @ret LayerNode
   */
  std::shared_ptr<LayerNode> getSortedLayerNode(unsigned int ith) {
    return std::static_pointer_cast<LayerNode>(graph.getSortedNode(ith));
  }

  /**
   * @brief getter of LayerNode with layer name
   * @param[in] layer name
   * @retval LayerNode
   */
  std::shared_ptr<LayerNode> getLayerNode(const std::string &layer_name) {
    return std::static_pointer_cast<LayerNode>(graph.getNode(layer_name));
  }

  /**
   * @brief getter of Layer with layer name
   * @param[in] layer name
   * @retval Layer
   */
  std::shared_ptr<Layer> getLayer(const std::string &layer_name) {
    return getLayerNode(layer_name)->getObject();
  }

  /**
   * @brief getter all the layer nodes in the model
   * @retval Layer nodes
   * @note these layer nodes will be in sorted order if the model is compiled,
   * otherwise the order is the order of addition of layer nodes in the model.
   */
  std::vector<std::shared_ptr<LayerNode>> getLayerNodes() const;

  /**
   * @brief     join passed graph into the existing graph model
   * @param[in] graph graph to be added/to extend
   * @param[in] prefix prefix added to names of layers from this graph
   * @note It is assumed that this model is valid by itself
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   *
   * @todo rename to addLayers
   */
  void extendGraph(std::vector<std::shared_ptr<LayerNode>> graph,
                   std::string &prefix);

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
  const std::vector<std::shared_ptr<LayerNode>> getSorted() const;

  /**
   * @brief     get begin iterator for the backwarding
   * @retval    const reverse iterator marking the begin of backwarding
   */
  graph_reverse_iterator<const LayerNode> getBackwardingBeginIter() {
    return graph.crbegin<LayerNode>();
  }

  /**
   * @brief     get end iterator for the backwarding
   * @retval    const reverse iterator marking the end of backwarding
   */
  graph_reverse_iterator<const LayerNode> getBackwardingEndIter() {
    graph_reverse_iterator<const LayerNode> iter = graph.crend<LayerNode>();
    iter -= skip_non_trainable_layers;
    return iter;
  }

  /**
   * @brief     getter of output dimension of graph
   * @retval    output tensor dim list
   */
  std::vector<TensorDim> getOutputDimension() const;

  /**
   * @brief     getter of input dimension of graph
   * @retval    input tensor dim list
   */
  std::vector<TensorDim> getInputDimension() const;

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
    graph.copy(from.graph);
    skip_non_trainable_layers = from.skip_non_trainable_layers;
    return *this;
  }

  /**
   * @brief initialize network graph, with given manager
   * @note this is taken from neuralnet, This might need some changes
   *
   * @param manager manager to allocate tensors
   */
  int initialize(std::shared_ptr<Manager> manager);

private:
  std::map<std::string, std::string> sub_in_out; /** This is map to identify
                   input and output layer name of subgraph */

  GraphCore graph; /** core graph object */
  unsigned int
    skip_non_trainable_layers; /**< denotes the number of non-trainable layers
                                  at the start of the graph */
  bool compiled;               /**< if the model graph is compiled */

  /**
   * @brief     topological sort
   * @param[in] ith index of LayerNode
   * @param[in] visited temp list
   * @param[in] stack for Node list to visit.
   */
  void topologicalSortUtil(unsigned int ith, std::vector<bool> &visited,
                           std::stack<std::shared_ptr<LayerNode>> &Stack);

  /**
   * @brief     check if graph is ready to compile.
   * @retval #ML_ERROR_NONE graph is ready to compile
   * @retval #ML_ERROR_INVALID_PARAMETER not ready to compile.
   */
  int isCompilable();

  /**
   * @brief     check if the compiled graph is of correct form.
   * @retval #ML_ERROR_NONE graph is ready to compile
   * @retval #ML_ERROR_INVALID_PARAMETER not ready to compile.
   */
  int checkCompiledGraph();

  /**
   * @brief     make connection between nodes
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int connectGraph();

  /**
   * @brief     make connection for the given node idx
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  void connectGraph(unsigned int adj_idx);

  /**
   * @brief     Realize Graph Nodes
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int realizeGraph();

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
   * @brief     set output connections for all the layers
   */
  void setOutputLayers();

  /**
   * @brief     set default input layer connections
   */
  void addDefaultInputLayers();

  /**
   * @brief     Ensure that layer has a name.
   * @param[in] layer Layer whose name is to be ensured to be valid
   * @param[in] prefix Prefix to be attached to the layer name
   * @param[in] postfix Postfix to be attached to the layer name
   * @param[in] force_rename If the layer must be forcefully rename
   * @details   Ensures that the layer has a unique and a valid name. A valid
   * name pre-assigned to the layer can be changed if force_rename is enabled.
   */
  void ensureName(std::shared_ptr<Layer> layer, const std::string &prefix = "",
                  const std::string &postfix = "", bool force_rename = false);

  /**
   * @brief Create new LayerNode and add into Graph
   * @param[in] layer shared_ptr of Layer
   */
  void addLayerNode(std::shared_ptr<Layer> layer);

  /**
   * @brief     update name of the the connections
   */
  void updateConnectionName(const std::string &from, const std::string &to);

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
