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
#include <layer_internal.h>
#include <list>
#include <loss_layer.h>
#include <memory>
#include <stack>
#include <vector>

namespace nntrainer {

/**
 * @brief     Graph Node Type
 */
struct LayerNode {
  std::shared_ptr<Layer> layer;
  unsigned int index;
};

/**
 * @class   NeuralNetwork Graph Class
 * @brief   NeuralNetwork Graph Class which manage layers
 */
class NetworkGraph {

  friend class NeuralNetwork;

public:
  /**
   * @brief     Constructor of NeuralNetwork Graph Class
   */
  NetworkGraph() : num_node(0), def_name_count(0){};

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
   * @brief getter of number of nodes
   * @param[out] number of nodes
   */
  unsigned int getNumNode() { return num_node; }

  /**
   * @brief initialize net_input and net_hidden of each layer
   *        according to num_input and num_output
   */
  void setNumNetBufferSize();

  /**
   * @brief getter of number of nodes
   * @param[out] number of nodes
   */
  LayerNode &getLayerNode(unsigned int ith);

  LayerNode &getSortedLayerNode(unsigned int ith);

  LayerNode &getLayerNode(const std::string &layer_name);

  LayerNode &getSortedLayerNode(const std::string &layer_name);

  void ensureName(std::shared_ptr<Layer> layer, const std::string &prefix,
                  bool force_rename);

  void setOutputLayers(std::vector<std::shared_ptr<Layer>> layers);

  int setGraphNode(std::vector<std::shared_ptr<Layer>> layers,
                   const LossType loss_type);

  int realizeMultiInputType(Layer &current);

  int realizeMultiOutputType(Layer &current,
                             std::vector<std::shared_ptr<Layer>> layers);

  int realizeActivationType(Layer &current,
                            std::vector<std::shared_ptr<Layer>> layers);

  int realizeFlattenType(Layer &current,
                         std::vector<std::shared_ptr<Layer>> layers);

  int addLossLayer(const LossType loss_type);

  int setEdge();

  void setBatchSize(unsigned int batch_size);

  sharedConstTensors forwarding(sharedConstTensors input);

  void backwarding(sharedConstTensors input, int iteration);

  std::vector<LayerNode> getSorted() { return Sorted; }

  std::vector<TensorDim> getOutputDimension();

  std::vector<TensorDim> getInputDimension();

private:
  void topologicalSortUtil(unsigned int ith, bool visited[],
                           std::stack<LayerNode> &Stack);

  unsigned int num_node;                 /**< Total Number of Graph Nodes */
  std::vector<std::list<LayerNode>> adj; /**< Graph Structure */
  std::vector<LayerNode> Sorted;         /**< Ordered Graph Node List  */
  std::set<std::string>
    layer_names; /**< Set containing all the names of layers in the model */
  std::vector<std::shared_ptr<NetBuffers>>
    netBuffers;       /**< List of Buffers used to calculate layer */
  int def_name_count; /**< Count assigned to layer names declared by default */
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __NETWORK_GRAPH_H__ */
