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

struct LayerNode {
  std::shared_ptr<Layer> layer;
  unsigned int index;
};

class NetworkGraph {

  friend class NeuralNetwork;

public:
  NetworkGraph() : num_node(0), def_name_count(0){};

  void addEdge(unsigned int ith, LayerNode node);

  void topologicalSort();

  void addLayerNode(std::shared_ptr<Layer>);

  unsigned int getNumNode() { return num_node; }

  void setNumNetBufferSize();

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

private:
  void topologicalSortUtil(unsigned int ith, bool visited[],
                           std::stack<LayerNode> &Stack);

  unsigned int num_node;
  std::vector<std::list<LayerNode>> adj;
  std::vector<LayerNode> Sorted;
  std::set<std::string> layer_names;
  std::vector<std::shared_ptr<NetBuffers>> netBuffers;
  int def_name_count;
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __NETWORK_GRAPH_H__ */
