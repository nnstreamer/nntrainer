// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   models_test_utils.h
 * @date   15 Oct 2020
 * @brief  Models test utils
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */
#ifndef __MODEL_TEST_UTILS_H__
#define __MODEL_TEST_UTILS_H__

#include <fstream>
#include <string>
#include <vector>

#include <layer_node.h>
#include <neuralnet.h>

using NodeType = std::shared_ptr<nntrainer::LayerNode>;
using FlatGraphType = nntrainer::NeuralNetwork::FlatGraphType;
using NetworkGraphType = nntrainer::NetworkGraph;

/**
 * @brief NodeWatcher has an operation Node. This class monitors data in and out
 * happening in the node.
 *
 */
class NodeWatcher {
public:
  /**
   * @brief Construct a new Node Watcher object
   *
   */
  NodeWatcher();

  /**
   * @brief Construct a new Node Watcher object
   *
   * @param node node to watch.
   */
  NodeWatcher(const NodeType &node);

  /**
   * @brief clones from expected weights to node->weights
   *
   */
  void readLayerWeight(std::ifstream &f);

  /**
   * @brief forward node with verifying inputs/weights/outputs
   *
   * @param in input tensor
   * @param iteration iteration
   * @return nntrainer::sharedConstTensor
   */
  void forward(int iteration, bool verify);

  /**
   * @brief backward pass of the node with verifying inputs/gradients/outputs
   *
   * @param deriv dervatives
   * @param iteration iteration
   * @param verify_deriv should verify the derivatives
   * @param verify_grad should verify the derivatives
   * @return nntrainer::sharedConstTensor
   */
  void backward(int iteration, bool verify_derv = true,
                bool verify_grad = true);

  /**
   * @brief verify weights of the current node
   *
   * @param error_msg error_msg to put when fail
   */
  void verifyWeight(const std::string &error_msg);

  /**
   * @brief verify gradient of the current node
   *
   * @param error_msg error_msg to put when fail
   */
  void verifyGrad(const std::string &error_msg);

  /**
   * @brief Get the Loss
   *
   * @return float loss
   */
  float getLoss() { return node->getLoss(); }

  /**
   * @brief read Node
   *
   * @param in input file stream
   */
  void read(std::ifstream &in);

  /**
   * @brief get Node type
   *
   * @return LayerType
   */
  std::string getType() { return node->getType(); }

  /**
   * @brief get Node type
   *
   * @return LayerType
   */
  std::string getName() { return node->getName(); }

  /**
   * @brief is loss type
   *
   * @return true if loss type node, else false
   */
  bool isLossType() { return node->requireLabel(); }

  /**
   * @brief support in-place operation
   *
   * @return true if support in-place else false
   */
  bool supportInPlace() { return node->supportInPlace(); }

  /**
   * @brief needs backwarding operation
   *
   * @return true if needs backwarding else false
   */
  bool needsBackwarding() { return node->needsBackwarding(); }

private:
  NodeType node;
  std::vector<nntrainer::Tensor> expected_output;
  std::vector<nntrainer::Tensor> expected_dx;
  std::vector<nntrainer::Weight> expected_weights;
};

/**
 * @brief GraphWatcher monitors and checks the graph operation like
 * forwarding & backwarding
 */
class GraphWatcher {
public:
  using WatchedFlatGraph = std::vector<NodeWatcher>;
  /**
   * @brief   GraphWatcher constructor
   */
  GraphWatcher(const std::string &config, const bool opt);

  /**
   * @brief Construct a new Graph Watcher object
   *
   * @param net created model
   * @param opt optimize or not
   */
  GraphWatcher(std::unique_ptr<nntrainer::NeuralNetwork> &&net, const bool opt);

  /**
   * @brief check forwarding & backwarding & inference throws or not
   * @param reference model file name
   * @param label_shape shape of label tensor
   * @param iterations tensor dimension of label
   */
  void compareFor(const std::string &reference,
                  const nntrainer::TensorDim &label_shape,
                  unsigned int iterations);

  /**
   * @brief   Validate the running of the graph without any errors
   * @param label_shape shape of label tensor
   */
  void validateFor(const nntrainer::TensorDim &label_shape);

  /**
   * @brief compare values given the golden file format v2
   *
   * @param reference reference file name
   */
  void compareFor_V2(const std::string &reference);

  /**
   * @brief   Validate the running of the graph without any errors
   */
  void validateFor_V2();

private:
  /**
   * @brief read and prepare the image & label data
   * @param f input file stream
   * @param label_dim tensor dimension of label
   * @return std::array<nntrainer::Tensor, 2> {input, label} tensors
   */
  std::array<nntrainer::Tensor, 2>
  prepareData(std::ifstream &f, const nntrainer::TensorDim &label_dim);

  /**
   * @brief initialize models, should be run only on any constructor
   *
   */
  void initialize();

  /**
   * @brief read Graph
   * @param f input file stream
   */
  void readIteration(std::ifstream &f);

  std::unique_ptr<nntrainer::NeuralNetwork> nn;
  WatchedFlatGraph nodes;
  std::vector<NodeWatcher> loss_nodes;
  std::vector<float> expected_losses;
  bool optimize;
};

#endif // __MODEL_TEST_UTILS_H__
