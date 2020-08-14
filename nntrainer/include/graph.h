// SPDX-License-Identifier: Apache-2.0-only
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file	graph.h
 * @date	14 August 2020
 * @brief	This is Graph Class of Neural Network
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#ifndef __GRAPH_H__
#define __GRAPH_H__
#ifdef __cplusplus

#include <tensor_op.h>
#include <vector>

namespace nntrainer {

/**
 * @class   Graph
 * @brief   Graph of Tensors and Tensor Operations
 */
class Graph {
public:
  Graph() {}

  ~Graph() {}

  /**
   * @brief Register the tensor op in the graph along with its input/output
   * tensors
   *
   * @param op Operation to be registered
   */
  void registerTensorOp(TensorOp op) {}

  /**
   * @brief Get the all the tensors object
   *
   * @return std::vector<Tensor>& List of all the tensor objects for the all
   * registered ops
   */
  std::vector<Tensor> &getTensors() { return all_tensors; }

  /**
   * @brief Get the all the input tensors object
   *
   * @return std::vector<Tensor>& List of all the input tensor objects for the
   * graphs
   * @details These are the tensors to be updated every iteration when training
   * the model
   */
  std::vector<Tensor> getInputs() { return {}; }

  /**
   * @brief Get the all the output tensors object
   *
   * @return std::vector<Tensor>& List of all the output tensor objects for the
   * graphs
   * @details These are the tensors where the output of the model for every
   * iteration will be available when training the model
   */
  std::vector<Tensor> getOutputs() { return {}; }

  /**
   * @brief Add this tensor to the list of output tensors
   *
   * @param tensor Tensor to be added as output tensor
   */
  void addToOutputs(Tensor tensor) {}

  /**
   * @brief Set the optimizer to be used for this graph optimization
   *
   * @param opt Optimizer to be set
   */
  void setOptimizer(Optimizer opt) {}

  /**
   * @brief Run the given tensorOp with the current inputs and its dependencies
   *
   * @param op Operation to be run
   */
  void compute(TensorOp op) {}

  /**
   * @brief Run the given tensor with the current inputs
   *
   * @param tensor Tensor to be run
   * @note This tensor will be input/output tensor of an operation. This tensor
   * cannot be a gradient tensor.
   */
  void compute(Tensor tensor) {}

  /**
   * @brief Compute gradient for the given tensorOp with the current
   * inputs/outputs and its dependencies
   *
   * @param op Operation for which gradient is to be computed
   * @note For tensors/tensor operations dependent on this tensor operation
   * (behind this operation in this graph), gradient will not be computed
   */
  void computeGrad(TensorOp op) {}

  /**
   * @brief Compute gradient for the given tensor with the current
   * inputs/outputs and its dependencies
   *
   * @param tensor Tensor for which gradient is to be computed
   * @note For tensors/tensor operations dependent on this tensor (behind this
   * tensor in this graph), gradient will not be computed
   * @note This tensor will be input/output tensor of an operation. This tensor
   * cannot be a gradient tensor.
   */
  void computeGrad(Tensor tensor) {}

  /**
   * @brief Apply gradient for the given tensor
   *
   * @param tensor Tensor for which gradient is to be applied
   * @note This tensor will be input/output tensor of an operation. This tensor
   * cannot be a gradient tensor.
   */
  void applyGrad(Tensor tensor) {}

  /**
   * @brief Train the given tensorOp with the current inputs/outputs and its
   * dependencies
   *
   * @param op Operation to be trained
   * @note tensors/tensor operations dependent on this tensor operation (behind
   * this operation in this graph) will not be trained
   */
  void train(TensorOp op) {}

  /**
   * @brief Train the given tensor with the current inputs
   *
   * @param tensor Tensor to be trained
   * @note tensors/tensor operations dependent on this tensor (behind this
   * tensor in this graph will not be trained
   * @note This tensor will be input/output tensor of an operation. This tensor
   * cannot be a gradient tensor.
   */
  void train(Tensor tensor) {}

  /**
   * @brief     initialize the tensor operations and tensors inside it
   */
  void initialize() {}

protected:
  std::vector<TensorOp>
    all_ops; /** NOTE: this will change to graph than a list */
  std::vector<TensorParam> all_tensor_params;
  std::vector<Tensor> all_tensors;
  std::vector<Tensor> all_trainables;
};
} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __GRAPH_H__ */
