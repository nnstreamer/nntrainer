// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 DongHak Park <donghak.park@samsung.com>
 *
 * @file   flatbuffer_opnode.h
 * @date   10 February 2023
 * @brief  NNTrainer flatbuffer opnode
 * @see	   https://github.com/nnstreamer/nntrainer
 * @author Donghak Park <donghak.park@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#ifndef __FLATBUFFER_OPNODE_H__
#define __FLATBUFFER_OPNODE_H__

#include <functional>
#include <utility>
#include <vector>

#include <nntrainer_schema_generated.h>
#include <tensor.h>

namespace nntrainer {

class LayerNode;
class RunLayerContext;

/**
 * @brief FlatBufferOpNode class
 *
 */
class FlatBufferOpNode {
public:
  using Variables = std::vector<const Tensor *>;

  /**
   * @brief Construct a new Flat Buffer Op Node object
   *
   */
  FlatBufferOpNode();

  /**
   * @brief Set the Layer Node object
   *
   * @param layer layer node
   */
  void setLayerNode(const LayerNode &layer);

  /**
   * @brief Set the Op Type object
   *
   * @param op_type_
   */
  void setOpType(nntr::BuiltinOperator op_type_) { op_type = op_type_; }

  /**
   * @brief Set the Builtin Options object
   *
   * @param builtin_option_type_ builtin option type
   * @param builtin_ops_ flatbuffer offset of builtin ops
   */
  void setBuiltinOptions(nntr::BuiltinOptions builtin_option_type_,
                         const flatbuffers::Offset<void> &builtin_ops_);

  /**
   * @brief Get the Inputs object
   *
   * @return Variables& inputs
   */
  Variables &getInputs() { return inputs; }

  /**
   * @brief Get the Inputs object
   *
   * @return const Variables& inputs
   */
  const Variables &getInputs() const { return inputs; }

  /**
   * @brief Get the Weights object
   *
   * @return Variables& weights
   */
  Variables &getWeights() { return weights; }

  /**
   * @brief Get the Weights object
   *
   * @return const Variables& weights
   */
  const Variables &getWeights() const { return weights; }

  /**
   * @brief Get the Outputs object
   *
   * @return Variables& outputs
   */
  Variables &getOutputs() { return outputs; }

  /**
   * @brief Get the Outputs object
   *
   * @return const Variables& outputs
   */
  const Variables &getOutputs() const { return outputs; }

  /**
   * @brief check if the node is model input
   *
   * @return true if op node is model input
   * @return false  if op node is not model input
   */
  bool isInputNode() const { return is_input; }

  /**
   * @brief check if the node is model output
   *
   * @return true if op node is model output
   * @return false if op node is not model output
   */
  bool isOutputNode() const { return is_output; }

  /**
   * @brief check if the node is virtual node
   *
   * @return true if this op node is virtual node
   * @return false if this op node is not virtual node
   */
  bool isVirtualNode() const { return is_virtual; }

  /**
   * @brief Get the Op Type object
   *
   * @return const nntr::BuiltinOperator
   */
  const nntr::BuiltinOperator getOpType() const { return op_type; }

  /**
   * @brief Get the Option Type object
   *
   * @return const nntr::BuiltinOptions
   */
  const nntr::BuiltinOptions getOptionType() const {
    return builtin_option_type;
  }

  /**
   * @brief Get the Builtin Ops object
   * @param f Flatbuffer builder
   *
   * @return flatbuffers::Offset<void>
   */
  flatbuffers::Offset<void> getBuiltinOps() const;

  /**
   * @brief Get the Input Nodes object
   *
   * @return const std::vector<FlatBufferOpNode *> &input_nodes
   */
  const std::vector<FlatBufferOpNode *> &getInputNodes() const {
    return input_nodes;
  }

  /**
   * @brief Set arity
   *
   * @param value value to set
   */
  void arity(size_t value) { input_nodes.resize(value); }

  /**
   * @brief Get arity
   *
   * @return const unsigned input_nodes size
   */
  const unsigned arity() const { return input_nodes.size(); }

  /**
   * @brief Set the Arg object
   *
   * @param index argument index to set
   * @param node the node to be argument
   */
  void setArg(size_t index, FlatBufferOpNode *node) {
    input_nodes.at(index) = node;
  }

  /**
   * @brief Get the Arg object
   *
   * @param index argument index to get
   * @return FlatBufferOpNode *input_nodes.at(index)
   */
  FlatBufferOpNode *arg(size_t index) const { return input_nodes.at(index); }

private:
  Variables inputs;                            /**< input variables */
  Variables outputs;                           /**< output variables */
  Variables weights;                           /**< weight variables */
  std::vector<FlatBufferOpNode *> input_nodes; /**< input nodes */

  bool is_input;   /**< true if given input is model input */
  bool is_output;  /**< true if given output is model output */
  bool is_virtual; /**< true if given node is virtual */

  nntr::BuiltinOperator op_type;            /**< op type */
  nntr::BuiltinOptions builtin_option_type; /**< builtin option type */
  flatbuffers::Offset<void> builtin_ops;    /**< builtin ops */
};

} // namespace nntrainer

#endif // __FLATBUFFER_OPNODE_H__
