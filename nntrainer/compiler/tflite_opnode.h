// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file tflite_opnode.h
 * @date 28 April 2021
 * @brief contains tflite opnode which has information to convert to tflite file
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @author Donghak Park <donghak.park@samsung.com>
 * @bug No known bugs except for NYI items
 */

#ifndef __TFLITE_OPNODE_H__
#define __TFLITE_OPNODE_H__

#include <functional>
#include <utility>
#include <vector>

#include <tensor.h>
#include <tf_schema_generated.h>

namespace nntrainer {

class LayerNode;
class RunLayerContext;
/**
 * @brief tensorflow operational node representation. This class contains,
 * information to build operation flatbuffer
 *
 */
class TfOpNode {
public:
  using Variables = std::vector<const Tensor *>;

  using TransformFn =
    std::function<std::vector<Tensor>(std::vector<const Tensor *> &)>;

  /**
   * @brief Construct a new Tf object
   *
   */
  TfOpNode();

  /**
   * @brief finalize tf op node will be transformed to required variables
   * in this phase, weights are merged into inputs
   *
   */
  void finalize();

  /**
   * @brief Set common informations from layer node
   *
   * @param layer node layer node
   */
  void setLayerNode(const LayerNode &layer);

  /**
   * @brief Set the Weight Transform Fn object
   *
   * @param fn fn will be called before get
   */
  void setWeightTransformFn(TransformFn fn);

  /**
   * @brief Set the Input Transform Fn object
   *
   * @param fn fn will be called before get
   */
  void setInputTransformFn(TransformFn fn);

  /**
   * @brief Set the Op Type object
   *
   * @param op_type_ operation type
   */
  void setOpType(tflite::BuiltinOperator op_type_) { op_type = op_type_; }

  /**
   * @brief Set the Builtin Options object,
   * @note this can go private, export from a layer and fill this out
   *
   * @param builtin_option_type_ builtin option type
   * @param builtin_ops_ flatbuffer offset of builtin_ops
   */
  void setBuiltinOptions(tflite::BuiltinOptions builtin_option_type_,
                         const flatbuffers::Offset<void> &builtin_ops_);

  /**
   * @brief Set the Need Reorder Weight object
   *
   */
  void setNeedReorderWeight() { need_reorder_weight = true; }

  /**
   * @brief Set the To Be Removed object
   *
   */
  void setToBeRemoved(bool to_be_removed) { is_to_be_removed = to_be_removed; }

  /**
   * @brief Set the Trainable object
   *
   */
  void setTrainable(bool trainable) { is_trainable = trainable; }

  /**
   * @brief Set the Inputs object
   *
   * @param inputs_
   */
  void setInputs(const Variables &inputs_) { inputs = inputs_; }

  /**
   * @brief Set the Outputs object
   *
   * @param outputs_
   */
  void setOutputs(const Variables &outputs_) { outputs = outputs_; }

  /**
   * @brief Set the Weights object
   *
   * @param weights_
   */
  void setWeights(Variables weights_, bool weight_transpose = false);
  /**
   * @brief Replace the Weights object
   *
   * @param weights_
   */
  void replaceWeights(const Variables &weights_) { weights = weights_; }
  /**
   * @brief Set(Append) the Props object
   *
   * @param value
   */
  void AppendProps(const int &value) { props_vector.push_back(value); }

  /**
   * @brief Set(Append) the Additional Props object
   *
   * @param value
   */
  void AppendAdditionalProps(const float &value) {
    additional_props.push_back(value);
  }

  /**
   * @brief Reorder Weight in case of NCHW --> NHWC
   *
   * @param node_count
   */
  void weightReorder(unsigned int node_count);

  /**
   * @brief Get the Inputs object
   *
   * @return Variables& inputs
   */
  Variables &getInputs() { return inputs; }

  /**
   * @brief Get the weights object
   *
   * @return const Variables& weights
   */
  const Variables &getWeights() const { return weights; }

  /**
   * @brief Get the weights object
   *
   * @return Variables& weights
   */
  Variables &getWeights() { return weights; }

  /**
   * @brief Get the Inputs object
   *
   * @return const Variables& inputs
   */
  const Variables &getInputs() const { return inputs; }

  /**
   * @brief Get the Outputs object
   *
   * @return Variables&
   */
  Variables &getOutputs() { return outputs; }

  /**
   * @brief Get the Outputs object
   *
   * @return const Variables& outputs
   */
  const Variables &getOutputs() const { return outputs; }

  /**
   * @brief check if this op node is model input
   *
   * @retval true if op node is model input
   * @retval false if op node is not model input
   */
  bool isInputNode() const { return is_input; }

  /**
   * @brief check if this op node is model output
   *
   * @retval true if op node is model output
   * @retval false if op node is not model output
   */
  bool isOutputNode() const { return is_output; }

  /**
   * @brief check if this op node is virtual node
   *
   * virtual node is a node that will not be exported
   */
  bool isVirtualNode() const { return is_virtual; }

  /**
   * @brief check if this layer need to reorder
   *
   * @return true if weight need to reorder
   * @return false if reordering is not required
   */
  bool isNeedReorder() const { return need_reorder_weight; }

  /**
   * @brief check if this layer is trainable
   *
   * @return true if layer(OpNode) trainable
   * @return false if layer(OpNode) non-trainable
   */
  bool isTrainable() const { return is_trainable; }

  /**
   * @brief check if this layer is to be removed
   *
   * @return true
   * @return false
   */
  bool isToBeRemoved() const { return is_to_be_removed; }

  /**
   * @brief Get the Props Vector
   *
   * @return const std::vector<int> props_vector
   */
  std::vector<int> getProps() const { return props_vector; }

  /**
   * @brief Get the Additional Props Vector
   *
   * @return const std::vector<float> additional_props
   */
  std::vector<float> getAdditionalProps() const { return additional_props; }

  /**
   * @brief Get the Op Type object
   *
   * @return const tflite::BuiltinOperator
   */
  const tflite::BuiltinOperator getOpType() const { return op_type; }

  /**
   * @brief Get the Op Type object
   *
   * @return const tflite::BuiltinOperator
   */
  const tflite::BuiltinOptions getOptionType() const {
    return builtin_option_type;
  }

  /**
   * @brief Get the Op Options object
   * @param f Flatbuffer Builder
   * @retval const tflite::Offset<void>
   */
  flatbuffers::Offset<void> getBuiltinOps() const;

  /**
   * @brief Get input nodes
   *
   * @return const std::vector<TfOpNode *> &input_nodes
   */
  const std::vector<TfOpNode *> &getInputNodes() const { return input_nodes; }

  /**
   * @brief Set arity
   *
   * @param value value to set
   */
  void arity(size_t value) { input_nodes.resize(value); }

  /**
   * @brief Get arity
   *
   * @return const unsigned input_nodes.size()
   */
  const unsigned arity() const { return input_nodes.size(); }

  /**
   * @brief Set n-th argument of the node
   *
   * @param index argument index to set
   * @param node the node to be argument
   */
  void setArg(size_t index, TfOpNode *node) { input_nodes.at(index) = node; }

  /**
   * @brief Get n-th argument of the node
   *
   * @return TfOpNode *input_nodes.at(index)
   */
  TfOpNode *arg(size_t index) const { return input_nodes.at(index); }

private:
  Variables inputs;                    /**< input variables */
  Variables outputs;                   /**< output variables */
  Variables weights;                   /**< weight variables */
  std::vector<TfOpNode *> input_nodes; /**< input nodes */
  std::vector<int> props_vector;       /**< props vector */
  std::vector<float> additional_props; /**< additional props vector */

  /**
   * Q) Why do we need input transform?
   * A) To transform the nntrainer input data format(NCHW) to tflite
   *format(NHWC)
   **/
  TransformFn weight_transform; /**< weight transforms */
  TransformFn input_transform;  /**< input transforms */

  bool is_input;            /**< true if given input is input; */
  bool is_output;           /**< true if given output is output; */
  bool is_virtual;          /**< true if given node is virtual; */
  bool is_trainable;        /**< true if given node has weight and trainable */
  bool is_to_be_removed;    /**< true if given node is to be removed */
  bool need_reorder_weight; /**< true if given node need to reorder weight; */

  /// @todo change to shared_ptr or unique_ptr
  /// why? the addresses of existing tensors in the vector could become invalid
  /// due to memory reallocation
  std::vector<Tensor>
    node_owned_variable; /**< when node should be transformed it's own type, it
                          * needs to be owned by someone, so @a TfOpNode owns
                          * those orphaned tensors until the instance is
                          * destroyed */

  tflite::BuiltinOperator op_type;

  /// retrieve this from export_to
  flatbuffers::Offset<void> builtin_ops;
  tflite::BuiltinOptions builtin_option_type;
};

} // namespace nntrainer

#endif // __TFLITE_OPNODE_H__
