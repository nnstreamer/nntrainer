// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file tflite_opnode.h
 * @date 28 April 2021
 * @brief contains tflite opnode which has information to convert to tflite file
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
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
   * @brief finalize tf op node will be transfored to required variables
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

private:
  Variables inputs;  /**< input variables */
  Variables outputs; /**< output variables */
  Variables weights; /**< weight variables */

  TransformFn weight_transform; /**< weight transforms */

  bool is_input;  /**< true if given input is input; */
  bool is_output; /**< true if given output is output; */

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
