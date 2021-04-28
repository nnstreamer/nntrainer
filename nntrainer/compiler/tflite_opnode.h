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

#include <utility>
#include <vector>

#include <tf_schema_generated.h>

#include <layer_internal.h>
#include <var_grad.h>

namespace nntrainer {
/**
 * @brief tensorflow operational node representation. This class contains,
 * information to build operation flatbuffer
 *
 */
class TfOpNode {
public:
  using Variables = std::vector<const Var_Grad *>;

  /**
   * @brief Construct a new Tf object
   *
   */
  TfOpNode() = default;

  /**
   * @brief Construct a new Tf Op Node object from layer
   * @note this is a shortcut to skip if layer does not need to be devided or
   * fused
   * @param layer layer that is converted to TfOpNode
   */
  TfOpNode(const Layer &layer);

  /**
   * @brief Check and set if layer has model in/out
   *
   * @param layer layer to check
   */
  void setInOut(const Layer &layer);

  /**
   * @brief Set the Inputs object from layer
   *
   * @param inputs_ input to be inserted
   */
  void setInputs(const std::vector<std::shared_ptr<Var_Grad>> &inputs_);

  /**
   * @brief Set the Outputs object
   *
   * @param outputs_
   */
  void setOutputs(const std::vector<std::shared_ptr<Var_Grad>> &outputs_);

  /**
   * @brief Set the Weights object
   *
   * @param weights_ set weights from the object
   */
  void setWeights(const std::vector<Weight> &weights_);

  /**
   * @brief Set the Op Type object
   * @todo Considering number of alternatives to optimize this, for now it is
   * just workable.
   * 1. add and maintain global unordered map
   * 2. Save information in the appcontext later we can retrieve
   * 3. let type be an immutable property and let exporter handle this instead
   * of this method (preferrable)
   * @param type type to convert
   */
  void setOpType(const std::string &type);

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
   * @brief Get the Weights object
   *
   * @return Variables&
   */
  Variables &getWeights() { return weights; }

  /**
   * @brief Get the Weights object
   *
   * @return const Variables& outputs
   */
  const Variables &getWeights() const { return weights; }

  /**
   * @brief check if this op node is model input
   *
   * @return true if op node is model input
   * @return false if op node is not model input
   */
  bool isInputNode() const { return is_input; }

  /**
   * @brief check if this op node is model output
   *
   * @return true if op node is model output
   * @return false if op node is not model output
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
  /**
   * @brief Set the Op Type object
   *
   * @param op_type_ operation type
   */
  void setOpType(tflite::BuiltinOperator op_type_) { op_type = op_type_; }

  Variables inputs;  /**< input variables */
  Variables outputs; /**< output variables */
  Variables weights; /**< weight variables */

  bool is_input;  /**< true if given input is input; */
  bool is_output; /**< true if given output is output; */

  tflite::BuiltinOperator op_type;

  /// retrieve this from export_to
  flatbuffers::Offset<void> builtin_ops;
  tflite::BuiltinOptions builtin_option_type;
};

} // namespace nntrainer

#endif // __TFLITE_OPNODE_H__
