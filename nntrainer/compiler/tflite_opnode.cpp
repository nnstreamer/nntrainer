// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file tflite_opnode.cpp
 * @date 28 April 2021
 * @brief contains tflite opnode which has information to convert to tflite file
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug No known bugs except for NYI items
 */

#include <tflite_opnode.h>

#include <fc_layer.h>

namespace nntrainer {

void TfOpNode::setInOut(const Layer &layer) {
  auto &in = layer.getInputLayers();
  is_input = std::find(in.begin(), in.end(), "__data__") != in.end();

  auto &out = layer.getOutputLayers();
  is_output = std::find(out.begin(), out.end(), "__exit__") != out.end();
}

void TfOpNode::setInputs(
  const std::vector<std::shared_ptr<Var_Grad>> &inputs_) {

  inputs.reserve(inputs_.size());
  std::transform(inputs_.begin(), inputs_.end(), std::back_inserter(inputs),
                 [](const auto &data) { return data.get(); });
}

void TfOpNode::setOutputs(
  const std::vector<std::shared_ptr<Var_Grad>> &outputs_) {
  outputs.reserve(outputs_.size());
  std::transform(outputs_.begin(), outputs_.end(), std::back_inserter(outputs),
                 [](const auto &data) { return data.get(); });
}

void TfOpNode::setWeights(const std::vector<Weight> &weights_) {
  weights.reserve(weights_.size());
  std::transform(weights_.begin(), weights_.end(), std::back_inserter(weights),
                 [](const auto &data) { return &data; });
}

void TfOpNode::setBuiltinOptions(
  tflite::BuiltinOptions builtin_option_type_,
  const flatbuffers::Offset<void> &builtin_ops_) {
  builtin_ops = builtin_ops_;
  builtin_option_type = builtin_option_type_;
}

} // namespace nntrainer
