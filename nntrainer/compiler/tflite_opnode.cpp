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

#include <layer_context.h>
#include <layer_node.h>

namespace nntrainer {

TfOpNode::TfOpNode(){};

void TfOpNode::setLayerNode(const LayerNode &layer) {
  is_input = layer.getNumInputConnections() == 0;
  is_output = layer.getNumOutputConnections() == 0;

  auto &context = layer.getRunContext();
  auto create_variables = [](auto tensor_getter, unsigned size) {
    Variables v;
    v.reserve(size);
    for (unsigned i = 0; i < size; ++i) {
      v.push_back(tensor_getter(i));
    }
    return v;
  };

  inputs = create_variables(
    [&context](unsigned idx) { return &context.getInput(idx); },
    context.getNumInputs());
  outputs = create_variables(
    [&context](unsigned idx) { return &context.getOutput(idx); },
    context.getNumOutputs());
  weights = create_variables(
    [&context](unsigned idx) {
      auto &t = context.getWeight(idx);
      NNTR_THROW_IF(t.empty() || !t.isAllocated(), std::invalid_argument)
        << "every weight tensor must be allocated";
      return &t;
    },
    context.getNumWeights());
}

void TfOpNode::setWeightTransformFn(TransformFn fn) { weight_transform = fn; }

void TfOpNode::finalize() {
  auto transform_if = [this](TransformFn &fn, Variables &v) {
    if (fn) {
      auto result = fn(v);
      NNTR_THROW_IF(result.size() != v.size(), std::invalid_argument)
        << "result size must match with given variable size";
      node_owned_variable.insert(node_owned_variable.end(), result.begin(),
                                 result.end());
      std::transform(node_owned_variable.end() - result.size(),
                     node_owned_variable.end(), v.begin(),
                     [](Tensor &t) { return &t; });
    }
  };

  transform_if(weight_transform, weights);
}

void TfOpNode::setBuiltinOptions(
  tflite::BuiltinOptions builtin_option_type_,
  const flatbuffers::Offset<void> &builtin_ops_) {
  builtin_ops = builtin_ops_;
  builtin_option_type = builtin_option_type_;
}

} // namespace nntrainer
