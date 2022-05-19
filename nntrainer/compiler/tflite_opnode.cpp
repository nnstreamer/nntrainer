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

TfOpNode::TfOpNode() :
  inputs(),
  outputs(),
  weights(),
  weight_transform(nullptr),
  is_input(false),
  is_output(false),
  node_owned_variable(),
  /// @todo distinguish between uninitialized and ADD operator.
  op_type(tflite::BuiltinOperator_ADD),
  builtin_ops(),
  builtin_option_type(tflite::BuiltinOptions_NONE){};

void TfOpNode::setLayerNode(const LayerNode &layer) {
  is_input = layer.getNumInputConnections() == 0;
  is_output = layer.getNumOutputConnections() == 0;
  /// @todo support more loss layers
  static const std::set<std::string> loss_type = {"mse", "cross"};
  /** set to graph output node if output connection of the node includes loss
   *layer string
   *  @note this is workaround because it cannot be guaranteed that a loss layer
   *always has a loss type in its name.
   *
   *  There are two ways to pass `GraphRepresentation` parameters to `serialize`
   *method.
   *
   *  1. with loss layer at the end of the graph
   *  2. wihtout loss layer but last node has loss layer output connection
   *
   *  Loss layer of the first case is removed by `LossRealizer` and the previous
   *layer of the loss layer is set as the output node. And, the below logic is
   *for the second case.
   **/
  /// aussume that loss layers have single output
  if (layer.getNumOutputConnections() == 1) {
    for (auto loss : loss_type) {
      if (layer.getOutputConnections()[0].find(loss) != std::string::npos) {
        is_output = true;
      }
    }
  }
  /// @todo support more virtual nodes
  is_virtual = layer.getType() == "multiout";

  auto &context = layer.getRunContext();
  auto create_variables = [](auto tensor_getter, unsigned size) {
    Variables v;
    v.reserve(size);
    for (unsigned i = 0; i < size; ++i) {
      v.push_back(tensor_getter(i));
    }
    return v;
  };

  /**
   * Q1) Why convert from NCHW to NHWC?
   * A1) the tflite uses NHWC format; nntrainer uses NCHW
   *
   * Q2) Why are only output tensors reshaped?
   * A2) the tflite needs only one tensor between nodes. Therefore,
   *basically, outputs are used for tflite tensors
   **/
  auto create_variables_with_NCHW_to_NHWC = [](auto tensor_getter,
                                               unsigned size) {
    Variables v;
    v.reserve(size);
    for (unsigned i = 0; i < size; ++i) {
      Tensor *tensor = const_cast<Tensor *>(tensor_getter(i));
      tensor->reshape(TensorDim{tensor->batch(), tensor->height(),
                                tensor->width(), tensor->channel()});
      v.push_back(tensor);
    }
    return v;
  };

  inputs = create_variables(
    [&context](unsigned idx) { return &context.getInput(idx); },
    context.getNumInputs());
  outputs = create_variables_with_NCHW_to_NHWC(
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

void TfOpNode::setInputTransformFn(TransformFn fn) { input_transform = fn; }

void TfOpNode::finalize() {
  auto transform_if = [this](TransformFn &fn, Variables &v) {
    if (fn) {
      auto result = fn(v);
      v.resize(result.size());
      /// basically, result.size() == v.size() except InputLayer because a
      /// Transpose operator is added for converting nchw to nhwc
      /// @todo comment out below codes. TfOpNode needs to have LayerNode
      /// pointer
      // NNTR_THROW_IF(dynamic_cast<InputLayer>(layer_ptr->getLayer()) ==
      // nulltpr && result.size() != v.size(), std::invalid_argument)
      //   << "result size must match with given variable size";
      node_owned_variable.insert(node_owned_variable.end(), result.begin(),
                                 result.end());
      std::transform(node_owned_variable.end() - result.size(),
                     node_owned_variable.end(), v.begin(),
                     [](Tensor &t) { return &t; });
    }
  };

  transform_if(weight_transform, weights);
  transform_if(input_transform, inputs);
}

flatbuffers::Offset<void> TfOpNode::getBuiltinOps() const {
  switch (op_type) {
  case tflite::BuiltinOperator_ADD:
  case tflite::BuiltinOperator_AVERAGE_POOL_2D:
  case tflite::BuiltinOperator_CONV_2D:
  case tflite::BuiltinOperator_FULLY_CONNECTED:
  case tflite::BuiltinOperator_RELU:
  case tflite::BuiltinOperator_RESHAPE:
  case tflite::BuiltinOperator_SOFTMAX:
  case tflite::BuiltinOperator_TRANSPOSE:
    return builtin_ops;
  default:
    throw std::runtime_error{"Unsupproted operator"};
  }
}

void TfOpNode::setBuiltinOptions(
  tflite::BuiltinOptions builtin_option_type_,
  const flatbuffers::Offset<void> &builtin_ops_) {
  builtin_ops = builtin_ops_;
  builtin_option_type = builtin_option_type_;
}

} // namespace nntrainer
