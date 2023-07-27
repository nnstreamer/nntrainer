// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file tflite_opnode.cpp
 * @date 28 April 2021
 * @brief contains tflite opnode which has information to convert to tflite file
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @author Donghak Park <donghak.park@samsung.com>
 * @bug No known bugs except for NYI items
 */

#include <tflite_opnode.h>

#include <layer_context.h>
#include <layer_node.h>
#include <memory.h>
namespace nntrainer {

TfOpNode::TfOpNode() :
  inputs(),
  outputs(),
  weights(),
  weight_transform(nullptr),
  is_input(false),
  is_output(false),
  is_virtual(false),
  is_trainable(true),
  is_to_be_removed(false),
  need_reorder_weight(false),
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
   *  2. without loss layer but last node has loss layer output connection
   *
   *  Loss layer of the first case is removed by `LossRealizer` and the previous
   *layer of the loss layer is set as the output node. And, the below logic is
   *for the second case.
   **/
  /// assume that loss layers have single output
  if (layer.getNumOutputConnections() == 1) {
    for (auto &loss : loss_type) {
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
   * basically, outputs are used for tflite tensors
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

  if (context.getNumWeights() == 0) {
    is_trainable = false;
  }
}

void TfOpNode::setWeightTransformFn(TransformFn fn) { weight_transform = fn; }

void TfOpNode::setInputTransformFn(TransformFn fn) { input_transform = fn; }

void TfOpNode::setWeights(Variables weights_, bool weight_transpose) {
  unsigned int cnt = 0;

  for (auto &w : weights_) {
    const unsigned int unit = w->batch();
    const unsigned int channel = w->channel();
    const unsigned int height = w->height();
    const unsigned int width = w->width();
    auto weight_data = weights.at(cnt)->getData();

    auto *ptr = const_cast<float *>(weight_data);
    memcpy(&ptr[0], &w->getData()[0],
           sizeof(float) * (unit * channel * height * width));
    cnt++;
  }

  auto weight_transform_fn = [](std::vector<const Tensor *> &weights) {
    std::vector<Tensor> new_weights;
    new_weights.reserve(weights.size());
    new_weights.push_back(weights[0]->transpose("2:1:0"));
    return new_weights;
  };

  auto transform_if = [this](TransformFn &fn, Variables &v) {
    if (fn) {
      auto result = fn(v);
      v.resize(result.size());
      node_owned_variable.insert(node_owned_variable.end(), result.begin(),
                                 result.end());
      std::transform(node_owned_variable.end() - result.size(),
                     node_owned_variable.end(), v.begin(),
                     [](Tensor &t) { return &t; });
    }
  };

  if (weight_transpose == true) {
    setWeightTransformFn(weight_transform_fn);
    transform_if(weight_transform, weights);
  }
}

void TfOpNode::weightReorder(unsigned int node_count) {

  if (need_reorder_weight == true) {

    auto previous_input_shape = input_nodes[0]->getInputs()[0];

    const unsigned int unit = outputs[0]->height();
    const unsigned int channel = previous_input_shape->channel();
    const unsigned int height = previous_input_shape->height();
    const unsigned int width = previous_input_shape->width();

    auto weight_data = weights[0]->getData();
    auto *ptr = const_cast<float *>(weight_data);

    std::vector<float> old_value_list(unit * channel * height * width);
    memcpy(&old_value_list[0], &ptr[0],
           sizeof(float) * (unit * channel * height * width));

    for (unsigned int h = 0; h < height; h++) {
      for (unsigned int w = 0; w < width; w++) {
        for (unsigned int c = 0; c < channel; c++) {

          unsigned int now_position = h * (width * channel) + w * channel + c;
          unsigned int next_position = c * (height * width) + h * width + w;

          memcpy(&ptr[now_position * unit],
                 &old_value_list[next_position * unit], sizeof(float) * unit);
        }
      }
    }
  }

  auto weight_transform_fn = [](std::vector<const Tensor *> &weights) {
    std::vector<Tensor> new_weights;
    new_weights.reserve(weights.size());
    new_weights.push_back(weights[0]->transpose("0:2:1"));
    new_weights.push_back(*weights[1]);
    return new_weights;
  };

  setWeightTransformFn(weight_transform_fn);

  auto transform_if = [this](TransformFn &fn, Variables &v) {
    if (fn) {
      auto result = fn(v);
      v.resize(result.size());
      node_owned_variable.insert(node_owned_variable.end(), result.begin(),
                                 result.end());
      std::transform(node_owned_variable.end() - result.size(),
                     node_owned_variable.end(), v.begin(),
                     [](Tensor &t) { return &t; });
    }
  };

  transform_if(weight_transform, weights);
}

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
      // nullptr && result.size() != v.size(), std::invalid_argument)
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
  case tflite::BuiltinOperator_MUL:

    return builtin_ops;
  default:
    throw std::runtime_error{"Unsupported operator"};
  }
}

void TfOpNode::setBuiltinOptions(
  tflite::BuiltinOptions builtin_option_type_,
  const flatbuffers::Offset<void> &builtin_ops_) {
  builtin_ops = builtin_ops_;
  builtin_option_type = builtin_option_type_;
}

} // namespace nntrainer
