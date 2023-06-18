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

#include <flatbuffer_opnode.h>

#include <layer_context.h>
#include <layer_node.h>
#include <memory.h>

namespace nntrainer {

FlatBufferOpNode::FlatBufferOpNode() :
  is_input(false),
  is_output(false),
  is_virtual(false),
  layer_type(nntr::LayerTypes_FULLY_CONNECTED),
  layer_option_type(nntr::LayerOptions_NONE){};

void FlatBufferOpNode::setLayerNode(const LayerNode &layer) {
  is_input = (layer.getNumInputConnections() == 0);
  is_output = (layer.getNumOutputConnections() == 0);

  /// @todo Now support only mse, cross
  static const std::set<std::string> loss_type = {"mse", "cross"};

  if (layer.getNumOutputConnections() == 1) {
    for (auto loss : loss_type) {
      if (layer.getOutputConnections()[0].find(loss) != std::string::npos) {
        is_output = true;
      }
    }
  }

  is_virtual = (layer.getType() == "multiout");

  auto &context = layer.getRunContext();

  auto create_variables = [](auto tensor_getter, unsigned size) {
    Variables v;
    v.reserve(size);
    for (unsigned i = 0; i < size; ++i) {
      v.emplace_back(tensor_getter(i));
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
        << "Every weight must be allocated";
      return &t;
    },
    context.getNumWeights());
}

flatbuffers::Offset<void> FlatBufferOpNode::getLayerOps() const {
  switch (layer_type) {
  // Now support only fully connected Layer for test
  case nntr::LayerTypes_FULLY_CONNECTED:
    return layer_ops;
  default:
    throw std::runtime_error("Unsupported operator");
  }
}

void FlatBufferOpNode::setLayerOptions(
  nntr::LayerOptions layer_option_type_,
  const flatbuffers::Offset<void> &layer_ops_) {
  layer_ops = layer_ops_;
  layer_option_type = layer_option_type_;
}

} // namespace nntrainer
