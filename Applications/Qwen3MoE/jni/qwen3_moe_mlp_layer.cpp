// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file qwen3_moe_mlp_layer.cpp
 * @date 09 January 2025
 * @brief Qwen3 MoE MLP Layer Implementation
 * @see https://github.com/nnstreamer/nntrainer
 * @author Samsung Electronics
 * @bug No known bugs except for NYI items
 */

#include "qwen3_moe_mlp_layer.h"

#include <cmath>
#include <stdexcept>

#include <layer_context.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <util_func.h>
#include <initializer.h>
#include <blas_interface.h>

namespace nntrainer {

Qwen3MoeMlpLayer::Qwen3MoeMlpLayer() : LayerImpl() {}

void Qwen3MoeMlpLayer::setProperty(const std::vector<std::string> &values) {
  for (const auto &value : values) {
    if (value.find("hidden_size=") == 0) {
      mlp_props.hidden_size = std::stoul(value.substr(12));
    } else if (value.find("intermediate_size=") == 0) {
      mlp_props.intermediate_size = std::stoul(value.substr(18));
    } else {
      NNTR_THROW(std::invalid_argument) << "[Qwen3MoeMlpLayer] Unknown property: " << value;
    }
  }
}

void Qwen3MoeMlpLayer::finalize(InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "Qwen3MoeMlp layer takes only one input";

  // Get input dimensions
  auto const &input_dim = context.getInputDimensions()[0];
  unsigned int batch_size = input_dim.batch();
  unsigned int sequence_length = input_dim.height();
  unsigned int hidden_size = input_dim.width();

  // Validate hidden size matches
  NNTR_THROW_IF(hidden_size != mlp_props.hidden_size, std::invalid_argument)
    << "Input hidden size (" << hidden_size << ") doesn't match layer hidden size (" 
    << mlp_props.hidden_size << ")";

  // Gate projection weight: [hidden_size, intermediate_size]
  TensorDim gate_weight_dim({mlp_props.hidden_size, mlp_props.intermediate_size});
  context.requestWeight(gate_weight_dim, Initializer::XAVIER_UNIFORM, 
                       WeightRegularizer::NONE, 1.0f, true, "gate_weight");

  // Up projection weight: [hidden_size, intermediate_size]
  TensorDim up_weight_dim({mlp_props.hidden_size, mlp_props.intermediate_size});
  context.requestWeight(up_weight_dim, Initializer::XAVIER_UNIFORM,
                       WeightRegularizer::NONE, 1.0f, true, "up_weight");

  // Down projection weight: [intermediate_size, hidden_size]
  TensorDim down_weight_dim({mlp_props.intermediate_size, mlp_props.hidden_size});
  context.requestWeight(down_weight_dim, Initializer::XAVIER_UNIFORM,
                       WeightRegularizer::NONE, 1.0f, true, "down_weight");

  // Set output dimensions (same as input)
  context.setOutputDimensions({input_dim});
}

void Qwen3MoeMlpLayer::forwarding(RunLayerContext &context, bool training) {
  const Tensor &input = context.getInput(SINGLE_INOUT_IDX);
  const Tensor &gate_weight = context.getWeight(GATE_WEIGHT_IDX);
  const Tensor &up_weight = context.getWeight(UP_WEIGHT_IDX);
  const Tensor &down_weight = context.getWeight(DOWN_WEIGHT_IDX);
  Tensor &output = context.getOutput(SINGLE_INOUT_IDX);

  auto input_dim = input.getDim();
  unsigned int batch_size = input_dim.batch();
  unsigned int sequence_length = input_dim.height();

  // Create intermediate tensors
  TensorDim gate_dim({batch_size, sequence_length, mlp_props.intermediate_size});
  TensorDim up_dim({batch_size, sequence_length, mlp_props.intermediate_size});
  
  Tensor gate_output(gate_dim);
  Tensor up_output(up_dim);
  Tensor gate_activated(gate_dim);
  Tensor combined_output(up_dim);

  // Gate projection: input @ gate_weight
  input.dot(gate_weight, gate_output);
  
  // Up projection: input @ up_weight
  input.dot(up_weight, up_output);

  // Apply SiLU to gate output
  applySiLU(gate_output, gate_activated);

  // Element-wise multiplication: gate_activated * up_output
  gate_activated.multiply(up_output, combined_output);

  // Down projection: combined @ down_weight
  combined_output.dot(down_weight, output);
}

void Qwen3MoeMlpLayer::calcDerivative(RunLayerContext &context) {
  const Tensor &input = context.getInput(SINGLE_INOUT_IDX);
  const Tensor &gate_weight = context.getWeight(GATE_WEIGHT_IDX);
  const Tensor &up_weight = context.getWeight(UP_WEIGHT_IDX);
  const Tensor &down_weight = context.getWeight(DOWN_WEIGHT_IDX);
  
  const Tensor &incoming_derivative = context.getIncomingDerivative(SINGLE_INOUT_IDX);
  Tensor &outgoing_derivative = context.getOutgoingDerivative(SINGLE_INOUT_IDX);
  
  Tensor &gate_weight_grad = context.getWeightGrad(GATE_WEIGHT_IDX);
  Tensor &up_weight_grad = context.getWeightGrad(UP_WEIGHT_IDX);
  Tensor &down_weight_grad = context.getWeightGrad(DOWN_WEIGHT_IDX);

  auto input_dim = input.getDim();
  unsigned int batch_size = input_dim.batch();
  unsigned int sequence_length = input_dim.height();

  // Recreate forward pass intermediate results for gradient calculation
  TensorDim gate_dim({batch_size, sequence_length, mlp_props.intermediate_size});
  TensorDim up_dim({batch_size, sequence_length, mlp_props.intermediate_size});
  
  Tensor gate_output(gate_dim);
  Tensor up_output(up_dim);
  Tensor gate_activated(gate_dim);
  Tensor combined_output(up_dim);

  // Recreate forward pass
  input.dot(gate_weight, gate_output);
  input.dot(up_weight, up_output);
  applySiLU(gate_output, gate_activated);
  gate_activated.multiply(up_output, combined_output);

  // Backward pass
  // 1. Down projection gradients
  // down_weight_grad = combined_output^T @ incoming_derivative
  combined_output.transpose().dot(incoming_derivative, down_weight_grad);
  
  // combined_grad = incoming_derivative @ down_weight^T
  Tensor combined_grad(combined_output.getDim());
  incoming_derivative.dot(down_weight.transpose(), combined_grad);

  // 2. Element-wise multiplication gradients
  // gate_activated_grad = combined_grad * up_output
  // up_grad = combined_grad * gate_activated
  Tensor gate_activated_grad(gate_activated.getDim());
  Tensor up_grad(up_output.getDim());
  
  combined_grad.multiply(up_output, gate_activated_grad);
  combined_grad.multiply(gate_activated, up_grad);

  // 3. SiLU gradient for gate
  Tensor gate_grad(gate_output.getDim());
  applySiLUDerivative(gate_output, gate_grad);
  gate_grad.multiply_i(gate_activated_grad);

  // 4. Weight gradients
  // gate_weight_grad = input^T @ gate_grad
  input.transpose().dot(gate_grad, gate_weight_grad);
  
  // up_weight_grad = input^T @ up_grad
  input.transpose().dot(up_grad, up_weight_grad);

  // 5. Input gradient
  // outgoing_derivative = gate_grad @ gate_weight^T + up_grad @ up_weight^T
  Tensor gate_input_grad(input.getDim());
  Tensor up_input_grad(input.getDim());
  
  gate_grad.dot(gate_weight.transpose(), gate_input_grad);
  up_grad.dot(up_weight.transpose(), up_input_grad);
  
  gate_input_grad.add(up_input_grad, outgoing_derivative);
}

void Qwen3MoeMlpLayer::exportTo(Exporter &exporter,
                               const ml_train_format_e format) const {
  LayerImpl::exportTo(exporter, format);
}

std::unique_ptr<Layer> Qwen3MoeMlpLayer::clone() const {
  return std::make_unique<Qwen3MoeMlpLayer>(*this);
}

void Qwen3MoeMlpLayer::applySiLU(const Tensor &input, Tensor &output) const {
  input.apply_i([](float x) { return silu(x); }, output);
}

void Qwen3MoeMlpLayer::applySiLUDerivative(const Tensor &input, Tensor &output) const {
  input.apply_i([](float x) { return silu_derivative(x); }, output);
}

float Qwen3MoeMlpLayer::silu(float x) {
  return x * sigmoid(x);
}

float Qwen3MoeMlpLayer::silu_derivative(float x) {
  float sig = sigmoid(x);
  return sig * (1.0f + x * (1.0f - sig));
}

float Qwen3MoeMlpLayer::sigmoid(float x) {
  // Numerically stable sigmoid implementation
  if (x >= 0) {
    float exp_neg_x = std::exp(-x);
    return 1.0f / (1.0f + exp_neg_x);
  } else {
    float exp_x = std::exp(x);
    return exp_x / (1.0f + exp_x);
  }
}

} // namespace nntrainer