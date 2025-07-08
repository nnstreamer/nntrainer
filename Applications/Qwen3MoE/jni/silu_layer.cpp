// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file silu_layer.cpp
 * @date 09 January 2025
 * @brief SiLU (Sigmoid Linear Unit) Activation Layer Implementation
 * @see https://github.com/nnstreamer/nntrainer
 * @author Samsung Electronics
 * @bug No known bugs except for NYI items
 */

#include "silu_layer.h"

#include <cmath>
#include <stdexcept>

#include <layer_context.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <util_func.h>

namespace nntrainer {

SiLULayer::SiLULayer() : LayerImpl() {}

void SiLULayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, {});
  NNTR_THROW_IF(!remain_props.empty(), std::invalid_argument)
    << "[SiLULayer] Unknown properties: " << remain_props;
}

void SiLULayer::forwarding(RunLayerContext &context, bool training) {
  Tensor &input = context.getInput(SINGLE_INOUT_IDX);
  Tensor &output = context.getOutput(SINGLE_INOUT_IDX);

  // Apply SiLU activation element-wise
  input.apply_i([&](float x) { return silu(x); }, output);
}

void SiLULayer::calcDerivative(RunLayerContext &context) {
  const Tensor &derivative = context.getIncomingDerivative(SINGLE_INOUT_IDX);
  Tensor &input_derivative = context.getOutgoingDerivative(SINGLE_INOUT_IDX);
  const Tensor &input = context.getInput(SINGLE_INOUT_IDX);

  // Calculate derivative: dy/dx = silu'(x) * incoming_derivative
  input.apply_i([&](float x) { return silu_derivative(x); }, input_derivative);
  input_derivative.multiply_i(derivative);
}

void SiLULayer::exportTo(Exporter &exporter,
                         const ml_train_format_e format) const {
  LayerImpl::exportTo(exporter, format);
}

std::unique_ptr<Layer> SiLULayer::clone() const {
  return std::make_unique<SiLULayer>(*this);
}

float SiLULayer::silu(float x) {
  return x * sigmoid(x);
}

float SiLULayer::silu_derivative(float x) {
  float sig = sigmoid(x);
  return sig * (1.0f + x * (1.0f - sig));
}

float SiLULayer::sigmoid(float x) {
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