// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file rms_norm_layer.cpp
 * @date 09 January 2025
 * @brief RMS Normalization Layer Implementation for Qwen3 MoE
 * @see https://github.com/nnstreamer/nntrainer
 * @author Samsung Electronics
 * @bug No known bugs except for NYI items
 */

#include "rms_norm_layer.h"

#include <cmath>
#include <stdexcept>

#include <layer_context.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <util_func.h>
#include <initializer.h>

namespace nntrainer {

RMSNormLayer::RMSNormLayer() : LayerImpl(), rms_norm_props(props::Epsilon()) {}

void RMSNormLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, rms_norm_props);
  NNTR_THROW_IF(!remain_props.empty(), std::invalid_argument)
    << "[RMSNormLayer] Unknown properties: " << remain_props;
}

void RMSNormLayer::finalize(InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "RMSNorm layer takes only one input";

  // Get input dimensions
  auto const &input_dim = context.getInputDimensions()[0];
  
  // Weight tensor (gamma) - same shape as the last dimension
  auto weight_dim = input_dim;
  // For RMSNorm, weight has the same shape as the input's last dimension
  // Typically this is the hidden_size dimension
  
  // Initialize weight tensor
  context.requestWeight(weight_dim, Initializer::ONES, WeightRegularizer::NONE, 
                       1.0f, true, "weight");

  // Set output dimensions (same as input)
  context.setOutputDimensions({input_dim});
}

void RMSNormLayer::forwarding(RunLayerContext &context, bool training) {
  const Tensor &input = context.getInput(SINGLE_INOUT_IDX);
  const Tensor &weight = context.getWeight(WEIGHT_IDX);
  Tensor &output = context.getOutput(SINGLE_INOUT_IDX);

  float eps = std::get<props::Epsilon>(rms_norm_props).get();
  
  applyRMSNorm(input, weight, eps, output);
}

void RMSNormLayer::calcDerivative(RunLayerContext &context) {
  const Tensor &input = context.getInput(SINGLE_INOUT_IDX);
  const Tensor &weight = context.getWeight(WEIGHT_IDX);
  const Tensor &incoming_derivative = context.getIncomingDerivative(SINGLE_INOUT_IDX);
  Tensor &outgoing_derivative = context.getOutgoingDerivative(SINGLE_INOUT_IDX);
  Tensor &weight_grad = context.getWeightGrad(WEIGHT_IDX);

  float eps = std::get<props::Epsilon>(rms_norm_props).get();

  // Calculate gradients
  auto input_shape = input.getDim();
  unsigned int batch_size = input_shape.batch();
  unsigned int feature_size = input_shape.getFeatureLen();

  for (unsigned int b = 0; b < batch_size; ++b) {
    // Calculate RMS for this batch
    float rms = calculateRMS(input.getBatchSlice(b, 1), eps);
    float inv_rms = 1.0f / rms;
    
    // Weight gradient: sum over batch of (input_normalized * incoming_grad)
    Tensor input_normalized = input.getBatchSlice(b, 1);
    input_normalized.multiply_i(inv_rms);
    
    Tensor grad_slice = incoming_derivative.getBatchSlice(b, 1);
    input_normalized.multiply_i(grad_slice);
    
    if (b == 0) {
      weight_grad.copy(input_normalized);
    } else {
      weight_grad.add_i(input_normalized);
    }

    // Input gradient calculation
    Tensor outgoing_slice = outgoing_derivative.getBatchSlice(b, 1);
    Tensor input_slice = input.getBatchSlice(b, 1);
    
    // Simplified gradient calculation for RMSNorm
    // d_input = weight * incoming_grad * (1/rms - input * input_mean / rms^3)
    float input_mean_square = 0.0f;
    const float *input_data = input_slice.getData();
    for (unsigned int i = 0; i < feature_size; ++i) {
      input_mean_square += input_data[i] * input_data[i];
    }
    input_mean_square /= feature_size;

    float *outgoing_data = outgoing_slice.getData();
    const float *incoming_data = grad_slice.getData();
    const float *weight_data = weight.getData();
    
    for (unsigned int i = 0; i < feature_size; ++i) {
      float normalized_input = input_data[i] * inv_rms;
      float grad_factor = inv_rms - (input_data[i] * input_mean_square) / (rms * rms * rms);
      outgoing_data[i] = weight_data[i] * incoming_data[i] * grad_factor;
    }
  }
}

void RMSNormLayer::exportTo(Exporter &exporter,
                           const ml_train_format_e format) const {
  LayerImpl::exportTo(exporter, format);
}

std::unique_ptr<Layer> RMSNormLayer::clone() const {
  return std::make_unique<RMSNormLayer>(*this);
}

float RMSNormLayer::calculateRMS(const Tensor &input, float eps) const {
  // Calculate Root Mean Square: sqrt(mean(x^2) + eps)
  const float *data = input.getData();
  unsigned int size = input.size();
  
  float sum_squares = 0.0f;
  for (unsigned int i = 0; i < size; ++i) {
    sum_squares += data[i] * data[i];
  }
  
  float mean_square = sum_squares / size;
  return std::sqrt(mean_square + eps);
}

void RMSNormLayer::applyRMSNorm(const Tensor &input, const Tensor &weight, 
                               float eps, Tensor &output) const {
  auto input_shape = input.getDim();
  unsigned int batch_size = input_shape.batch();
  
  for (unsigned int b = 0; b < batch_size; ++b) {
    // Get batch slice
    Tensor input_slice = input.getBatchSlice(b, 1);
    Tensor output_slice = output.getBatchSlice(b, 1);
    
    // Calculate RMS for this batch
    float rms = calculateRMS(input_slice, eps);
    
    // Normalize: x / rms * weight
    const float *input_data = input_slice.getData();
    const float *weight_data = weight.getData();
    float *output_data = output_slice.getData();
    
    unsigned int size = input_slice.size();
    float inv_rms = 1.0f / rms;
    
    for (unsigned int i = 0; i < size; ++i) {
      output_data[i] = input_data[i] * inv_rms * weight_data[i];
    }
  }
}

} // namespace nntrainer