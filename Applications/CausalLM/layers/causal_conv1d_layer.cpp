// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Hyeong-Gwon Hong
 *
 * @file   causal_conv1d_layer.cpp
 * @date   01 April 2026
 * @brief  Causal depthwise Conv1D layer with conv-state cache for CausalLM
 * @see    https://github.com/nntrainer/nntrainer
 * @author Hyeong-Gwon Hong
 * @bug    No known bugs except for NYI items
 */

#include "causal_conv1d_layer.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <cpu_backend.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <tensor.h>

namespace causallm {

namespace {

/**
 * @brief Reverse the rows of a [B, T, W] contiguous buffer along T, in-place.
 */
void flip_rows_along_time(float *data, unsigned int B, unsigned int T,
                          unsigned int W) {
  for (unsigned int b = 0; b < B; ++b) {
    float *base = data + static_cast<size_t>(b) * T * W;
    for (unsigned int t = 0; t < T / 2; ++t) {
      float *front = base + static_cast<size_t>(t) * W;
      float *back = base + static_cast<size_t>(T - 1 - t) * W;
      std::swap_ranges(front, front + W, back);
    }
  }
}

} // namespace

// ===========================================================================

CausalConv1DLayer::CausalConv1DLayer() : LayerImpl() {
  weight_idx.fill(std::numeric_limits<unsigned int>::max());
  tensor_idx.fill(std::numeric_limits<unsigned int>::max());
}

void CausalConv1DLayer::validateInputShape(
  const nntrainer::TensorDim &input_dim) const {
  NNTR_THROW_IF(input_dim.channel() != 1, std::invalid_argument)
    << "[CausalConv1DLayer] input channel must be 1 (B×1×T×W layout), got "
    << input_dim.channel();
  NNTR_THROW_IF(input_dim.height() < 1 || input_dim.width() < 1,
                std::invalid_argument)
    << "[CausalConv1DLayer] invalid input shape: H and W must be positive.";
}

void CausalConv1DLayer::finalize(nntrainer::InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "[CausalConv1DLayer] requires exactly 1 input, got "
    << context.getNumInputs();

  const nntrainer::TensorDim &in_dim = context.getInputDimensions()[0];
  validateInputShape(in_dim);

  const unsigned int B = in_dim.batch();
  const unsigned int W = in_dim.width(); // number of features/channels

  // Weight: [1, 1, K, W] FP32
  //   Row k  (offset k*W) = kernel weights for position k:
  //     k=0 → w0: applied to current token x_t
  //     k=1 → w1: applied to x_{t-1}
  //     ...
  nntrainer::TensorDim weight_dim(
    {1, 1, kernel_size, W},
    {context.getFormat(), ml::train::TensorDim::DataType::FP32});
  weight_idx[weight] =
    context.requestWeight(weight_dim, nntrainer::Initializer::NONE,
                          nntrainer::WeightRegularizer::NONE, 0.0f, 0.0f,
                          "causal_conv1d_weight", false);

  // Conv-state cache: [B, 1, K-1, W] FP32
  //   state[b, 0, 0, f] = oldest cached token
  //   state[b, 0, K-2, f] = x_{t-1} (newest)
  nntrainer::TensorDim state_dim(
    {B, 1, kernel_size - 1, W},
    {context.getFormat(), ml::train::TensorDim::DataType::FP32});
  tensor_idx[conv_state] = context.requestTensor(
    state_dim, "conv_state", nntrainer::Initializer::ZEROS, false,
    nntrainer::TensorLifespan::MAX_LIFESPAN);

  // Output has same shape as input
  context.setOutputDimensions({in_dim});
}

void CausalConv1DLayer::forwarding(nntrainer::RunLayerContext &context,
                                   bool training) {
  (void)training;
  // Full-sequence forward, used by training. The decode-time conv state is
  // not touched here; a later incremental prefill rewrites it from scratch
  // before any decode step runs.
  nntrainer::Tensor &input = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &output = context.getOutput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &w_tensor = context.getWeight(weight_idx[weight]);

  const unsigned int B = input.batch();
  const unsigned int T = input.height(); // sequence length
  const unsigned int W = input.width();  // feature dimension

  const float *x = input.getData<float>();
  const float *w_ptr = w_tensor.getData<float>();
  float *y = output.getData<float>();

  if (kernel_size == 4)
    nntrainer::causal_depthwise_conv1d_k4(x, w_ptr, nullptr, y, B, T, W);
  else
    nntrainer::causal_depthwise_conv1d_k3(x, w_ptr, nullptr, y, B, T, W);
}

void CausalConv1DLayer::incremental_forwarding(
  nntrainer::RunLayerContext &context, unsigned int from, unsigned int to,
  bool training) {

  NNTR_THROW_IF(training, std::invalid_argument)
    << "[CausalConv1DLayer] training/backward must use forwarding().";
  NNTR_THROW_IF(to == 0 || to <= from, std::invalid_argument)
    << "[CausalConv1DLayer] invalid range: from=" << from << ", to=" << to;

  nntrainer::Tensor &input = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &output = context.getOutput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &w_tensor = context.getWeight(weight_idx[weight]);
  nntrainer::Tensor &state = context.getTensor(tensor_idx[conv_state]);

  const unsigned int B = input.batch();
  const unsigned int H = input.height(); // full sequence length (INIT_SEQ_LEN)
  const unsigned int W = input.width();  // feature dimension

  const float *w_ptr = w_tensor.getData<float>();
  float *state_data = state.getData<float>(); // [B, 1, K-1, W]

  if (to - from == 1) {
    // ----------------------------------------------------------------
    // Decode path (hot): single-token inference.
    // NNTrainer places the current token at offset 0 within each batch
    // slice (not at offset `from`).
    // ----------------------------------------------------------------
    for (unsigned int b = 0; b < B; ++b) {
      const float *x_cur = input.getData<float>() + b * H * W;
      float *y_cur = output.getData<float>() + b * H * W;
      float *s = state_data + b * (kernel_size - 1) * W;

      // y = w0*x + w1*x_{t-1} + ... + w_{K-1}*x_{t-K+1}; the state is
      // updated in-place by the kernel.
      if (kernel_size == 4)
        nntrainer::causal_depthwise_conv1d_k4_decode(x_cur, w_ptr, s, y_cur, W);
      else
        nntrainer::causal_depthwise_conv1d_k3_decode(x_cur, w_ptr, s, y_cur, W);
    }

  } else {
    // ----------------------------------------------------------------
    // Prefill path: process all positions [0, to).
    // ----------------------------------------------------------------
    for (unsigned int b = 0; b < B; ++b) {
      const float *x = input.getData<float>() + b * H * W;
      float *y = output.getData<float>() + b * H * W;

      if (kernel_size == 4)
        nntrainer::causal_depthwise_conv1d_k4(x, w_ptr, nullptr, y, 1, to, W);
      else
        nntrainer::causal_depthwise_conv1d_k3(x, w_ptr, nullptr, y, 1, to, W);

      // Save last K-1 tokens to state for the decode steps.
      // Row k holds the token at position to - (K-1) + k, oldest first;
      // positions before the sequence start (short prefill) stay zero.
      float *s = state_data + b * (kernel_size - 1) * W;
      for (unsigned int k = 0; k < kernel_size - 1; ++k) {
        const int pos = static_cast<int>(to) -
                        static_cast<int>(kernel_size - 1) + static_cast<int>(k);
        if (pos >= 0)
          std::memcpy(s + k * W, x + static_cast<unsigned int>(pos) * W,
                      W * sizeof(float));
        else
          std::memset(s + k * W, 0, W * sizeof(float));
      }
    }
  }
}

void CausalConv1DLayer::calcDerivative(nntrainer::RunLayerContext &context) {
  const nntrainer::Tensor &dy = context.getIncomingDerivative(SINGLE_INOUT_IDX);
  nntrainer::Tensor &dx = context.getOutgoingDerivative(SINGLE_INOUT_IDX);
  nntrainer::Tensor &w_tensor = context.getWeight(weight_idx[weight]);

  const unsigned int B = dy.batch();
  const unsigned int T = dy.height();
  const unsigned int W = dy.width();
  const float *w_ptr = w_tensor.getData<float>();

  // Forward: y[t] = sum_j w_j * x[t-j]  (x[<0] = 0).
  // Adjoint: dx[t] = sum_j w_j * dy[t+j] (dy[>=T] = 0), an anti-causal conv.
  // With r = flip(dy) and o = flip(dx) this becomes a causal conv of r with
  // the same kernel, so the prefill kernels are reused as-is.
  nntrainer::Tensor dy_flipped(dy.getDim());
  for (unsigned int b = 0; b < B; ++b) {
    const float *src = dy.getData<float>() + static_cast<size_t>(b) * T * W;
    float *dst = dy_flipped.getData<float>() + static_cast<size_t>(b) * T * W;
    for (unsigned int t = 0; t < T; ++t)
      std::memcpy(dst + static_cast<size_t>(T - 1 - t) * W,
                  src + static_cast<size_t>(t) * W, W * sizeof(float));
  }

  float *dx_ptr = dx.getData<float>();
  if (kernel_size == 4)
    nntrainer::causal_depthwise_conv1d_k4(dy_flipped.getData<float>(), w_ptr,
                                          nullptr, dx_ptr, B, T, W);
  else
    nntrainer::causal_depthwise_conv1d_k3(dy_flipped.getData<float>(), w_ptr,
                                          nullptr, dx_ptr, B, T, W);

  flip_rows_along_time(dx_ptr, B, T, W);
}

void CausalConv1DLayer::calcGradient(nntrainer::RunLayerContext &context) {
  // The conv weight is requested trainable=false, so the node-level trainable
  // gate normally skips this call. Only dX flows through this layer.
  (void)context;
  throw std::invalid_argument(
    "[CausalConv1DLayer] trainable conv weight is not supported; only "
    "input gradients flow through this layer.");
}

void CausalConv1DLayer::updateTensorsByInputDimensions(
  nntrainer::RunLayerContext &context,
  std::vector<nntrainer::TensorDim> input_dimensions) {
  // No dynamic updates needed
}

void CausalConv1DLayer::exportTo(nntrainer::Exporter &exporter,
                                 const ml::train::ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
}

void CausalConv1DLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = nntrainer::loadProperties(values, conv_props);
  LayerImpl::setProperty(remain_props);

  auto &k = std::get<nntrainer::props::KernelSize>(conv_props);
  if (!k.empty()) {
    NNTR_THROW_IF(k.get() != 3 && k.get() != 4, std::invalid_argument)
      << "[CausalConv1DLayer] supported kernel sizes are 3 and 4, got "
      << k.get();
    kernel_size = k.get();
  }
}

} // namespace causallm

#ifdef PLUGGABLE
extern "C" {

nntrainer::Layer *create_causal_conv1d_layer() {
  return new causallm::CausalConv1DLayer();
}

void destroy_causal_conv1d_layer(nntrainer::Layer *layer) { delete layer; }

nntrainer::LayerPluggable ml_train_layer_pluggable{
  create_causal_conv1d_layer, destroy_causal_conv1d_layer,
  causallm::CausalConv1DLayer::type};
}
#endif
