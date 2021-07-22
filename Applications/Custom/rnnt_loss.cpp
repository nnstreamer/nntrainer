// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   rnnt_loss.cpp
 * @date   22 July 2021
 * @brief  This file contains the rnnt loss
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */
#include "rnnt_loss.h"

#include <cmath>

#include <tensor.h>

constexpr const float EPSILON_ = 1e-7;
namespace custom {

void RNNTLossLayer::finalize(nntrainer::InitLayerContext &context) {
  context.setOutputDimensions(context.getInputDimensions());
  // NYI
}

void RNNTLossLayer::setProperty(const std::vector<std::string> &values) {
  if (!values.empty()) {
    std::string msg = "[RNNTLossLayer] Unknown Layer Properties count " +
                      std::to_string(values.size());
    throw std::invalid_argument(msg);
  }
}

void RNNTLossLayer::forwarding(nntrainer::RunLayerContext &context,
                               bool training) {
  // nntrainer::Tensor &predicted = context.getInput(SINGLE_INOUT_IDX);
  // nntrainer::Tensor &output = context.getOutput(SINGLE_INOUT_IDX);
  // NYI
}

void RNNTLossLayer::calcDerivative(nntrainer::RunLayerContext &context) {
  // nntrainer::Tensor &predicted = context.getInput(SINGLE_INOUT_IDX);
  // nntrainer::Tensor &label = context.getLabel(SINGLE_INOUT_IDX);

  // nntrainer::Tensor &deriv = context.getOutgoingDerivative(SINGLE_INOUT_IDX);
  // NYI
}

#ifdef PLUGGABLE

nntrainer::Layer *create_rnnt_loss_layer() {
  auto layer = new RNNTLossLayer();
  std::cout << "rnnt loss layer created\n";
  return layer;
}

void destory_rnnt_loss_layer(nntrainer::Layer *layer) {
  std::cout << "rnnt loss layer destroyed\n";
  delete layer;
}

/**
 * @note ml_train_layer_pluggable defines the entry point for nntrainer to
 * register a plugin layer
 */
extern "C" {
nntrainer::LayerPluggable ml_train_layer_pluggable{create_rnnt_loss_layer,
                                                   destory_rnnt_loss_layer};
}

#endif
} // namespace custom
