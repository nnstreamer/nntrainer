// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   mae_loss.cpp
 * @date   10 June 2021
 * @brief  This file contains the mean absolute error loss as a sample layer
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */
#include "mae_loss.h"

#include <cmath>

#include <iostream>
#include <tensor.h>

constexpr const float EPSILON_ = 1e-7;
namespace custom {

static constexpr size_t SINGLE_INOUT_IDX = 0;

void MaeLossLayer::forwarding(nntrainer::RunLayerContext &context,
                              bool training) {
  nntrainer::Tensor &predicted = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &output = context.getOutput(SINGLE_INOUT_IDX);

  if (!context.executeInPlace())
    output.fill(predicted);
}

void MaeLossLayer::calcDerivative(nntrainer::RunLayerContext &context) {
  nntrainer::Tensor &predicted = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &label = context.getLabel(SINGLE_INOUT_IDX);

  nntrainer::Tensor &deriv = context.getOutgoingDerivative(SINGLE_INOUT_IDX);

  /// This can be saved at MaeLossLayer::forwarding, but this is done here on
  /// purpose for demonstration purpose
  predicted.subtract(label, deriv);
  unsigned int size = predicted.size();
  float deriv_val = 1.0f / (float)size;

  deriv.apply_i([deriv_val](float x) {
    if (fabs(x) < EPSILON_) {
      return 0.0f;
    }
    return x > 0 ? deriv_val : -deriv_val;
  });
}

#ifdef PLUGGABLE

nntrainer::Layer *create_mae_loss_layer() {
  auto layer = new MaeLossLayer();
  std::cout << "mae loss layer created\n";
  return layer;
}

void destory_mae_loss_layer(nntrainer::Layer *layer) {
  std::cout << "mae loss layer destroyed\n";
  delete layer;
}

/**
 * @note ml_train_layer_pluggable defines the entry point for nntrainer to
 * register a plugin layer
 */
extern "C" {
nntrainer::LayerPluggable ml_train_layer_pluggable{create_mae_loss_layer,
                                                   destory_mae_loss_layer};
}

#endif
} // namespace custom
