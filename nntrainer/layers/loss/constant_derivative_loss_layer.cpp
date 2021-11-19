// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file constant_derivative_loss_layer.cpp
 * @date 05 Oct 2021
 * @brief This patch contains constant derivative loss implementation
 * @note This is special type of loss to feed an arbitrary derivative value to
 * the last layer.
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug No known bugs except for NYI items
 */

#include <constant_derivative_loss_layer.h>

#include <layer_context.h>

namespace nntrainer {

static constexpr int SINGLE_INOUT_IDX = 0;
/// @todo make this property
static constexpr float value = 1.0f;

ConstantDerivativeLossLayer::ConstantDerivativeLossLayer() : LossLayer() {}
ConstantDerivativeLossLayer::~ConstantDerivativeLossLayer(){};

void ConstantDerivativeLossLayer::forwarding(RunLayerContext &context,
                                             bool training) {
  Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);
  Tensor &y = context.getInput(SINGLE_INOUT_IDX);

  // fill the output
  hidden_.fill(y);

  if (context.isLabelAvailable(SINGLE_INOUT_IDX)) {
    Tensor l(1);
    l.setValue(value);
    // update the loss value
    LossLayer::updateLoss(context, l);
  }
}

void ConstantDerivativeLossLayer::setProperty(
  const std::vector<std::string> &values) {
  /// update set value
  LossLayer::setProperty(values);
}

void ConstantDerivativeLossLayer::calcDerivative(RunLayerContext &context) {
  Tensor &ret_derivative = context.getOutgoingDerivative(SINGLE_INOUT_IDX);
  ret_derivative.setValue(1.0f);
}

} // namespace nntrainer
