// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   mse_loss_layer.cpp
 * @date   24 June 2021
 * @brief  This is MSE Loss Layer Class of Neural Network
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#include <lazy_tensor.h>
#include <mse_loss_layer.h>

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

void MSELossLayer::forwarding(RunLayerContext &context, bool training) {
  Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);
  // TODO: try Tensor & - it should work
  Tensor y = context.getInput(SINGLE_INOUT_IDX);
  Tensor l;

  // fill the output
  hidden_.fill(y);
  // y2 <- y2 - y;
  if (context.isLabelAvailable(SINGLE_INOUT_IDX)) {
    Tensor &y2 = context.getLabel(SINGLE_INOUT_IDX);
    Tensor residual = y2.subtract(y);
    l = residual.chain().multiply_i(residual).average().run();
    LossLayer::updateLoss(context, l);
  }
}

void MSELossLayer::calcDerivative(RunLayerContext &context) {
  Tensor &ret_derivative = context.getOutgoingDerivative(SINGLE_INOUT_IDX);
  Tensor &y2 = context.getIncomingDerivative(SINGLE_INOUT_IDX);
  Tensor &y = context.getInput(SINGLE_INOUT_IDX);

  y.subtract(y2, ret_derivative);
  ret_derivative.multiply_i(2);
  if (ret_derivative.divide_i(y.length()) != ML_ERROR_NONE) {
    throw std::runtime_error(
      "[MSELossLayer::calcDerivative] Error when calculating loss");
  }
}

} // namespace nntrainer
