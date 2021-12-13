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

#include <layer_context.h>
#include <mse_loss_layer.h>

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

void MSELossLayer::forwarding(RunLayerContext &context, bool training) {
  Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);
  Tensor &y = context.getInput(SINGLE_INOUT_IDX);

  // hidden_ <- y2 - y;
  if (context.isLabelAvailable(SINGLE_INOUT_IDX)) {
    Tensor &y2 = context.getLabel(SINGLE_INOUT_IDX);
    y2.subtract(y, hidden_);

    /** calculate sum of squares normalized by size */
    float l2norm = hidden_.l2norm();
    l2norm *= l2norm / hidden_.size();

    /** wrap in tensor for update loss */
    Tensor l = Tensor(TensorDim(1, 1, 1, 1), &l2norm);
    LossLayer::updateLoss(context, l);
  }

  // fill the output
  hidden_.fill(y);
}

void MSELossLayer::calcDerivative(RunLayerContext &context) {
  Tensor &ret_derivative = context.getOutgoingDerivative(SINGLE_INOUT_IDX);
  Tensor &y2 = context.getIncomingDerivative(SINGLE_INOUT_IDX);
  Tensor &y = context.getInput(SINGLE_INOUT_IDX);

  y.subtract(y2, ret_derivative);
  float divider = ((float)y.size()) / 2;
  if (ret_derivative.divide_i(divider) != ML_ERROR_NONE) {
    throw std::runtime_error(
      "[MSELossLayer::calcDerivative] Error when calculating loss");
  }
}

} // namespace nntrainer
