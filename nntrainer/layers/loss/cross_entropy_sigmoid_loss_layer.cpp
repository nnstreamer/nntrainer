// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   cross_entropy_sigmoid_loss_layer.cpp
 * @date   24 June 2021
 * @brief  This is MSE Loss Layer Class of Neural Network
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#include <cmath>

#include <cross_entropy_sigmoid_loss_layer.h>

#include <acti_func.h>
#include <lazy_tensor.h>
#include <util_func.h>

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

void CrossEntropySigmoidLossLayer::forwarding(RunLayerContext &context,
                                              bool training) {
  Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);
  // TODO: try Tensor & - it should work
  Tensor y = context.getInput(SINGLE_INOUT_IDX);

  // fill the output
  hidden_ = y.apply(ActiFunc::sigmoid, hidden_);

  if (context.isLabelAvailable(SINGLE_INOUT_IDX)) {
    Tensor &y2 = context.getLabel(SINGLE_INOUT_IDX);
    // @todo: change this to apply_i
    // @note: the output should be logit before applying sigmoid
    // log(1 + exp(-abs(y))) + max(y, 0)
    Tensor mid_term = y.apply(static_cast<float (*)(float)>(&std::fabs))
                        .multiply(-1.0)
                        .apply(static_cast<float (*)(float)>(&std::exp))
                        .add(1.0)
                        .apply(logFloat);
    mid_term = mid_term.add(y.apply(ActiFunc::relu));

    // y * y2
    Tensor end_term = y2.chain().multiply_i(y).run();

    // loss = log(1 + exp(-abs(y))) + max(y, 0) - (y * y2)
    l = mid_term.subtract(end_term).average();

    // update the loss value
    updateLoss(context, l);
  }
}

void CrossEntropySigmoidLossLayer::calcDerivative(RunLayerContext &context) {
  Tensor &ret_derivative = context.getOutgoingDerivative(SINGLE_INOUT_IDX);
  Tensor &y2 = context.getIncomingDerivative(SINGLE_INOUT_IDX);
  Tensor &y = context.getInput(SINGLE_INOUT_IDX);

  y.apply(ActiFunc::sigmoid, ret_derivative);
  ret_derivative.subtract_i(y2);
  if (ret_derivative.divide_i(ret_derivative.length()) != ML_ERROR_NONE) {
    throw std::runtime_error("[CrossEntropySigmoidLossLayer::calcDerivative] "
                             "Error when calculating loss");
  }
}

} // namespace nntrainer
