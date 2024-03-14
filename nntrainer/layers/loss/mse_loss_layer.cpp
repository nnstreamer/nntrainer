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

#include "tensor.h"
#include <layer_context.h>
#include <mse_loss_layer.h>

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

void MSELossLayer::forwarding(RunLayerContext &context, bool training) {
  Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);

  Tensor empty_tensor;
  Tensor &y = context.getInput(SINGLE_INOUT_IDX).getDataType() ==
                  ml::train::TensorDim::DataType::FP32
                ? context.getInput(SINGLE_INOUT_IDX)
                : empty_tensor;

  if (y.empty())
    y = context.getInput(SINGLE_INOUT_IDX)
          .clone(ml::train::TensorDim::DataType::FP32);

  // hidden_ <- y2 - y;
  auto out_type = hidden_.getDataType();
  if (out_type != y_.getDataType()) {
    Tensor y = y_.clone(out_type);
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
  } else {
    if (context.isLabelAvailable(SINGLE_INOUT_IDX)) {
      Tensor &y2 = context.getLabel(SINGLE_INOUT_IDX);
      y2.subtract(y_, hidden_);

      /** calculate sum of squares normalized by size */
      float l2norm = hidden_.l2norm();
      l2norm *= l2norm / hidden_.size();

      /** wrap in tensor for update loss */
      Tensor l = Tensor(TensorDim(1, 1, 1, 1), &l2norm);
      LossLayer::updateLoss(context, l);
    }
    // fill the output
    hidden_.fill(y_);
  }
}

void MSELossLayer::calcDerivative(RunLayerContext &context) {
  Tensor empty_tensor;

  Tensor &ret_derivative =
    context.getOutgoingDerivative(SINGLE_INOUT_IDX).getDataType() ==
        ml::train::TensorDim::DataType::FP32
      ? context.getOutgoingDerivative(SINGLE_INOUT_IDX)
      : empty_tensor;

  if (ret_derivative.empty())
    ret_derivative = context.getOutgoingDerivative(SINGLE_INOUT_IDX)
                       .clone(ml::train::TensorDim::DataType::FP32);

  Tensor &y = context.getInput(SINGLE_INOUT_IDX).getDataType() ==
                  ml::train::TensorDim::DataType::FP32
                ? context.getInput(SINGLE_INOUT_IDX)
                : empty_tensor;

  if (y.empty())
    y = context.getInput(SINGLE_INOUT_IDX)
          .clone(ml::train::TensorDim::DataType::FP32);

  const Tensor &y2 = context.getIncomingDerivative(SINGLE_INOUT_IDX);

  const auto &in_type = y.getDataType();
  if (in_type != y2.getDataType()) {
    Tensor y2_ = y2.clone(in_type);
    y.subtract(y2_, ret_derivative);
  } else {
    y.subtract(y2, ret_derivative);
  }

  applyLossScale(ret_derivative);

  float divider = ((float)y.size()) / 2;

  /**
   * ret_derivative may be eliminated by big divider with fp16 calculation.
   * So, it calcuated with larger precision.
   */
  int ret;
  if (ret_derivative.getDataType() != ml::train::TensorDim::DataType::FP32) {
    Tensor ret_derivative_ =
      ret_derivative.clone(ml::train::TensorDim::DataType::FP32);
    ret = ret_derivative_.divide_i(divider);
    ret_derivative.copyData(ret_derivative_);
  } else {
    ret = ret_derivative.divide_i(divider);
  }

  if (ret != ML_ERROR_NONE) {
    throw std::runtime_error(
      "[MSELossLayer::calcDerivative] Error when calculating loss");
  }

  // Loss Scale needs Full precsiion of ret_derivative. Therefore,
  // ret_derivateive should be FP32 when applying scale, and after applying it
  // need to convert original type for backpropagating.

  LossLayer::applyLossScale(context, ret_derivative);

  if (context.getOutgoingDerivative(SINGLE_INOUT_IDX).getDataType() !=
      ml::train::TensorDim::DataType::FP32)
    context.getOutgoingDerivative(SINGLE_INOUT_IDX).copyData(ret_derivative);
}

} // namespace nntrainer
