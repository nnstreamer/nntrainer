// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   flatten_layer.cpp
 * @date   16 June 2020
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug	   No known bugs except for NYI items
 * @brief  This is Flatten Layer Class for Neural Network
 *
 */

#include <flatten_layer.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

void FlattenLayer::finalize(InitLayerContext &context) {
  if (context.getNumInputs() != 1) {
    throw std::invalid_argument("input_shape keyword is only for one input");
  }

  TensorDim out_dim;
  const TensorDim &in_dim = context.getInputDimensions()[0];
  if (in_dim.channel() == 1 && in_dim.height() == 1) {
    ml_logw("Warning: the flatten layer is redundant");
  }

  out_dim.batch(in_dim.batch());
  out_dim.channel(1);
  out_dim.height(1);
  out_dim.width(in_dim.getFeatureLen());

  context.setOutputDimensions({out_dim});
}

void FlattenLayer::forwarding(RunLayerContext &context, bool training) {
  /**
   * Below is intentionally not a reference.
   * Using refernce would change the shape of the output tensor for the previous
   * layer.
   */
  Tensor temp = context.getInput(SINGLE_INOUT_IDX);
  Tensor &out = context.getOutput(SINGLE_INOUT_IDX);

  temp.reshape(out.getDim());
  out = temp;
}

void FlattenLayer::calcDerivative(RunLayerContext &context) {
  /**
   * Below is intentionally not a reference.
   * Using refernce would change the shape of the output tensor for the previous
   * layer.
   */
  Tensor temp = context.getIncomingDerivative(SINGLE_INOUT_IDX);
  Tensor &ret_derivative = context.getOutgoingDerivative(SINGLE_INOUT_IDX);

  temp.reshape(ret_derivative.getDim());
  ret_derivative = temp;
}

void FlattenLayer::setProperty(const std::vector<std::string> &values) {
  if (!values.empty()) {
    std::string msg = "[FlattenLayer] Unknown Layer Properties count " +
                      std::to_string(values.size());
    throw exception::not_supported(msg);
  }
}
} /* namespace nntrainer */
