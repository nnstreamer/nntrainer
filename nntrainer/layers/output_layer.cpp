// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file        output_layer.cpp
 * @date        05 Nov 2020
 * @see         https://github.com/nnstreamer/nntrainer
 * @author      Jijoong Moon <jijoong.moon@samsung.com>
 * @bug         No known bugs except for NYI items
 * @brief       This is Multi Output Layer Class for Neural Network
 *
 */

#include <cstring>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <output_layer.h>
#include <parse_util.h>
#include <util_func.h>

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

void OutputLayer::finalize(InitLayerContext &context) {
  std::vector<TensorDim> out_dims(context.getNumOutputs());
  const TensorDim &in_dim = context.getInputDimensions()[0];

  std::fill(out_dims.begin(), out_dims.end(), in_dim);
  context.setOutputDimensions(out_dims);
}

void OutputLayer::forwarding(RunLayerContext &context, bool training) {
  const Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);
  for (unsigned int idx = 0; idx < context.getNumOutputs(); ++idx) {
    context.getOutput(idx).fill(input_);
  }
}

void OutputLayer::calcDerivative(RunLayerContext &context) {
  Tensor &ret = context.getOutgoingDerivative(SINGLE_INOUT_IDX);
  for (unsigned int idx = 0; idx < context.getNumOutputs(); ++idx) {
    if (idx == 0) {
      ret.copy(context.getIncomingDerivative(idx));
    } else {
      ret.add_i(context.getIncomingDerivative(idx));
    }
  }
}

void OutputLayer::setProperty(const std::vector<std::string> &values) {
  if (!values.empty()) {
    std::string msg = "[FlattenLayer] Unknown Layer Properties count " +
                      std::to_string(values.size());
    throw exception::not_supported(msg);
  }
}

} /* namespace nntrainer */
