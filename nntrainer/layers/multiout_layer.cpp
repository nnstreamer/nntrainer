// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file        multiout_layer.cpp
 * @date        05 Nov 2020
 * @see         https://github.com/nnstreamer/nntrainer
 * @author      Jijoong Moon <jijoong.moon@samsung.com>
 * @bug         No known bugs except for NYI items
 * @brief       This is Multi Output Layer Class for Neural Network
 *
 */

#include <cstring>
#include <layer_context.h>
#include <multiout_layer.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <util_func.h>

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

void MultiOutLayer::finalize(InitLayerContext &context) {
  std::vector<TensorDim> out_dims(context.getNumRequestedOutputs());
  const TensorDim &in_dim = context.getInputDimensions()[0];

  std::fill(out_dims.begin(), out_dims.end(), in_dim);
  context.setOutputDimensions(out_dims);
}

void MultiOutLayer::forwarding(RunLayerContext &context, bool training) {
  if (!context.executeInPlace()) {
    const Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);
    for (unsigned int idx = 0; idx < context.getNumOutputs(); ++idx) {
      context.getOutput(idx).fill(input_);
    }
  }
}

void MultiOutLayer::incremental_forwarding(RunLayerContext &context,
                                           unsigned int from, unsigned int to,
                                           bool training) {
  if (!context.executeInPlace()) {
    const Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);
    TensorDim input_dim = input_.getDim();
    TensorDim input_step_dim = input_dim;
    input_step_dim.height(to - from);

    Tensor input_step = input_.getSharedDataTensor(
      input_step_dim, from * input_dim.width(), true);

    for (unsigned int idx = 0; idx < context.getNumOutputs(); ++idx) {
      Tensor &output = context.getOutput(idx);

      TensorDim output_dim = output.getDim();
      TensorDim output_step_dim = output_dim;
      output_step_dim.height(to - from);
      // @todo: set reset stride as false. This implementation only works when
      // batch size is 1
      Tensor output_step = output.getSharedDataTensor(
        output_step_dim, from * output_dim.width(), true);
      output_step.fill(input_step);
    }
  }
}

void MultiOutLayer::calcDerivative(RunLayerContext &context) {
  Tensor &ret = context.getOutgoingDerivative(SINGLE_INOUT_IDX);
  for (unsigned int idx = 0; idx < context.getNumOutputs(); ++idx) {
    if (idx == 0) {
      ret.copy(context.getIncomingDerivative(idx));
    } else {
      ret.add_i(context.getIncomingDerivative(idx));
    }
  }
}

void MultiOutLayer::setProperty(const std::vector<std::string> &values) {
  if (!values.empty()) {
    std::string msg = "[MultioutLayer] Unknown Layer Properties count " +
                      std::to_string(values.size());
    throw exception::not_supported(msg);
  }
}

} /* namespace nntrainer */
