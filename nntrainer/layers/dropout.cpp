// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   dropout_layer.cpp
 * @date   16 June 2020
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug	   No known bugs except for NYI items
 * @brief  This is Dropout Layer Class for Neural Network
 *
 */

#include <dropout.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <util_func.h>

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

void DropOutLayer::finalize(InitLayerContext &context) {
  auto const &input_dims = context.getInputDimensions();
  context.setOutputDimensions(input_dims);

  mask_idx.reserve(input_dims.size());
  for (auto &t : input_dims) {
    mask_idx.push_back(
      context.requestTensor(t, "Mask", Tensor::Initializer::NONE, false,
                            TensorLifespan::ITERATION_LIFESPAN));
  }
}

void DropOutLayer::forwarding(RunLayerContext &context, bool training) {
  auto &rate_ = std::get<props::DropOutRate>(dropout_rate).get();

  // Assume it is in-place calculation. It means input and output share mem
  // buffer. So if the training is false, the output is the same with input. In
  // other words, there is nothing happen during inference.

  for (unsigned int i = 0; i < context.getNumInputs(); ++i) {
    Tensor &input_ = context.getInput(i);
    Tensor &output_ = context.getOutput(i);

    /** @todo make this in-place */
    if (training && rate_ > epsilon) {
      Tensor &mask_ = context.getTensor(mask_idx[i]);
      mask_.dropout_mask(rate_);
      input_.multiply(mask_, output_);
    } else {
      output_.fill(input_);
    }
  }
}

void DropOutLayer::calcDerivative(RunLayerContext &context) {
  // Assume it is in-place calculation
  auto &rate_ = std::get<props::DropOutRate>(dropout_rate).get();

  for (unsigned int i = 0; i < context.getNumInputs(); ++i) {
    Tensor &derivative_ = context.getIncomingDerivative(i);
    Tensor &ret_ = context.getOutgoingDerivative(SINGLE_INOUT_IDX);

    /** @todo make this in-place */
    if (rate_ > epsilon) {
      Tensor &mask_ = context.getTensor(mask_idx[i]);
      derivative_.multiply(mask_, ret_);
    } else {
      ret_.fill(derivative_);
    }
  }
}

void DropOutLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, dropout_rate);
  if (!remain_props.empty()) {
    std::string msg = "[DropOutLayer] Unknown Layer Properties count " +
                      std::to_string(values.size());
    throw exception::not_supported(msg);
  }
}

} /* namespace nntrainer */
