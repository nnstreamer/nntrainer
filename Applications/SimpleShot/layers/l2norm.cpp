// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   l2norm.cpp
 * @date   09 Jan 2021
 * @brief  This file contains the simple l2norm layer which normalizes
 * the given feature
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#include <iostream>
#include <regex>
#include <sstream>

#include <nntrainer_error.h>
#include <nntrainer_log.h>

#include <l2norm.h>

namespace simpleshot {
namespace layers {

static constexpr size_t SINGLE_INOUT_IDX = 0;

void L2NormLayer::finalize(nntrainer::InitLayerContext &context) {
  const auto &input_dim = context.getInputDimensions()[0];
  if (context.getNumInputs() != 1)
    throw std::invalid_argument(
      "l2norm layer is designed for a single input only");
  if (input_dim.channel() != 1 || input_dim.height() != 1) {
    throw std::invalid_argument(
      "l2norm layer is designed for channel and height is 1 for now, "
      "please check");
  }

  context.setOutputDimensions(context.getInputDimensions());
}

void L2NormLayer::forwarding(nntrainer::RunLayerContext &context,
                             bool training) {
  auto &hidden_ = context.getOutput(SINGLE_INOUT_IDX);
  auto &input_ = context.getInput(SINGLE_INOUT_IDX);

  input_.multiply(1 / input_.l2norm(), hidden_);
}

void L2NormLayer::calcDerivative(nntrainer::RunLayerContext &context) {
  throw std::invalid_argument("[L2Norm::calcDerivative] This Layer "
                              "does not support backward propagation");
}

void L2NormLayer::setProperty(const std::vector<std::string> &values) {
  if (!values.empty()) {
    std::string msg = "[FlattenLayer] Unknown Layer Properties count " +
                      std::to_string(values.size());
    throw nntrainer::exception::not_supported(msg);
  }
}

} // namespace layers
} // namespace simpleshot
