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
#include <tensor.h>

#include <l2norm.h>

namespace simpleshot {
namespace layers {

const std::string L2NormLayer::type = "l2norm";

int L2NormLayer::initialize(nntrainer::Manager &manager) {
  if (input_dim[0].channel() != 1 || input_dim[0].height() != 1) {
    ml_logw("l2norm layer is designed for channel and height is 1 for now, "
            "please check");
  }
  output_dim[0] = input_dim[0];

  return ML_ERROR_NONE;
}

void L2NormLayer::forwarding(bool training) {
  auto &hidden_ = net_hidden[0]->getVariableRef();
  auto &input_ = net_input[0]->getVariableRef();

  input_.multiply(1 / input_.l2norm(), hidden_);
}

void L2NormLayer::calcDerivative() {
  throw std::invalid_argument("[L2Norm::calcDerivative] This Layer "
                              "does not support backward propagation");
}

} // namespace layers
} // namespace simpleshot
