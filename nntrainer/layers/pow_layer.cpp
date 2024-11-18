// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 SeungBaek Hong <sb92.hong@samsung.com>
 *
 * @file   pow_layer.cpp
 * @date   20 Nov 2024
 * @see    https://github.com/nnstreamer/nntrainer
 * @author SeungBaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is pow layer class (operation layer)
 *
 */

#include "common_properties.h"
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <pow_layer.h>
#include <util_func.h>

#include <layer_context.h>

namespace nntrainer {

void PowLayer::finalize(InitLayerContext &context) {
  context.setOutputDimensions({context.getInputDimensions()[0]});
}

void PowLayer::forwarding_operation(const Tensor &input, Tensor &hidden) {
  float exponent = std::get<props::Exponent>(pow_props).get();
  input.pow(exponent, hidden);
}

void PowLayer::calcDerivative(RunLayerContext &context) {
  float exp = std::get<props::Exponent>(pow_props).get();
  context.getOutgoingDerivative(0).copy(
    context.getIncomingDerivative(SINGLE_INOUT_IDX)
      .multiply(exp)
      .multiply(context.getInput(0).pow(exp - 1.0f)));
}

void PowLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, pow_props);
  if (!remain_props.empty()) {
    std::string msg = "[PowLayer] Unknown Layer Properties count " +
                      std::to_string(values.size());
    throw exception::not_supported(msg);
  }
}
} /* namespace nntrainer */
