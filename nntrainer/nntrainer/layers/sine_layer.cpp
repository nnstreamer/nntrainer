// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 SeungBaek Hong <sb92.hong@samsung.com>
 *
 * @file   sine_layer.cpp
 * @date   19 March 2025
 * @see    https://github.com/nnstreamer/nntrainer
 * @author SeungBaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is sine layer class (operation layer)
 *
 */

#include "common_properties.h"
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <sine_layer.h>
#include <util_func.h>

#include <layer_context.h>

namespace nntrainer {

void SineLayer::finalize(InitLayerContext &context) {
  context.setOutputDimensions({context.getInputDimensions()[0]});
}

void SineLayer::forwarding_operation(const Tensor &input, Tensor &hidden) {
  input.sin(hidden);
}

void SineLayer::calcDerivative(RunLayerContext &context) {
  auto &deriv = context.getOutgoingDerivative(SINGLE_INOUT_IDX);
  context.getInput(SINGLE_INOUT_IDX).cos(deriv);
  deriv.multiply(context.getIncomingDerivative(SINGLE_INOUT_IDX));
}

void SineLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, sine_props);
  if (!remain_props.empty()) {
    std::string msg = "[SineLayer] Unknown Layer Properties count " +
                      std::to_string(values.size());
    throw exception::not_supported(msg);
  }
}
} /* namespace nntrainer */
