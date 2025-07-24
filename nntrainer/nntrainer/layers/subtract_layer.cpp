// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 SeungBaek Hong <sb92.hong@samsung.com>
 *
 * @file   subtract_layer.cpp
 * @date   10 Oct 2024
 * @see    https://github.com/nnstreamer/nntrainer
 * @author SeungBaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is subtract layer class (operation layer)
 *
 */

#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <subtract_layer.h>
#include <util_func.h>

#include <layer_context.h>

namespace nntrainer {

void SubtractLayer::finalize(InitLayerContext &context) {
  context.setOutputDimensions({context.getInputDimensions()[0]});
}

void SubtractLayer::forwarding_operation(const Tensor &input0,
                                         const Tensor &input1, Tensor &hidden) {
  input0.subtract(input1, hidden);
}

void SubtractLayer::calcDerivative(RunLayerContext &context) {
  context.getOutgoingDerivative(0).copy(
    context.getIncomingDerivative(SINGLE_INOUT_IDX));

  context.getOutgoingDerivative(1).copy(
    context.getIncomingDerivative(SINGLE_INOUT_IDX).multiply(-1));
}

void SubtractLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, subtract_props);
  if (!remain_props.empty()) {
    std::string msg = "[SubtractLayer] Unknown Layer Properties count " +
                      std::to_string(values.size());
    throw exception::not_supported(msg);
  }
}
} /* namespace nntrainer */
