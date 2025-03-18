// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 SeungBaek Hong <sb92.hong@samsung.com>
 *
 * @file   sqrt_layer.cpp
 * @date   18 March 2025
 * @see    https://github.com/nnstreamer/nntrainer
 * @author SeungBaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is sqrt layer class (operation layer)
 *
 */

#include "common_properties.h"
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <sqrt_layer.h>
#include <util_func.h>

#include <layer_context.h>

namespace nntrainer {

void SQRTLayer::finalize(InitLayerContext &context) {
  context.setOutputDimensions({context.getInputDimensions()[0]});
}

void SQRTLayer::forwarding_operation(const Tensor &input, Tensor &hidden) {
  throw std::invalid_argument(
    "SQRTLayer forwarding operation not supported yet");
}

void SQRTLayer::calcDerivative(RunLayerContext &context) {
  throw std::invalid_argument(
    "SQRTLayer backwarding operation not supported yet");
}

void SQRTLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, sqrt_props);
  if (!remain_props.empty()) {
    std::string msg = "[SQRTLayer] Unknown Layer Properties count " +
                      std::to_string(values.size());
    throw exception::not_supported(msg);
  }
}
} /* namespace nntrainer */
