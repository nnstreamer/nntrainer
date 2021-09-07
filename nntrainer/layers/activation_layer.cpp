// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   activation_layer.cpp
 * @date   17 June 2020
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is Activation Layer Class for Neural Network
 *
 */

#include <algorithm>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <vector>

#include <activation_layer.h>
#include <blas_interface.h>
#include <common_properties.h>
#include <lazy_tensor.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <parse_util.h>
#include <tensor.h>
#include <util_func.h>

namespace nntrainer {
ActivationLayer::ActivationLayer() :
  Layer(),
  activation_props(new PropTypes(props::Activation())) {
  acti_func.setActiFunc(ActivationType::ACT_NONE);
}

static constexpr size_t SINGLE_INOUT_IDX = 0;

void ActivationLayer::finalize(InitLayerContext &context) {
  auto &act = std::get<props::Activation>(*activation_props);
  NNTR_THROW_IF(act.empty(), std::invalid_argument)
    << "activation has not been set!";
  acti_func.setActiFunc(act.get());
  context.setOutputDimensions(context.getInputDimensions());
}

void ActivationLayer::forwarding(RunLayerContext &context, bool training) {
  Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);
  Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);
  acti_func.run_fn(input_, hidden_);
}

void ActivationLayer::calcDerivative(RunLayerContext &context) {
  Tensor &deriv = context.getIncomingDerivative(SINGLE_INOUT_IDX);
  Tensor &ret = context.getOutgoingDerivative(SINGLE_INOUT_IDX);
  Tensor &out = context.getOutput(SINGLE_INOUT_IDX);

  acti_func.run_prime_fn(out, ret, deriv);
}

void ActivationLayer::exportTo(Exporter &exporter,
                               const ExportMethods &method) const {
  exporter.saveResult(*activation_props, method, this);
}

void ActivationLayer::setProperty(const std::vector<std::string> &values) {
  auto left = loadProperties(values, *activation_props);
  NNTR_THROW_IF(!left.empty(), std::invalid_argument)
    << "Failed to set property";
}

}; // namespace nntrainer
