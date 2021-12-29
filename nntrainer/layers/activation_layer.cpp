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
#include <stdexcept>
#include <vector>

#include <activation_layer.h>
#include <blas_interface.h>
#include <common_properties.h>
#include <layer_context.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <tensor.h>
#include <tensor_wrap_specs.h>
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

  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "activation layer, " << context.getName()
    << "requires exactly one input, but given: " << context.getNumInputs()
    << ", check graph connection if it is correct";

  /// @todo for only certain types of activation needs lifespan of
  /// forward_derivative order
  std::vector<VarGradSpecV2> out_specs;
  out_specs.push_back(
    InitLayerContext::outSpec(context.getInputDimensions()[0], "out",
                              TensorLifespan::FORWARD_DERIV_LIFESPAN));
  context.requestOutputs(std::move(out_specs));
  acti_func.executeInPlace(context.executeInPlace());
}

void ActivationLayer::forwarding(RunLayerContext &context, bool training) {
  Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);
  Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);
  acti_func.run_fn(input_, hidden_);
}

void ActivationLayer::calcDerivative(RunLayerContext &context) {
  const Tensor &deriv = context.getIncomingDerivative(SINGLE_INOUT_IDX);
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

  auto &act = std::get<props::Activation>(*activation_props);
  if (!act.empty())
    acti_func.setActiFunc(act.get());
}

}; // namespace nntrainer
