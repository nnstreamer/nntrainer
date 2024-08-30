// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 SeungBaek Hong <sb92.hong@samsung.com>
 *
 * @file   mul_layer.cpp
 * @date   30 August 2024
 * @see    https://github.com/nnstreamer/nntrainer
 * @author SeungBaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is mul layer class (operation layer)
 *
 */

#include <mul_layer.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <util_func.h>

#include <layer_context.h>

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

void MulLayer::finalize(InitLayerContext &context) {
  context.setOutputDimensions({context.getInputDimensions()[0]});
}

void MulLayer::forwarding(RunLayerContext &context, bool training) {
  Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);

  const Tensor &input0 = context.getInput(0);
  const Tensor &input1 = context.getInput(1);

  input0.multiply(input1, hidden_);
}

void MulLayer::incremental_forwarding(RunLayerContext &context,
                                      unsigned int from, unsigned int to,
                                      bool training) {
  Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);
  TensorDim hidden_dim = hidden_.getDim();
  TensorDim hidden_step_dim = hidden_dim;

  if (from) {
    NNTR_THROW_IF(to - from != 1, std::invalid_argument)
      << "incremental step size is not 1";
    from = 0;
    to = 1;
  }

  hidden_step_dim.batch(1);
  hidden_step_dim.height(to - from);

  for (unsigned int b = 0; b < hidden_.batch(); ++b) {
    Tensor hidden_step = hidden_.getSharedDataTensor(
      hidden_step_dim, b * hidden_dim.getFeatureLen(), true);

    const Tensor &input0 = context.getInput(0);
    const Tensor &input1 = context.getInput(1);

    TensorDim input_dim = input0.getDim();
    TensorDim input_step_dim = input_dim;
    input_step_dim.batch(1);
    input_step_dim.height(to - from);

    Tensor input0_step = input0.getSharedDataTensor(
      input_step_dim, b * input_dim.getFeatureLen(), true);

    Tensor input1_step = input1.getSharedDataTensor(
      input_step_dim, b * input_dim.getFeatureLen(), true);

    input0_step.multiply(input1_step, hidden_step);
  }
}

void MulLayer::calcDerivative(RunLayerContext &context) {
  context.getOutgoingDerivative(0).copy(
    context.getIncomingDerivative(SINGLE_INOUT_IDX)
      .multiply(context.getInput(1)));

  context.getOutgoingDerivative(1).copy(
    context.getIncomingDerivative(SINGLE_INOUT_IDX)
      .multiply(context.getInput(0)));
}

void MulLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, mul_props);
  if (!remain_props.empty()) {
    std::string msg = "[MulLayer] Unknown Layer Properties count " +
                      std::to_string(values.size());
    throw exception::not_supported(msg);
  }
}
} /* namespace nntrainer */
