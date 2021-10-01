// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   attention_layer.h
 * @date   1 October 2021
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is Attention Layer Class for Neural Network
 *
 */

#include <attention_layer.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

enum AttentionParams { query = 0, value = 1 };

void AttentionLayer::finalize(InitLayerContext &context) {
  if (context.getNumInputs() != 2)
    throw std::runtime_error(
      "Attention layer does not support exclusive keys.");

  sm.setActiFunc(ActivationType::ACT_SOFTMAX);

  auto const &all_shapes = context.getInputDimensions();
  auto const &query_shape = all_shapes[AttentionParams::query];

  context.setOutputDimensions({query_shape});
}

void AttentionLayer::forwarding(RunLayerContext &context, bool training) {
  Tensor &query = context.getInput(AttentionParams::query);
  Tensor &value = context.getInput(AttentionParams::value);

  Tensor &output = context.getOutput(SINGLE_INOUT_IDX);
  Tensor distribution;

  Tensor score = query.dot(value, false, true);
  sm.run_fn(score, distribution);
  distribution.dot(value, output);
}

void AttentionLayer::calcDerivative(RunLayerContext &context) {
  /**
   * Not yet implemented
   */
}

void AttentionLayer::setProperty(const std::vector<std::string> &values) {
  if (!values.empty()) {
    std::string msg = "[AttentionLayer] Unknown Layer Properties count " +
                      std::to_string(values.size());
    throw exception::not_supported(msg);
  }
}

} /* namespace nntrainer */
