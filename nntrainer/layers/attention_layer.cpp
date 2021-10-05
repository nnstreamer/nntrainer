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

enum AttentionParams { query = 0, value = 1, score, weights };

void AttentionLayer::finalize(InitLayerContext &context) {
  if (context.getNumInputs() != 2)
    throw std::runtime_error(
      "Attention layer does not support exclusive keys.");

  sm.setActiFunc(ActivationType::ACT_SOFTMAX);

  auto const &all_dims = context.getInputDimensions();
  auto const &query_dim = all_dims[AttentionParams::query];
  auto const &value_dim = all_dims[AttentionParams::value];

  wt_idx[AttentionParams::query] = query;
  wt_idx[AttentionParams::value] = value;

  auto weights_dim = query_dim;
  weights_dim.width(value_dim.height());
  wt_idx[AttentionParams::weights] = context.requestTensor(
    weights_dim, context.getName() + ":weights", Tensor::Initializer::NONE,
    false, TensorLifespan::ITERATION_LIFESPAN);

  wt_idx[AttentionParams::score] = context.requestTensor(
    weights_dim, context.getName() + ":score", Tensor::Initializer::NONE, false,
    TensorLifespan::FORWARD_FUNC_LIFESPAN);

  context.setOutputDimensions({query_dim});
}

void AttentionLayer::forwarding(RunLayerContext &context, bool training) {
  Tensor &query = context.getInput(AttentionParams::query);
  Tensor &value = context.getInput(AttentionParams::value);

  Tensor &output = context.getOutput(SINGLE_INOUT_IDX);
  Tensor &weights = context.getTensor(wt_idx[AttentionParams::weights]);
  Tensor &score = context.getTensor(wt_idx[AttentionParams::score]);

  for (unsigned int b = 0; b < query.batch(); b++) {
    /** @todo try using transpose to speedup the operation */
    Tensor query_b = query.getBatchSlice(b, 1);
    Tensor value_b = value.getBatchSlice(b, 1);
    Tensor score_b = score.getBatchSlice(b, 1);
    Tensor weights_b = weights.getBatchSlice(b, 1);
    Tensor output_b = output.getBatchSlice(b, 1);

    query_b.dot(value_b, score_b, false, true);
    sm.run_fn(score_b, weights_b);
    weights_b.dot(value_b, output_b);
  }
}

void AttentionLayer::calcDerivative(RunLayerContext &context) {
  Tensor &derivative = context.getIncomingDerivative(SINGLE_INOUT_IDX);
  Tensor &query = context.getInput(AttentionParams::query);
  Tensor &value = context.getInput(AttentionParams::value);

  Tensor &dquery = context.getOutgoingDerivative(AttentionParams::query);
  Tensor &dvalue = context.getOutgoingDerivative(AttentionParams::value);
  Tensor &weights = context.getTensor(wt_idx[AttentionParams::weights]);

  for (unsigned int b = 0; b < query.batch(); b++) {
    /** @todo try using transpose to speedup the operation */
    Tensor query_b = query.getBatchSlice(b, 1);
    Tensor value_b = value.getBatchSlice(b, 1);
    Tensor weights_b = weights.getBatchSlice(b, 1);

    Tensor dquery_b = dquery.getBatchSlice(b, 1);
    Tensor dvalue_b = dvalue.getBatchSlice(b, 1);
    Tensor deriv_b = derivative.getBatchSlice(b, 1);

    Tensor dweight = deriv_b.dot(value_b, false, true);

    Tensor t1;
    sm.run_prime_fn(weights_b, t1, dweight);

    Tensor t2 = t1.dot(value_b);
    t2.dot(value_b, false, true).dot(deriv_b, dquery_b);

    Tensor p1 = t1.dot(query_b).dot(value_b, false, true);
    Tensor p2 = p1.add(weights_b); 
    p2.dot(deriv_b, dvalue_b, true, false);
  }

}

void AttentionLayer::setProperty(const std::vector<std::string> &values) {
  if (!values.empty()) {
    std::string msg = "[AttentionLayer] Unknown Layer Properties count " +
                      std::to_string(values.size());
    throw exception::not_supported(msg);
  }
}

} /* namespace nntrainer */
