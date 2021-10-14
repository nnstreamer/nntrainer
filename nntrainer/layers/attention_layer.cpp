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

enum AttentionParams { query = 0, value = 1, key = 2, score, weights };

void AttentionLayer::finalize(InitLayerContext &context) {
  if (context.getNumInputs() < 2 || context.getNumInputs() > 3)
    throw std::runtime_error("Attention layer needs 2-3 inputs.");

  sm.setActiFunc(ActivationType::ACT_SOFTMAX);

  auto const &all_dims = context.getInputDimensions();
  auto const &query_dim = all_dims[AttentionParams::query];
  auto const &value_dim = all_dims[AttentionParams::value];

  wt_idx[AttentionParams::query] = AttentionParams::query;
  wt_idx[AttentionParams::value] = AttentionParams::value;
  wt_idx[AttentionParams::key] = AttentionParams::value;

  if (context.getNumInputs() == 3) {
    auto const &key_dim = all_dims[AttentionParams::key];
    if (key_dim != value_dim)
      throw std::invalid_argument("Key and value must have same shape");

    wt_idx[AttentionParams::key] = AttentionParams::key;
  }

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
  Tensor &query = context.getInput(wt_idx[AttentionParams::query]);
  Tensor &value = context.getInput(wt_idx[AttentionParams::value]);
  Tensor &key = context.getInput(wt_idx[AttentionParams::key]);

  Tensor &output = context.getOutput(SINGLE_INOUT_IDX);
  Tensor &weights = context.getTensor(wt_idx[AttentionParams::weights]);
  Tensor &score = context.getTensor(wt_idx[AttentionParams::score]);

  for (unsigned int b = 0; b < query.batch(); b++) {
    /** @todo try using transpose to speedup the operation */
    Tensor query_b = query.getBatchSlice(b, 1);
    Tensor value_b = value.getBatchSlice(b, 1);
    Tensor key_b = key.getBatchSlice(b, 1);
    Tensor score_b = score.getBatchSlice(b, 1);
    Tensor weights_b = weights.getBatchSlice(b, 1);
    Tensor output_b = output.getBatchSlice(b, 1);

    query_b.dot(key_b, score_b, false, true);
    sm.run_fn(score_b, weights_b);
    weights_b.dot(value_b, output_b);
  }
}

void AttentionLayer::calcDerivative(RunLayerContext &context) {
  Tensor &derivative = context.getIncomingDerivative(SINGLE_INOUT_IDX);

  Tensor &query = context.getInput(wt_idx[AttentionParams::query]);
  Tensor &value = context.getInput(wt_idx[AttentionParams::value]);
  Tensor &key = context.getInput(wt_idx[AttentionParams::key]);

  Tensor &dquery =
    context.getOutgoingDerivative(wt_idx[AttentionParams::query]);
  Tensor &dvalue =
    context.getOutgoingDerivative(wt_idx[AttentionParams::value]);
  Tensor &dkey = context.getOutgoingDerivative(wt_idx[AttentionParams::key]);

  Tensor &weights = context.getTensor(wt_idx[AttentionParams::weights]);

  for (unsigned int b = 0; b < query.batch(); b++) {
    /** @todo try using transpose to speedup the operation */
    Tensor query_b = query.getBatchSlice(b, 1);
    Tensor value_b = value.getBatchSlice(b, 1);
    Tensor key_b = key.getBatchSlice(b, 1);
    Tensor weights_b = weights.getBatchSlice(b, 1);

    Tensor dquery_b = dquery.getBatchSlice(b, 1);
    Tensor dvalue_b = dvalue.getBatchSlice(b, 1);
    Tensor dkey_b = dkey.getBatchSlice(b, 1);
    Tensor deriv_b = derivative.getBatchSlice(b, 1);

    Tensor dweight = deriv_b.dot(value_b, false, true);

    Tensor t1;
    sm.run_prime_fn(weights_b, t1, dweight);
    t1.dot(key_b, dquery_b);

    weights_b.dot(deriv_b, dvalue_b, true, false);
    if (context.getNumInputs() == 2)
      t1.dot(query_b, dvalue_b, true, false, 1.0);
    else
      t1.dot(query_b, dkey_b, true, false);
  }
}

void AttentionLayer::setProperty(const std::vector<std::string> &values) {
  if (!values.empty()) {
    std::string msg = "[AttentionLayer] Unknown Layer Properties count " +
                      std::to_string(values.size());
    throw exception::not_supported(msg);
  }
}

void AttentionLayer::setBatch(RunLayerContext &context, unsigned int batch) {
  context.updateTensor(wt_idx[AttentionParams::score], batch);
  context.updateTensor(wt_idx[AttentionParams::weights], batch);
}

} /* namespace nntrainer */
