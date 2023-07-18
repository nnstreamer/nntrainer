// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   attention_layer.cpp
 * @date   1 October 2021
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is Attention Layer Class for Neural Network
 *
 */

#include <cmath>

#include <attention_layer.h>
#include <layer_context.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>

namespace nntrainer {

AttentionLayer::AttentionLayer() :
  attention_props(props::ScaledDotProduct(), props::CausalMask()),
  sm(ActivationType::ACT_SOFTMAX) {
  wt_idx.fill(std::numeric_limits<unsigned>::max());
}

AttentionLayer::~AttentionLayer() {}

static constexpr size_t SINGLE_INOUT_IDX = 0;

enum AttentionParams { query = 0, value = 1, key = 2, weights };

void AttentionLayer::finalizeCommon(InitLayerContext &context) {
  if (context.getNumInputs() < 2 || context.getNumInputs() > 3)
    throw std::runtime_error("Attention layer needs 2-3 inputs.");

  auto const &all_dims = context.getInputDimensions();
  auto const &query_dim = all_dims[AttentionParams::query];
  auto const &value_dim = all_dims[AttentionParams::value];

  NNTR_THROW_IF(query_dim.width() != value_dim.width(), std::invalid_argument)
    << "Query and Value dimension mismatch for layer " << context.getName();

  wt_idx[AttentionParams::query] = AttentionParams::query;
  wt_idx[AttentionParams::value] = AttentionParams::value;
  wt_idx[AttentionParams::key] = AttentionParams::value;

  if (context.getNumInputs() == 3) {
    auto const &key_dim = all_dims[AttentionParams::key];
    if (key_dim != value_dim)
      throw std::invalid_argument("Key and value must have same shape");

    wt_idx[AttentionParams::key] = AttentionParams::key;
  }
}

void AttentionLayer::finalize(InitLayerContext &context) {
  finalizeCommon(context);

  auto const &all_dims = context.getInputDimensions();
  auto const &query_dim = all_dims[AttentionParams::query];
  auto const &value_dim = all_dims[AttentionParams::value];

  auto weights_dim = query_dim;
  weights_dim.width(value_dim.height());
  wt_idx[AttentionParams::weights] =
    context.requestTensor(weights_dim, "weights", Tensor::Initializer::NONE,
                          false, TensorLifespan::ITERATION_LIFESPAN);

  context.setOutputDimensions({query_dim});
}

void AttentionLayer::forwarding(RunLayerContext &context, bool training) {
  Tensor &query = context.getInput(wt_idx[AttentionParams::query]);
  Tensor &value = context.getInput(wt_idx[AttentionParams::value]);
  Tensor &key = context.getInput(wt_idx[AttentionParams::key]);

  Tensor &output = context.getOutput(SINGLE_INOUT_IDX);
  Tensor &weights = context.getTensor(wt_idx[AttentionParams::weights]);

  query.dotBatched(key, weights, false, true); /** dot 1 */
  if (std::get<props::ScaledDotProduct>(attention_props).get()) {
    weights.multiply_i(1 / sqrt((float)key.getDim().width()));
  }

  if (std::get<props::CausalMask>(attention_props).get()) {
    unsigned int mask_size = weights.getDim().width();
    unsigned int mask_dim_height = mask_size;
    unsigned int mask_dim_width = mask_size;

    Tensor causal_mask(TensorDim{mask_size, mask_size});

    causal_mask.setZero();
    for (unsigned int i = 0; i < mask_dim_height; ++i) {
      for (unsigned int j = i + 1; j < mask_dim_width; ++j) {
        causal_mask.setValue(0, 0, i, j, -1e10);
      }
    }

    weights.add_i(causal_mask);
  }

  sm.run_fn(weights, weights);       /** softmax */
  weights.dotBatched(value, output); /** dot 2 */
}

void AttentionLayer::incremental_forwarding(RunLayerContext &context,
                                            unsigned int from, unsigned int to,
                                            bool training) {
  Tensor &query = context.getInput(wt_idx[AttentionParams::query]);
  Tensor &value = context.getInput(wt_idx[AttentionParams::value]);
  Tensor &key = context.getInput(wt_idx[AttentionParams::key]);

  TensorDim query_dim = query.getDim();
  TensorDim value_dim = value.getDim();
  TensorDim key_dim = key.getDim();
  TensorDim query_step_dim = {query_dim.batch(), query_dim.channel(), to - from,
                              query_dim.width()};
  TensorDim value_step_dim = {value_dim.batch(), value_dim.channel(), to,
                              value_dim.width()};
  TensorDim key_step_dim = {key_dim.batch(), key_dim.channel(), to,
                            key_dim.width()};
  Tensor query_step =
    query.getSharedDataTensor(query_step_dim, from * query_dim.width(), true);
  Tensor value_step = value.getSharedDataTensor(value_step_dim, 0, true);
  Tensor key_step = key.getSharedDataTensor(key_step_dim, 0, true);

  Tensor &output = context.getOutput(SINGLE_INOUT_IDX);
  TensorDim output_dim = output.getDim();
  TensorDim output_step_dim = {output_dim.batch(), output_dim.channel(),
                               to - from, output_dim.width()};
  Tensor output_step = output.getSharedDataTensor(
    output_step_dim, from * output_dim.width(), true);

  Tensor &weights = context.getTensor(wt_idx[AttentionParams::weights]);
  TensorDim weights_dim = weights.getDim();
  TensorDim weights_step_dim = {
    query_step_dim.batch(), query_step_dim.channel(), query_step_dim.height(),
    value_step_dim.height()};
  Tensor weights_step = weights.getSharedDataTensor(
    weights_step_dim, from * weights_dim.width(), true);

  query_step.dotBatched(key_step, weights_step, false, true); /** dot 1 */
  if (std::get<props::ScaledDotProduct>(attention_props).get()) {
    weights_step.multiply_i(1 / sqrt((float)key.getDim().width()));
  }

  if (std::get<props::CausalMask>(attention_props).get() && !from) {
    unsigned int mask_size = weights_step.getDim().width();
    unsigned int mask_dim_height = mask_size;
    unsigned int mask_dim_width = mask_size;

    Tensor causal_mask(TensorDim{mask_size, mask_size});

    causal_mask.setZero();
    for (unsigned int i = 0; i < mask_dim_height; ++i) {
      for (unsigned int j = i + 1; j < mask_dim_width; ++j) {
        causal_mask.setValue(0, 0, i, j, -1e10);
      }
    }

    weights_step.add_i(causal_mask);
  }

  sm.run_fn(weights_step, weights_step);            /** softmax */
  weights_step.dotBatched(value_step, output_step); /** dot 2 */
}

void AttentionLayer::calcDerivative(RunLayerContext &context) {
  const Tensor &derivative = context.getIncomingDerivative(SINGLE_INOUT_IDX);

  Tensor &query = context.getInput(wt_idx[AttentionParams::query]);
  Tensor &value = context.getInput(wt_idx[AttentionParams::value]);
  Tensor &key = context.getInput(wt_idx[AttentionParams::key]);

  Tensor &dquery =
    context.getOutgoingDerivative(wt_idx[AttentionParams::query]);
  Tensor &dvalue =
    context.getOutgoingDerivative(wt_idx[AttentionParams::value]);
  Tensor &dkey = context.getOutgoingDerivative(wt_idx[AttentionParams::key]);

  Tensor &weights = context.getTensor(wt_idx[AttentionParams::weights]);

  Tensor dweight = Tensor(
    TensorDim({derivative.batch(), 1, derivative.height(), value.height()}));

  /** derivative for dot 2 */
  dweight.dot_batched_deriv_wrt_1(value, derivative);
  weights.dot_batched_deriv_wrt_2(dvalue, derivative);

  /** derivative for softmax */
  sm.run_prime_fn(weights, dweight, dweight);

  if (std::get<props::ScaledDotProduct>(attention_props).get()) {
    dweight.multiply_i(1 / sqrt((float)key.getDim().width()));
  }

  /** derivative for dot 1 */
  dquery.dot_batched_deriv_wrt_1(key, dweight, false, true);
  query.dot_batched_deriv_wrt_2(dkey, dweight, false, true,
                                context.getNumInputs() == 2);
}

void AttentionLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, attention_props);
  if (!remain_props.empty()) {
    std::string msg = "[AttentionLayer] Unknown Layer Properties count " +
                      std::to_string(values.size());
    throw exception::not_supported(msg);
  }
}

void AttentionLayer::setBatch(RunLayerContext &context, unsigned int batch) {
  context.updateTensor(wt_idx[AttentionParams::weights], batch);
}

} /* namespace nntrainer */
