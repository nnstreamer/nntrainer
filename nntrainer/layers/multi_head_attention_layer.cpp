// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2022 hyeonseok Lee <hs89.lee@samsung.com>
 *
 * @file   multi_head_attention_layer.cpp
 * @date   08 July 2022
 * @see    https://github.com/nnstreamer/nntrainer
 *         https://arxiv.org/abs/1706.03762
 * @author hyeonseok Lee <hs89.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is MultiHeadAttention Layer Class for Neural Network
 *
 */

#include <cmath>

#include <layer_context.h>
#include <multi_head_attention_layer.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>

namespace nntrainer {

MultiHeadAttentionLayer::MultiHeadAttentionLayer() :
  multi_head_attention_props(
    props::NumHeads(), props::ProjectedKeyDim(), props::ProjectedValueDim(),
    props::OutputShape(), props::DropOutRate(), props::ProvideAttentionMask(),
    props::ReturnAttentionWeight(), props::AverageAttentionWeight()),
  epsilon(1e-3) {
  inout_idx.fill(std::numeric_limits<unsigned>::max());
  weight_idx.fill(std::numeric_limits<unsigned>::max());
}

MultiHeadAttentionLayer::~MultiHeadAttentionLayer() {}

enum INOUT_INDEX {
  /** input index */
  QUERY = 0,
  KEY = 1,
  VALUE = 2,
  MASK = 3,
  /** output index */
  OUTPUT = 0,
  RETURN_ATTENTION_WEIGHT = 1,
};

enum AttentionParams {
  query_fc_weight,
  query_fc_bias,
  key_fc_weight,
  key_fc_bias,
  value_fc_weight,
  value_fc_bias,
  fc_weight,
  fc_bias,
  projected_query,
  d_projected_query,
  projected_key,
  d_projected_key,
  projected_value,
  d_projected_value,
  transposed_projected_query,
  transposed_projected_key,
  transposed_projected_value,
  attention_score,
  d_attention_score,
  attention_mask,
  attention_weight,
  dropout_mask,
  attention_output,
  d_attention_output,
  transposed_attention_output,
};

void MultiHeadAttentionLayer::finalizeCommon(InitLayerContext &context) {}

void MultiHeadAttentionLayer::finalize(InitLayerContext &context) {
  finalizeCommon(context);
}

void MultiHeadAttentionLayer::forwarding(RunLayerContext &context,
                                         bool training) {}

void MultiHeadAttentionLayer::calcCommonDerivative(RunLayerContext &context) {}

void MultiHeadAttentionLayer::calcDerivative(RunLayerContext &context) {
  if (!context.getTrainable()) {
    calcCommonDerivative(context);
  }
}

void MultiHeadAttentionLayer::calcGradient(RunLayerContext &context) {
  calcCommonDerivative(context);
}

void MultiHeadAttentionLayer::setProperty(
  const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, multi_head_attention_props);
  LayerImpl::setProperty(remain_props);
}

void MultiHeadAttentionLayer::setBatch(RunLayerContext &context,
                                       unsigned int batch) {
  const float dropout_rate =
    std::get<props::DropOutRate>(multi_head_attention_props).get();

  context.updateTensor(weight_idx[AttentionParams::projected_query], batch);
  context.updateTensor(weight_idx[AttentionParams::d_projected_query], batch);
  context.updateTensor(weight_idx[AttentionParams::projected_key], batch);
  context.updateTensor(weight_idx[AttentionParams::d_projected_key], batch);
  context.updateTensor(weight_idx[AttentionParams::projected_value], batch);
  context.updateTensor(weight_idx[AttentionParams::d_projected_value], batch);
  context.updateTensor(weight_idx[AttentionParams::transposed_projected_query],
                       batch);
  context.updateTensor(weight_idx[AttentionParams::transposed_projected_key],
                       batch);
  context.updateTensor(weight_idx[AttentionParams::transposed_projected_value],
                       batch);
  context.updateTensor(weight_idx[AttentionParams::attention_score], batch);
  context.updateTensor(weight_idx[AttentionParams::d_attention_score], batch);
  context.updateTensor(weight_idx[AttentionParams::attention_weight], batch);
  if (dropout_rate > epsilon) {
    context.updateTensor(weight_idx[AttentionParams::dropout_mask], batch);
  }
  context.updateTensor(weight_idx[AttentionParams::attention_output], batch);
  context.updateTensor(weight_idx[AttentionParams::d_attention_output], batch);
  context.updateTensor(weight_idx[AttentionParams::transposed_attention_output],
                       batch);
}

void MultiHeadAttentionLayer::exportTo(
  Exporter &exporter, const ml::train::ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
  exporter.saveResult(multi_head_attention_props, method, this);
}

} /* namespace nntrainer */
