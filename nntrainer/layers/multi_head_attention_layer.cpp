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
    props::OutputShape(), props::DropOutRate(), props::ReturnAttentionWeight(),
    props::AverageAttentionWeight()),
  epsilon(1e-3) {
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
  projected_key,
  projected_value,
  attention_score,
  d_attention_score,
  /** intended comment for later use of attention_mask */
  // attention_mask,
  attention_weight,
  dropout_mask,
  attention_output,
};

void MultiHeadAttentionLayer::finalize(InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() < 3 || context.getNumInputs() > 4,
                std::invalid_argument)
    << "Multi head Attention layer needs 3 or 4 inputs. (query, key, value and "
       "mask is optional";
  const bool provide_attention_mask = context.getNumInputs() == 4;

  TensorDim empty_dim;

  const std::vector<TensorDim> &input_dims = context.getInputDimensions();
  const TensorDim &query_dim = input_dims[INOUT_INDEX::QUERY];
  const TensorDim &key_dim = input_dims[INOUT_INDEX::KEY];
  const TensorDim &value_dim = input_dims[INOUT_INDEX::VALUE];
  const TensorDim &mask_dim =
    provide_attention_mask ? input_dims[INOUT_INDEX::MASK] : empty_dim;

  const unsigned int batch_size = query_dim.batch();
  const unsigned int query_height = query_dim.height();
  const unsigned int query_width = query_dim.width();
  const unsigned int key_height = key_dim.height();
  const unsigned int key_width = key_dim.width();
  const unsigned int value_height = value_dim.height();
  const unsigned int value_width = value_dim.width();

  const bool disable_bias =
    std::get<props::DisableBias>(*layer_impl_props).get();
  auto &weight_initializer =
    std::get<props::WeightInitializer>(*layer_impl_props).get();
  auto &bias_initializer = std::get<props::BiasInitializer>(*layer_impl_props);
  auto &weight_regularizer =
    std::get<props::WeightRegularizer>(*layer_impl_props);
  auto &weight_regularizer_constant =
    std::get<props::WeightRegularizerConstant>(*layer_impl_props);
  const float &weight_decay =
    std::get<props::WeightDecay>(*layer_impl_props).get();
  const float &bias_decay = std::get<props::BiasDecay>(*layer_impl_props).get();

  NNTR_THROW_IF(std::get<props::NumHeads>(multi_head_attention_props).empty(),
                std::invalid_argument)
    << "num_heads property is not provided for layer " << context.getName();
  const unsigned int num_heads =
    std::get<props::NumHeads>(multi_head_attention_props).get();

  if (std::get<props::ProjectedKeyDim>(multi_head_attention_props).empty()) {
    NNTR_THROW_IF(query_width % num_heads, std::invalid_argument)
      << "query_width: " << query_width
      << " is not divisible by num_heads: " << num_heads << " for layer "
      << context.getName();

    ml_logw("[multi head attention] ProjectedKeyDim property is not given. "
            "Default value(query_width / num_heads) is set");

    std::get<props::ProjectedKeyDim>(multi_head_attention_props)
      .set(query_width / num_heads);
  }
  const unsigned int projected_key_dim_prop =
    std::get<props::ProjectedKeyDim>(multi_head_attention_props).get();

  if (std::get<props::ProjectedValueDim>(multi_head_attention_props).empty()) {
    std::get<props::ProjectedValueDim>(multi_head_attention_props)
      .set(projected_key_dim_prop);
  }
  const unsigned int projected_value_dim_prop =
    std::get<props::ProjectedValueDim>(multi_head_attention_props).get();

  if (std::get<props::OutputShape>(multi_head_attention_props).empty()) {
    std::get<props::OutputShape>(multi_head_attention_props).set(query_width);
  }
  const unsigned int output_shape =
    std::get<props::OutputShape>(multi_head_attention_props).get();

  const float dropout_rate =
    std::get<props::DropOutRate>(multi_head_attention_props).get();

  if (std::get<props::AverageAttentionWeight>(multi_head_attention_props)
        .empty()) {
    std::get<props::AverageAttentionWeight>(multi_head_attention_props)
      .set(true);
  }
  const bool average_attention_weight =
    std::get<props::AverageAttentionWeight>(multi_head_attention_props).get();

  const props::ReturnAttentionWeightInfo::Enum return_attention_weight =
    std::get<props::ReturnAttentionWeight>(multi_head_attention_props).get();

  const unsigned int projected_query_dim_prop = projected_key_dim_prop;

  sm.setActiFunc(ActivationType::ACT_SOFTMAX);

  NNTR_THROW_IF(query_dim.channel() != 1, std::invalid_argument)
    << "Dimension of input query channel: " << query_dim.channel()
    << " is not 1 for layer " << context.getName();
  NNTR_THROW_IF(key_dim.channel() != 1, std::invalid_argument)
    << "Dimension of input key channel: " << key_dim.channel()
    << " is not 1 for layer " << context.getName();
  NNTR_THROW_IF(value_dim.channel() != 1, std::invalid_argument)
    << "Dimension of input value channel: " << value_dim.channel()
    << " is not 1 for layer " << context.getName();
  NNTR_THROW_IF(provide_attention_mask && mask_dim.channel() != num_heads,
                std::invalid_argument)
    << "Dimension of input mask channel: " << mask_dim.channel()
    << " is not matched with num_heads property: " << num_heads << " for layer "
    << context.getName();

  NNTR_THROW_IF(key_height != value_height, std::invalid_argument)
    << "Dimension of input key height: " << key_height
    << " is not matched with Dimension of input value height: " << value_height
    << " for layer " << context.getName();
  NNTR_THROW_IF(provide_attention_mask && mask_dim.height() != query_height,
                std::invalid_argument)
    << "Dimension of input mask height: " << mask_dim.height()
    << " is not matched with Dimension of input query height: " << query_height
    << " for layer " << context.getName();

  NNTR_THROW_IF(provide_attention_mask && mask_dim.width() != key_height,
                std::invalid_argument)
    << "Dimension of input mask width: " << mask_dim.width()
    << " is not matched with Dimension of input key height: " << key_height
    << " for layer " << context.getName();

  /** weight/bias for query fc */
  TensorDim query_fc_weight_dim(
    {1, 1, query_width, num_heads * projected_query_dim_prop});
  weight_idx[AttentionParams::query_fc_weight] = context.requestWeight(
    query_fc_weight_dim, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, "query_fc_weight", true);
  if (!disable_bias) {
    TensorDim query_fc_bias_dim(
      {1, 1, 1, num_heads * projected_query_dim_prop});
    weight_idx[AttentionParams::query_fc_bias] = context.requestWeight(
      query_fc_bias_dim, weight_initializer, weight_regularizer,
      weight_regularizer_constant, weight_decay, "query_fc_bias", true);
  }

  /** weight/bias for key fc */
  TensorDim key_fc_weight_dim(
    {1, 1, key_width, num_heads * projected_key_dim_prop});
  weight_idx[AttentionParams::key_fc_weight] = context.requestWeight(
    key_fc_weight_dim, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, "key_fc_weight", true);
  if (!disable_bias) {
    TensorDim key_fc_bias_dim({1, 1, 1, num_heads * projected_key_dim_prop});
    weight_idx[AttentionParams::key_fc_bias] = context.requestWeight(
      key_fc_bias_dim, weight_initializer, weight_regularizer,
      weight_regularizer_constant, weight_decay, "key_fc_bias", true);
  }

  /** weight/bias for value fc */
  TensorDim value_fc_weight_dim(
    {1, 1, value_width, num_heads * projected_value_dim_prop});
  weight_idx[AttentionParams::value_fc_weight] = context.requestWeight(
    value_fc_weight_dim, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, "value_fc_weight", true);
  if (!disable_bias) {
    TensorDim value_fc_bias_dim(
      {1, 1, 1, num_heads * projected_value_dim_prop});
    weight_idx[AttentionParams::value_fc_bias] = context.requestWeight(
      value_fc_bias_dim, weight_initializer, weight_regularizer,
      weight_regularizer_constant, weight_decay, "value_fc_bias", true);
  }

  /** weight/bias for out fc */
  TensorDim fc_weight_dim(
    {1, 1, num_heads * projected_value_dim_prop, output_shape});
  weight_idx[AttentionParams::fc_weight] = context.requestWeight(
    fc_weight_dim, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, "fc_weight", true);
  if (!disable_bias) {
    TensorDim fc_bias_dim({1, 1, 1, output_shape});
    weight_idx[AttentionParams::fc_bias] = context.requestWeight(
      fc_bias_dim, weight_initializer, weight_regularizer,
      weight_regularizer_constant, weight_decay, "fc_bias", true);
  }

  /** tensor for output of query fc */
  TensorDim projected_query_dim(
    {batch_size, 1, query_height, num_heads * projected_query_dim_prop});
  weight_idx[AttentionParams::projected_query] = context.requestTensor(
    projected_query_dim, "projected_query", Tensor::Initializer::NONE, true,
    TensorLifespan::ITERATION_LIFESPAN);
  /** tensor for output of key fc */
  TensorDim projected_key_dim(
    {batch_size, 1, key_height, num_heads * projected_key_dim_prop});
  weight_idx[AttentionParams::projected_key] = context.requestTensor(
    projected_key_dim, "projected_key", Tensor::Initializer::NONE, true,
    TensorLifespan::ITERATION_LIFESPAN);
  /** tensor for output of value fc */
  TensorDim projected_value_dim(
    {batch_size, 1, value_height, num_heads * projected_value_dim_prop});
  weight_idx[AttentionParams::projected_value] = context.requestTensor(
    projected_value_dim, "projected_value", Tensor::Initializer::NONE, true,
    TensorLifespan::ITERATION_LIFESPAN);

  /** tensor for attention score */
  TensorDim attention_score_dim(
    {batch_size, num_heads, query_height, key_height});
  weight_idx[AttentionParams::attention_score] = context.requestTensor(
    attention_score_dim, "attention_score", Tensor::Initializer::NONE, false,
    TensorLifespan::FORWARD_FUNC_LIFESPAN);
  weight_idx[AttentionParams::d_attention_score] = context.requestTensor(
    attention_score_dim, "d_attention_score", Tensor::Initializer::NONE, false,
    TensorLifespan::BACKWARD_FUNC_LIFESPAN);
  if (provide_attention_mask) {
    /** Intended comment for bool type mask */
    // TensorDim attention_mask_dim(
    //   {batch_size, num_heads, query_height, key_height});
    // weight_idx[AttentionParams::attention_mask] = context.requestTensor(
    //   attention_mask_dim, "attention_mask", Tensor::Initializer::NONE, false,
    //   TensorLifespan::FORWARD_FUNC_LIFESPAN);
  }
  /** tensor for attention weight */
  TensorDim attention_weight_dim(
    {batch_size, num_heads, query_height, key_height});
  weight_idx[AttentionParams::attention_weight] = context.requestTensor(
    attention_weight_dim, "attention_weight", Tensor::Initializer::NONE, true,
    TensorLifespan::ITERATION_LIFESPAN);
  if (dropout_rate > epsilon) {
    /** tensor for dropout mask */
    TensorDim dropout_mask_dim(
      {batch_size, num_heads, query_height, key_height});
    weight_idx[AttentionParams::dropout_mask] = context.requestTensor(
      dropout_mask_dim, "dropout_mask", Tensor::Initializer::NONE, false,
      TensorLifespan::ITERATION_LIFESPAN);
  }

  /** tensor for attention output */
  TensorDim attention_output_dim(
    {batch_size, 1, query_height, num_heads * projected_value_dim_prop});
  weight_idx[AttentionParams::attention_output] = context.requestTensor(
    attention_output_dim, "attention_output", Tensor::Initializer::NONE, true,
    TensorLifespan::ITERATION_LIFESPAN);

  TensorDim output_dim({batch_size, 1, query_height, output_shape});
  if (return_attention_weight != props::ReturnAttentionWeightInfo::Enum::none) {
    TensorDim return_attention_weight_dim(
      {batch_size, average_attention_weight ? 1 : num_heads, query_height,
       key_height});
    context.setOutputDimensions({output_dim, return_attention_weight_dim});
  } else {
    context.setOutputDimensions({output_dim});
  }
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
  context.updateTensor(weight_idx[AttentionParams::projected_key], batch);
  context.updateTensor(weight_idx[AttentionParams::projected_value], batch);
  context.updateTensor(weight_idx[AttentionParams::attention_score], batch);
  context.updateTensor(weight_idx[AttentionParams::d_attention_score], batch);
  context.updateTensor(weight_idx[AttentionParams::attention_weight], batch);
  if (dropout_rate > epsilon) {
    context.updateTensor(weight_idx[AttentionParams::dropout_mask], batch);
  }
  context.updateTensor(weight_idx[AttentionParams::attention_output], batch);
}

void MultiHeadAttentionLayer::exportTo(
  Exporter &exporter, const ml::train::ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
  exporter.saveResult(multi_head_attention_props, method, this);
}

} /* namespace nntrainer */
