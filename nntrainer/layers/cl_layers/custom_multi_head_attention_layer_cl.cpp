// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Debadri Samaddar <s.debadri@samsung.com>
 *
 * @file   custom_multi_head_attention_layer_cl.cpp
 * @date   25 Jun 2024
 * @see    https://github.com/nnstreamer/nntrainer
 *         https://arxiv.org/abs/1706.03762
 * @author Debadri Samaddar <s.debadri@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is CustomMultiHeadAttention Layer Class GPU execution
 *
 */

#include <algorithm>
#include <blas_kernel_interface.h>
#include <cmath>
#include <custom_multi_head_attention_layer_cl.h>
#include <layer_context.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <thread>
#include <vector>

namespace nntrainer {

CustomMultiHeadAttentionLayerCl::CustomMultiHeadAttentionLayerCl() :
  multi_head_attention_props(
    props::NumHeads(), props::ProjectedKeyDim(), props::ProjectedValueDim(),
    props::OutputShape(), props::DropOutRate(), props::ReturnAttentionWeight(),
    props::AverageAttentionWeight(), props::MaxTimestep(), props::SmartReply()),
  sm(ActivationType::ACT_SOFTMAX),
  epsilon(1e-3),
  cache_index(0) {
  weight_idx.fill(std::numeric_limits<unsigned>::max());
}

CustomMultiHeadAttentionLayerCl::~CustomMultiHeadAttentionLayerCl() {}

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
  cache_key,
  cache_value,
  /** intended comment for later use of attention_mask */
  // attention_mask,
  attention_weight,
  dropout_mask,
  attention_output,
};

void CustomMultiHeadAttentionLayerCl::finalize(
  nntrainer::InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() < 3 || context.getNumInputs() > 4,
                std::invalid_argument)
    << "Multi head Attention layer needs 3 or 4 inputs. (query, key, value and "
       "mask is optional";
  const bool provide_attention_mask = context.getNumInputs() == 4;

  ml::train::TensorDim::TensorType weight_type = {context.getFormat(),
                                                  context.getWeightDataType()};

  ml::train::TensorDim::TensorType activation_type = {
    context.getFormat(), context.getActivationDataType()};

  ml::train::TensorDim empty_dim(activation_type);

  const std::vector<ml::train::TensorDim> &input_dims =
    context.getInputDimensions();
  const ml::train::TensorDim &query_dim = input_dims[INOUT_INDEX::QUERY];
  const ml::train::TensorDim &key_dim = input_dims[INOUT_INDEX::KEY];
  const ml::train::TensorDim &value_dim = input_dims[INOUT_INDEX::VALUE];
  const ml::train::TensorDim &mask_dim =
    provide_attention_mask ? input_dims[INOUT_INDEX::MASK] : empty_dim;

  const unsigned int batch_size = query_dim.batch();
  const unsigned int query_height = query_dim.height();
  const unsigned int query_width = query_dim.width();
  // const unsigned int key_height = key_dim.height();
  const unsigned int key_width = key_dim.width();
  // const unsigned int value_height = value_dim.height();
  const unsigned int value_width = value_dim.width();

  const bool disable_bias =
    std::get<nntrainer::props::DisableBias>(*layer_impl_props).get();
  //  auto &weight_initializer =
  //    std::get<nntrainer::props::WeightInitializer>(*layer_impl_props).get();
  auto weight_initializer = nntrainer::props::InitializerInfo::Enum::ZEROS;
  auto &weight_regularizer =
    std::get<nntrainer::props::WeightRegularizer>(*layer_impl_props);
  auto &weight_regularizer_constant =
    std::get<nntrainer::props::WeightRegularizerConstant>(*layer_impl_props);
  const float &weight_decay =
    std::get<nntrainer::props::WeightDecay>(*layer_impl_props).get();

  NNTR_THROW_IF(
    std::get<nntrainer::props::NumHeads>(multi_head_attention_props).empty(),
    std::invalid_argument)
    << "num_heads property is not provided for layer " << context.getName();
  const unsigned int num_heads =
    std::get<nntrainer::props::NumHeads>(multi_head_attention_props).get();

  if (std::get<nntrainer::props::ProjectedKeyDim>(multi_head_attention_props)
        .empty()) {
    NNTR_THROW_IF(query_width % num_heads, std::invalid_argument)
      << "query_width: " << query_width
      << " is not divisible by num_heads: " << num_heads << " for layer "
      << context.getName();

    ml_logw("[multi head attention] ProjectedKeyDim property is not given. "
            "Default value(query_width / num_heads) is set");

    std::get<nntrainer::props::ProjectedKeyDim>(multi_head_attention_props)
      .set(query_width / num_heads);
  }
  const unsigned int projected_key_dim_prop =
    std::get<nntrainer::props::ProjectedKeyDim>(multi_head_attention_props)
      .get();

  if (std::get<nntrainer::props::ProjectedValueDim>(multi_head_attention_props)
        .empty()) {
    std::get<nntrainer::props::ProjectedValueDim>(multi_head_attention_props)
      .set(projected_key_dim_prop);
  }
  const unsigned int projected_value_dim_prop =
    std::get<nntrainer::props::ProjectedValueDim>(multi_head_attention_props)
      .get();

  if (std::get<nntrainer::props::OutputShape>(multi_head_attention_props)
        .empty()) {
    std::get<nntrainer::props::OutputShape>(multi_head_attention_props)
      .set(query_width);
  }
  const unsigned int output_shape =
    std::get<nntrainer::props::OutputShape>(multi_head_attention_props).get();

  const float dropout_rate =
    std::get<nntrainer::props::DropOutRate>(multi_head_attention_props).get();

  if (std::get<nntrainer::props::AverageAttentionWeight>(
        multi_head_attention_props)
        .empty()) {
    std::get<nntrainer::props::AverageAttentionWeight>(
      multi_head_attention_props)
      .set(true);
  }
  const bool average_attention_weight =
    std::get<nntrainer::props::AverageAttentionWeight>(
      multi_head_attention_props)
      .get();

  const nntrainer::props::ReturnAttentionWeightInfo::Enum
    return_attention_weight = std::get<nntrainer::props::ReturnAttentionWeight>(
                                multi_head_attention_props)
                                .get();

  const unsigned int max_timestep =
    std::get<nntrainer::props::MaxTimestep>(multi_head_attention_props).get();

  // @todo: fix me
  const unsigned int key_height = max_timestep;
  const unsigned int value_height = max_timestep;

  const unsigned int projected_query_dim_prop = projected_key_dim_prop;

  if (activation_type.data_type == ml::train::TensorDim::DataType::FP32) {
    sm.setActiFunc(nntrainer::ActivationType::ACT_SOFTMAX);
  } else if (activation_type.data_type ==
             ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    sm.setActiFunc<_FP16>(nntrainer::ActivationType::ACT_SOFTMAX);
#else
    throw std::invalid_argument("Error: enable-fp16 is not enabled");
#endif
  }

  // sm.setActiFunc(ActivationType::ACT_SOFTMAX);

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
  ml::train::TensorDim query_fc_weight_dim(
    {1, 1, query_width, num_heads * projected_query_dim_prop}, weight_type);

  weight_idx[AttentionParams::query_fc_weight] = context.requestWeight(
    query_fc_weight_dim, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, "query_fc_weight", true);
  if (!disable_bias) {
    ml::train::TensorDim query_fc_bias_dim(
      {1, 1, 1, num_heads * projected_query_dim_prop}, weight_type);
    weight_idx[AttentionParams::query_fc_bias] = context.requestWeight(
      query_fc_bias_dim, weight_initializer, weight_regularizer,
      weight_regularizer_constant, weight_decay, "query_fc_bias", true);
  }

  /** weight/bias for key fc */
  ml::train::TensorDim key_fc_weight_dim(
    {1, 1, key_width, num_heads * projected_key_dim_prop}, weight_type);
  weight_idx[AttentionParams::key_fc_weight] = context.requestWeight(
    key_fc_weight_dim, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, "key_fc_weight", true);
  if (!disable_bias) {
    ml::train::TensorDim key_fc_bias_dim(
      {1, 1, 1, num_heads * projected_key_dim_prop}, weight_type);
    weight_idx[AttentionParams::key_fc_bias] = context.requestWeight(
      key_fc_bias_dim, weight_initializer, weight_regularizer,
      weight_regularizer_constant, weight_decay, "key_fc_bias", true);
  }

  /** weight/bias for value fc */
  ml::train::TensorDim value_fc_weight_dim(
    {1, 1, value_width, num_heads * projected_value_dim_prop}, weight_type);
  weight_idx[AttentionParams::value_fc_weight] = context.requestWeight(
    value_fc_weight_dim, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, "value_fc_weight", true);
  if (!disable_bias) {
    ml::train::TensorDim value_fc_bias_dim(
      {1, 1, 1, num_heads * projected_value_dim_prop}, weight_type);
    weight_idx[AttentionParams::value_fc_bias] = context.requestWeight(
      value_fc_bias_dim, weight_initializer, weight_regularizer,
      weight_regularizer_constant, weight_decay, "value_fc_bias", true);
  }

  /** weight/bias for out fc */
  ml::train::TensorDim fc_weight_dim(
    {1, 1, num_heads * projected_value_dim_prop, output_shape}, weight_type);
  weight_idx[AttentionParams::fc_weight] = context.requestWeight(
    fc_weight_dim, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, "fc_weight", true);
  if (!disable_bias) {
    ml::train::TensorDim fc_bias_dim({1, 1, 1, output_shape}, weight_type);
    weight_idx[AttentionParams::fc_bias] = context.requestWeight(
      fc_bias_dim, weight_initializer, weight_regularizer,
      weight_regularizer_constant, weight_decay, "fc_bias", true);
  }

  /** tensor for output of query fc */
  ml::train::TensorDim projected_query_dim(
    {batch_size, 1, query_height, num_heads * projected_query_dim_prop},
    activation_type);
  weight_idx[AttentionParams::projected_query] =
    context.requestTensor(projected_query_dim, "projected_query",
                          nntrainer::Tensor::Initializer::NONE, false,
                          nntrainer::TensorLifespan::ITERATION_LIFESPAN);
  /** tensor for output of key fc */
  ml::train::TensorDim projected_key_dim(
    {batch_size, 1, key_height, num_heads * projected_key_dim_prop},
    activation_type);
  weight_idx[AttentionParams::projected_key] = context.requestTensor(
    projected_key_dim, "projected_key", nntrainer::Tensor::Initializer::NONE,
    false, nntrainer::TensorLifespan::ITERATION_LIFESPAN);
  /** tensor for output of value fc */
  ml::train::TensorDim projected_value_dim(
    {batch_size, 1, value_height, num_heads * projected_value_dim_prop},
    activation_type);
  weight_idx[AttentionParams::projected_value] =
    context.requestTensor(projected_value_dim, "projected_value",
                          nntrainer::Tensor::Initializer::NONE, false,
                          nntrainer::TensorLifespan::ITERATION_LIFESPAN);

  ml::train::TensorDim cache_key_dim(
    {batch_size, 1, max_timestep, num_heads * projected_key_dim_prop},
    activation_type);
  weight_idx[AttentionParams::cache_key] = context.requestTensor(
    cache_key_dim, "cache_key", nntrainer::Tensor::Initializer::NONE, false,
    nntrainer::TensorLifespan::MAX_LIFESPAN);

  ml::train::TensorDim cache_value_dim(
    {batch_size, 1, max_timestep, num_heads * projected_value_dim_prop},
    activation_type);
  weight_idx[AttentionParams::cache_value] = context.requestTensor(
    cache_value_dim, "cache_value", nntrainer::Tensor::Initializer::NONE, false,
    nntrainer::TensorLifespan::MAX_LIFESPAN);

  /** tensor for attention weight */
  ml::train::TensorDim attention_weight_dim(
    {batch_size, num_heads, query_height, key_height}, activation_type);
  weight_idx[AttentionParams::attention_weight] =
    context.requestTensor(attention_weight_dim, "attention_weight",
                          nntrainer::Tensor::Initializer::NONE, false,
                          nntrainer::TensorLifespan::ITERATION_LIFESPAN);
  if (dropout_rate > epsilon) {
    /** tensor for dropout mask */
    ml::train::TensorDim dropout_mask_dim(
      {batch_size, num_heads, query_height, key_height}, activation_type);
    weight_idx[AttentionParams::dropout_mask] = context.requestTensor(
      dropout_mask_dim, "dropout_mask", nntrainer::Tensor::Initializer::NONE,
      false, nntrainer::TensorLifespan::ITERATION_LIFESPAN);
  }

  /** tensor for attention output */
  ml::train::TensorDim attention_output_dim(
    {batch_size, 1, query_height, num_heads * projected_value_dim_prop},
    activation_type);
  weight_idx[AttentionParams::attention_output] =
    context.requestTensor(attention_output_dim, "attention_output",
                          nntrainer::Tensor::Initializer::NONE, false,
                          nntrainer::TensorLifespan::ITERATION_LIFESPAN);

  ml::train::TensorDim output_dim({batch_size, 1, query_height, output_shape},
                                  activation_type);
  if (return_attention_weight !=
      nntrainer::props::ReturnAttentionWeightInfo::Enum::none) {
    ml::train::TensorDim return_attention_weight_dim(
      {batch_size, average_attention_weight ? 1 : num_heads, query_height,
       key_height},
      activation_type);
    context.setOutputDimensions({output_dim, return_attention_weight_dim});
  } else {
    context.setOutputDimensions({output_dim});
  }

  /**
   * @todo
   * check query width and key width
   *
   */
  if (freqs_cos == nullptr)
    precompute_freqs(projected_key_dim_prop, max_timestep);
}

void CustomMultiHeadAttentionLayerCl::forwarding(
  nntrainer::RunLayerContext &context, bool training) {}

void CustomMultiHeadAttentionLayerCl::one_batch_incremental_forwarding(
  const unsigned int batch, const unsigned int _from, const unsigned int from,
  const unsigned int to, const unsigned int num_heads,
  const unsigned int projected_query_dim_prop,
  const unsigned int projected_key_dim_prop,
  const unsigned int projected_value_dim_prop, const bool disable_bias,
  const bool cache_shift, nntrainer::Tensor &query, nntrainer::Tensor &key,
  nntrainer::Tensor &value, nntrainer::Tensor &output,
  nntrainer::Tensor &query_fc_weight, nntrainer::Tensor &query_fc_bias,
  nntrainer::Tensor &key_fc_weight, nntrainer::Tensor &key_fc_bias,
  nntrainer::Tensor &value_fc_weight, nntrainer::Tensor &value_fc_bias,
  nntrainer::Tensor &fc_weight, nntrainer::Tensor &fc_bias,
  nntrainer::Tensor &projected_query, nntrainer::Tensor &projected_key,
  nntrainer::Tensor &projected_value, nntrainer::Tensor &cache_key,
  nntrainer::Tensor &cache_value, nntrainer::Tensor &attention_weight,
  nntrainer::Tensor &attention_output, ml::train::TensorDim &query_dim,
  ml::train::TensorDim &query_step_dim, ml::train::TensorDim &key_dim,
  ml::train::TensorDim &key_step_dim, ml::train::TensorDim &value_dim,
  ml::train::TensorDim &value_step_dim, ml::train::TensorDim &output_dim,
  ml::train::TensorDim &output_step_dim,
  ml::train::TensorDim &projected_query_dim,
  ml::train::TensorDim &projected_query_step_dim,
  ml::train::TensorDim &projected_key_dim,
  ml::train::TensorDim &projected_key_step_dim,
  ml::train::TensorDim &cache_key_dim, ml::train::TensorDim &cache_key_step_dim,
  ml::train::TensorDim &cached_key_dim,
  ml::train::TensorDim &projected_value_dim,
  ml::train::TensorDim &projected_value_step_dim,
  ml::train::TensorDim &cache_value_dim,
  ml::train::TensorDim &cache_value_step_dim,
  ml::train::TensorDim &cached_value_dim,
  ml::train::TensorDim &attention_weight_dim,
  ml::train::TensorDim &attention_weight_step_dim,
  ml::train::TensorDim &attention_output_dim,
  ml::train::TensorDim &attention_output_step_dim,
  nntrainer::RunLayerContext &context) {

  nntrainer::Tensor query_step = query.getSharedDataTensor(
    query_step_dim, batch * query_dim.getFeatureLen(), true);
  nntrainer::Tensor key_step = key.getSharedDataTensor(
    key_step_dim, batch * key_dim.getFeatureLen(), true);
  nntrainer::Tensor value_step = value.getSharedDataTensor(
    value_step_dim, batch * value_dim.getFeatureLen(), true);

  nntrainer::Tensor output_step = output.getSharedDataTensor(
    output_step_dim, batch * output_dim.getFeatureLen(), true);

  nntrainer::Tensor projected_query_step = projected_query.getSharedDataTensor(
    projected_query_step_dim, batch * projected_query_dim.getFeatureLen(),
    true);
  nntrainer::Tensor projected_key_step = projected_key.getSharedDataTensor(
    projected_key_step_dim, batch * projected_key_dim.getFeatureLen(), true);
  nntrainer::Tensor projected_value_step = projected_value.getSharedDataTensor(
    projected_value_step_dim, batch * projected_value_dim.getFeatureLen(),
    true);

  nntrainer::Tensor cache_key_step = cache_key.getSharedDataTensor(
    cache_key_step_dim,
    batch * cache_key_dim.getFeatureLen() + from * cache_key_dim.width(), true);
  nntrainer::Tensor cache_value_step = cache_value.getSharedDataTensor(
    cache_value_step_dim,
    batch * cache_value_dim.getFeatureLen() + from * cache_value_dim.width(),
    true);

  nntrainer::Tensor cached_key = cache_key.getSharedDataTensor(
    cached_key_dim, batch * cache_key_dim.getFeatureLen(), true);
  nntrainer::Tensor cached_value = cache_value.getSharedDataTensor(
    cached_value_dim, batch * cache_value_dim.getFeatureLen(), true);

  nntrainer::Tensor attention_weight_step =
    attention_weight.getSharedDataTensor(
      attention_weight_step_dim, batch * attention_weight_dim.getFeatureLen(),
      true);

  nntrainer::Tensor attention_output_step =
    attention_output.getSharedDataTensor(
      attention_output_step_dim, batch * attention_output_dim.getFeatureLen(),
      true);
  // to do: use BiQGEMM Openl kernel
  //////////////////////////////////////////////////////////
  // custom_dot(projected_query_step, query_fc_weight, query_step, from, to);
  // custom_dot(cache_key_step, key_fc_weight, key_step, from, to);
  // custom_dot(cache_value_step, value_fc_weight, value_step, from, to);
  //////////////////////////////////////////////////////////
  dotCl(query_step, query_fc_weight, projected_query_step, context);
  dotCl(key_step, key_fc_weight, cache_key_step, context);
  dotCl(value_step, value_fc_weight, cache_value_step, context);
  ////////////////////////////////////////////////////////////////
  if (!disable_bias) {
    add_i_cl(projected_query_step, query_fc_bias, context);
    add_i_cl(cache_key_step, key_fc_bias, context);
    add_i_cl(cache_value_step, value_fc_bias, context);
  }

  apply_rotary_emb_tensor(projected_query_step, projected_query_dim_prop,
                          _from);
  apply_rotary_emb_tensor(cache_key_step, projected_key_dim_prop, _from);

  projected_query_step.reshape(
    ml::train::TensorDim({1, to - from, num_heads, projected_query_dim_prop}));
  cached_key.reshape(
    ml::train::TensorDim({1, to, num_heads, projected_key_dim_prop}));
  cached_value.reshape(
    ml::train::TensorDim({1, to, num_heads, projected_value_dim_prop}));

  if (to - from != 1) {
    projected_query_step.transpose("1:0:2", projected_query_step);
  }
  cached_key.transpose("1:0:2", projected_key_step);
  cached_value.transpose("1:0:2", projected_value_step);

  projected_query_step.reshape(ml::train::TensorDim(
    {1 * num_heads, 1, to - from, projected_query_dim_prop}));
  projected_key_step.reshape(
    ml::train::TensorDim({1 * num_heads, 1, to, projected_key_dim_prop}));
  projected_value_step.reshape(
    ml::train::TensorDim({1 * num_heads, 1, to, projected_value_dim_prop}));

  attention_weight_step.reshape(
    ml::train::TensorDim({1 * num_heads, 1, to - from, to}));
  attention_output_step.reshape(ml::train::TensorDim(
    {1 * num_heads, 1, to - from, projected_value_dim_prop}));

  /** scaled dot product attention */
  dotBatchedCl(projected_query_step, projected_key_step, attention_weight_step,
               context, false, true);

  multiplyCl(attention_weight_step, 1 / sqrt((float)projected_query_dim_prop),
             context);

  if (!from) {
    unsigned int mask_size = attention_weight_step.getDim().width();
    unsigned int mask_dim_height = mask_size;
    unsigned int mask_dim_width = mask_size;

    nntrainer::Tensor causal_mask(ml::train::TensorDim{
      1, 1, mask_size, mask_size, attention_weight_step.getTensorType()});

    causal_mask.setZero();

#ifdef ENABLE_FP16
#define _MASK_NUM -1e4
#else
#define _MASK_NUM -1e10
#endif

    for (unsigned int i = 0; i < mask_dim_height; ++i) {
      for (unsigned int j = i + 1; j < mask_dim_width; ++j) {
        causal_mask.setValue(0, 0, i, j, _MASK_NUM);
      }
    }

    add_i_cl(attention_weight_step, causal_mask, context);
  }

  sm.run_fn(attention_weight_step, attention_weight_step);

  dotBatchedCl(attention_weight_step, projected_value_step,
               attention_output_step, context);

  if (to - from != 1) {
    attention_output_step.reshape(ml::train::TensorDim(
      {1, num_heads, to - from, projected_value_dim_prop}));

    attention_output_step = attention_output_step.transpose("1:0:2");
  }

  // to do: use BiQGEMM
  /////////////////////////////////////////////////
  // if ((fc_weight.getDataType() == nntrainer::TensorDim::DataType::BCQ16) ||
  //     (fc_weight.getDataType() == nntrainer::TensorDim::DataType::BCQ32)) {
  //   attention_output_step.reshape(ml::train::TensorDim(
  //     {1, 1, 1 * (to - from), num_heads * projected_value_dim_prop}));
  // } else {
  //   attention_output_step.reshape(ml::train::TensorDim(
  //     {1 * (to - from), 1, 1, num_heads * projected_value_dim_prop}));
  // }

  // custom_dot(output_step, fc_weight, attention_output_step, from, to);
  ////////////////// //////////////////////////////////////
  attention_output_step.reshape(
    TensorDim({1 * (to - from), 1, 1, num_heads * projected_value_dim_prop}));

  dotCl(attention_output_step, fc_weight, output_step, context);

  if (!disable_bias) {
    add_i_cl(output_step, fc_bias, context);
  }

  if (cache_shift) {
    if (cache_key.getDataType() == ml::train::TensorDim::DataType::FP32) {
      float *buf = cache_key.getAddress<float>(batch, 0, 1, 0);
      float *dbuf = cache_key.getAddress<float>(batch, 0, 0, 0);
      memcpy(dbuf, buf,
             (cache_key.getDim().getFeatureLen() - cache_key.width()) *
               sizeof(float));
      buf = cache_value.getAddress<float>(batch, 0, 1, 0);
      dbuf = cache_value.getAddress<float>(batch, 0, 0, 0);
      memcpy(dbuf, buf,
             (cache_value.getDim().getFeatureLen() - cache_value.width()) *
               sizeof(float));
    } else if (cache_key.getDataType() ==
               ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16

      _FP16 *buf = cache_key.getAddress<_FP16>(batch, 0, 1, 0);
      _FP16 *dbuf = cache_key.getAddress<_FP16>(batch, 0, 0, 0);
      memcpy(dbuf, buf,
             (cache_key.getDim().getFeatureLen() - cache_key.width()) *
               sizeof(_FP16));
      buf = cache_value.getAddress<_FP16>(batch, 0, 1, 0);
      dbuf = cache_value.getAddress<_FP16>(batch, 0, 0, 0);
      memcpy(dbuf, buf,
             (cache_key.getDim().getFeatureLen() - cache_value.width()) *
               sizeof(_FP16));
#else
      throw std::invalid_argument("enable-fp16 is not set");
#endif
    }
  }
}

void CustomMultiHeadAttentionLayerCl::incremental_forwarding(
  nntrainer::RunLayerContext &context, unsigned int _from, unsigned int _to,
  bool training) {

  if (_from && (_to - _from != 1)) {
    throw std::invalid_argument(
      "if it is not initial forwarding, then step size(difference between to "
      "and from) should be 1");
  }

  unsigned int max_timestep =
    std::get<nntrainer::props::MaxTimestep>(multi_head_attention_props).get();

  bool cache_shift = false;
  unsigned int from = _from;
  unsigned int to = _to;

  if (to >= max_timestep) {
    if (!_from) {
      throw std::invalid_argument(
        "to shouldn't greater than max_timestep for initial forwarding");
    } else {
      cache_shift = true;
      from = max_timestep - 1;
      to = max_timestep;
    }
  }

  const bool disable_bias =
    std::get<nntrainer::props::DisableBias>(*layer_impl_props).get();

  const unsigned int num_heads =
    std::get<nntrainer::props::NumHeads>(multi_head_attention_props).get();
  const unsigned int projected_key_dim_prop =
    std::get<nntrainer::props::ProjectedKeyDim>(multi_head_attention_props)
      .get();
  const unsigned int projected_value_dim_prop =
    std::get<nntrainer::props::ProjectedValueDim>(multi_head_attention_props)
      .get();

  const unsigned int projected_query_dim_prop = projected_key_dim_prop;
  //   const bool enable_dropout = dropout_rate > epsilon;

  auto get_step_dim = [to, from](const ml::train::TensorDim &dim) {
    auto step_dim = dim;
    step_dim.batch(1);
    step_dim.height(to - from);
    return step_dim;
  };

  /** get inputs/outputs */
  nntrainer::Tensor &query = context.getInput(INOUT_INDEX::QUERY);
  nntrainer::Tensor &key = context.getInput(INOUT_INDEX::KEY);
  nntrainer::Tensor &value = context.getInput(INOUT_INDEX::VALUE);

  nntrainer::Tensor empty_tensor;

  empty_tensor.setTensorType(value.getTensorType());

  ml::train::TensorDim query_dim = query.getDim();
  ml::train::TensorDim key_dim = key.getDim();
  ml::train::TensorDim value_dim = value.getDim();

  ml::train::TensorDim query_step_dim = get_step_dim(query_dim);
  ml::train::TensorDim key_step_dim = get_step_dim(key_dim);
  ml::train::TensorDim value_step_dim = get_step_dim(value_dim);

  nntrainer::Tensor &output = context.getOutput(INOUT_INDEX::OUTPUT);
  ml::train::TensorDim output_dim = output.getDim();
  ml::train::TensorDim output_step_dim = get_step_dim(output_dim);

  /** get weights */
  nntrainer::Tensor qWeight, kWeight, vWeight, fcWeight, qbias, kbias, vbias,
    bias;
  nntrainer::Tensor &query_fc_weight = qWeight;
  nntrainer::Tensor &key_fc_weight = kWeight;
  nntrainer::Tensor &value_fc_weight = vWeight;
  nntrainer::Tensor &fc_weight = fcWeight;
  nntrainer::Tensor &query_fc_bias = qbias;
  nntrainer::Tensor &key_fc_bias = kbias;
  nntrainer::Tensor &value_fc_bias = vbias;
  nntrainer::Tensor &fc_bias = bias;

  context.getWeight(query_fc_weight,
                    weight_idx[AttentionParams::query_fc_weight]);
  context.getWeight(key_fc_weight, weight_idx[AttentionParams::key_fc_weight]);
  context.getWeight(value_fc_weight,
                    weight_idx[AttentionParams::value_fc_weight]);

  context.getWeight(fc_weight, weight_idx[AttentionParams::fc_weight]);

  if (!disable_bias) {
    context.getWeight(query_fc_bias,
                      weight_idx[AttentionParams::query_fc_bias]);
    context.getWeight(key_fc_bias, weight_idx[AttentionParams::key_fc_bias]);
    context.getWeight(value_fc_bias,
                      weight_idx[AttentionParams::value_fc_bias]);
    context.getWeight(fc_bias, weight_idx[AttentionParams::fc_bias]);
  }

  /** get tensors */
  nntrainer::Tensor &projected_query =
    context.getTensor(weight_idx[AttentionParams::projected_query]);
  nntrainer::Tensor &projected_key =
    context.getTensor(weight_idx[AttentionParams::projected_key]);
  nntrainer::Tensor &projected_value =
    context.getTensor(weight_idx[AttentionParams::projected_value]);
  nntrainer::Tensor &cache_key =
    context.getTensor(weight_idx[AttentionParams::cache_key]);
  nntrainer::Tensor &cache_value =
    context.getTensor(weight_idx[AttentionParams::cache_value]);

  ml::train::TensorDim projected_query_dim = projected_query.getDim();
  ml::train::TensorDim projected_key_dim = projected_key.getDim();
  ml::train::TensorDim projected_value_dim = projected_value.getDim();
  ml::train::TensorDim cache_key_dim = cache_key.getDim();
  ml::train::TensorDim cache_value_dim = cache_value.getDim();

  ml::train::TensorDim projected_query_step_dim =
    get_step_dim(projected_query_dim);
  ml::train::TensorDim projected_key_step_dim = get_step_dim(projected_key_dim);
  ml::train::TensorDim projected_value_step_dim =
    get_step_dim(projected_value_dim);

  ml::train::TensorDim cache_key_step_dim = get_step_dim(cache_key_dim);
  ml::train::TensorDim cache_value_step_dim = get_step_dim(cache_value_dim);
  projected_key_step_dim.height(to);
  projected_value_step_dim.height(to);

  ml::train::TensorDim cached_key_dim = get_step_dim(cache_key_dim);
  ml::train::TensorDim cached_value_dim = get_step_dim(cache_value_dim);
  cached_key_dim.height(to);
  cached_value_dim.height(to);

  nntrainer::Tensor &attention_weight =
    context.getTensor(weight_idx[AttentionParams::attention_weight]);
  nntrainer::Tensor &attention_output =
    context.getTensor(weight_idx[AttentionParams::attention_output]);
  ml::train::TensorDim attention_weight_dim = attention_weight.getDim();

  ml::train::TensorDim attention_weight_step_dim =
    get_step_dim(attention_weight_dim);
  attention_weight_step_dim.width(to);

  ml::train::TensorDim attention_output_dim = attention_output.getDim();
  ml::train::TensorDim attention_output_step_dim =
    get_step_dim(attention_output_dim);

  unsigned int batch_size = query_dim.batch();

  bool smart_reply =
    std::get<props::SmartReply>(multi_head_attention_props).get();

  unsigned int b_size = batch_size;
  if (smart_reply && !_from) {
    b_size = 1;
  }

  for (unsigned int batch = 0; batch < b_size; ++batch) {
    one_batch_incremental_forwarding(
      batch, _from, from, to, num_heads, projected_query_dim_prop,
      projected_key_dim_prop, projected_value_dim_prop, disable_bias,
      cache_shift, query, key, value, output, query_fc_weight, query_fc_bias,
      key_fc_weight, key_fc_bias, value_fc_weight, value_fc_bias, fc_weight,
      fc_bias, projected_query, projected_key, projected_value, cache_key,
      cache_value, attention_weight, attention_output, query_dim,
      query_step_dim, key_dim, key_step_dim, value_dim, value_step_dim,
      output_dim, output_step_dim, projected_query_dim,
      projected_query_step_dim, projected_key_dim, projected_key_step_dim,
      cache_key_dim, cache_key_step_dim, cached_key_dim, projected_value_dim,
      projected_value_step_dim, cache_value_dim, cache_value_step_dim,
      cached_value_dim, attention_weight_dim, attention_weight_step_dim,
      attention_output_dim, attention_output_step_dim, context);
  }

  // copying KV cache internally
  if (!_from) {
    nntrainer::Tensor cache_key_0_step =
      cache_key.getSharedDataTensor(cache_key_step_dim, 0, true);
    nntrainer::Tensor cache_value_0_step =
      cache_value.getSharedDataTensor(cache_value_step_dim, 0, true);

    for (unsigned int batch = 1; batch < batch_size; ++batch) {
      nntrainer::Tensor cache_key_nth_step = cache_key.getSharedDataTensor(
        cache_key_step_dim,
        batch * cache_key_dim.getFeatureLen() + from * cache_key_dim.width(),
        true);
      nntrainer::Tensor cache_value_nth_step = cache_value.getSharedDataTensor(
        cache_value_step_dim,
        batch * cache_value_dim.getFeatureLen() +
          from * cache_value_dim.width(),
        true);

      cache_key_nth_step.copyData(cache_key_0_step);
      cache_value_nth_step.copyData(cache_value_0_step);
    }
  }
}

void CustomMultiHeadAttentionLayerCl::calcCommonDerivative(
  nntrainer::RunLayerContext &context) {}

void CustomMultiHeadAttentionLayerCl::calcDerivative(
  nntrainer::RunLayerContext &context) {}

void CustomMultiHeadAttentionLayerCl::calcGradient(
  nntrainer::RunLayerContext &context) {}

void CustomMultiHeadAttentionLayerCl::setProperty(
  const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, multi_head_attention_props);
  LayerImpl::setProperty(remain_props);
}

void CustomMultiHeadAttentionLayerCl::setBatch(
  nntrainer::RunLayerContext &context, unsigned int batch) {
  const float dropout_rate =
    std::get<nntrainer::props::DropOutRate>(multi_head_attention_props).get();

  context.updateTensor(weight_idx[AttentionParams::projected_query], batch);
  context.updateTensor(weight_idx[AttentionParams::projected_key], batch);
  context.updateTensor(weight_idx[AttentionParams::projected_value], batch);
  context.updateTensor(weight_idx[AttentionParams::cache_key], batch);
  context.updateTensor(weight_idx[AttentionParams::cache_value], batch);
  context.updateTensor(weight_idx[AttentionParams::attention_weight], batch);
  if (dropout_rate > epsilon) {
    context.updateTensor(weight_idx[AttentionParams::dropout_mask], batch);
  }
  context.updateTensor(weight_idx[AttentionParams::attention_output], batch);
}

void CustomMultiHeadAttentionLayerCl::exportTo(
  nntrainer::Exporter &exporter, const ml::train::ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
  exporter.saveResult(multi_head_attention_props, method, this);
}

} // namespace nntrainer
