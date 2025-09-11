// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   mha_core.cpp
 * @date   11 July 2025
 * @see    https://github.com/nnstreamer/nntrainer
 *         https://arxiv.org/abs/1706.03762
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This code is based on custom_multi_head_attention_layer.cpp.
 *         This code is a part of the break down version of the mha layer.
 */
#include <algorithm>
#include <cmath>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <chrono>
#endif

#include <fp16.h>
#include <layer_context.h>
#include <mha_core.h>
#include <nntrainer_error.h>
#include <node_exporter.h>

#include <cassert>
#include <chrono>
#include <cstdint>
#include <type_traits>

inline float convert_scalar(uint16_t h) {
  return nntrainer::compute_fp16_to_fp32(h);
}

namespace causallm {

/************************************************************** */

/**
 * @brief constructor of MHACoreLayer
 */
MHACoreLayer::MHACoreLayer() :
  mha_core_props(
    nntrainer::props::NumHeads(), props::NumHeads_KV(),
    nntrainer::props::ProjectedKeyDim(), nntrainer::props::ProjectedValueDim(),
    nntrainer::props::OutputShape(), nntrainer::props::DropOutRate(),
    nntrainer::props::ReturnAttentionWeight(),
    nntrainer::props::AverageAttentionWeight(), nntrainer::props::MaxTimestep(),
    props::SlidingWindow(), props::MaxNewTokens(), props::RopeTheta(),
    props::MaxPositionEmbeddings()),
  sm(nntrainer::ActivationType::ACT_SOFTMAX),
  epsilon(1e-3),
  cache_index(0),
  num_heads_Q(0),
  num_heads_KV(0),
  head_dim(0),
  cache_shift(false) {
  weight_idx.fill(std::numeric_limits<unsigned>::max());
}

MHACoreLayer::~MHACoreLayer() {}

/************************************************************** */

void MHACoreLayer::finalize(nntrainer::InitLayerContext &context) {

  NNTR_THROW_IF(context.getNumInputs() < 3 || context.getNumInputs() > 4,
                std::invalid_argument)
    << "Multi head Attention layer needs 3 or 4 inputs. (query, key, value and "
       "mask is optional)";
  ml::train::TensorDim::TensorType activation_type = {
    context.getFormat(), context.getActivationDataType()};
  ml::train::TensorDim empty_dim(activation_type);

  const std::vector<ml::train::TensorDim> &input_dims =
    context.getInputDimensions();
  const ml::train::TensorDim &query_dim = input_dims[INOUT_INDEX::QUERY];
  const ml::train::TensorDim &key_dim = input_dims[INOUT_INDEX::KEY];

  /** max time step of this model */
  const unsigned int max_timestep =
    std::get<nntrainer::props::MaxTimestep>(mha_core_props).get();

  /** max position embeddings */
  const unsigned int max_position_embeddings =
    std::get<props::MaxPositionEmbeddings>(mha_core_props).get();

  /** query_dim = (B, 1, seq_len, H_Q * Head_Dim ) */
  const unsigned int batch_size = query_dim.batch();
  const unsigned int query_width = query_dim.width();
  /** key_dim = (B, 1, max_seq_len, H_KV * Head_Dim ) */
  const unsigned int key_width = key_dim.width();

  /**
   *  @note If NumHeads_KV is set, then use the value. Otherwise,
   *        we initialize num_heads_KV with num_heads_Q.
   */
  num_heads_Q = static_cast<size_t>(
    std::get<nntrainer::props::NumHeads>(mha_core_props).get());
  num_heads_KV =
    std::get<props::NumHeads_KV>(mha_core_props).empty()
      ? num_heads_Q
      : static_cast<size_t>(std::get<props::NumHeads_KV>(mha_core_props).get());

  // head_dim
  head_dim = static_cast<size_t>(query_width) / num_heads_Q;
  NNTR_THROW_IF(head_dim != key_width / num_heads_KV, std::invalid_argument)
    << "num_heads_Q and num_heads_KV are not properly given. Please check the "
       "num_heads_* are set correctly so that the `head_dim`s are all same for "
       "query / key / value";

  /** Tensor for KV-Cache */

  ml::train::TensorDim cache_key_dim(
    {batch_size, 1, max_timestep, num_heads_KV * head_dim},
    {context.getFormat(), ml::train::TensorDim::DataType::UINT16});
  ml::train::TensorDim cache_value_dim(
    {batch_size, 1, max_timestep, num_heads_KV * head_dim},
    {context.getFormat(), ml::train::TensorDim::DataType::UINT16});

  weight_idx[AttentionParams::cache_key] = context.requestTensor(
    cache_key_dim, "cache_key", nntrainer::Initializer::NONE, false,
    nntrainer::TensorLifespan::MAX_LIFESPAN);
  weight_idx[AttentionParams::cache_value] = context.requestTensor(
    cache_value_dim, "cache_value", nntrainer::Initializer::NONE, false,
    nntrainer::TensorLifespan::MAX_LIFESPAN);

  theta = (float)std::get<props::RopeTheta>(mha_core_props).get();

  /** precompute_freqs will be invoked only once */
  if (freqs_cos == nullptr)
    precompute_freqs(head_dim, max_position_embeddings, theta);

  /** set Output dimension! - one output */
  std::vector<nntrainer::TensorDim> output_dims(1);
  output_dims[0] = input_dims[0];
  output_dims[0].width(head_dim * num_heads_Q);
  output_dims[0].setTensorType(
    {context.getFormat(), context.getActivationDataType()});
  context.setOutputDimensions(output_dims);
}

/************************************************************** */

/**
 * @note This forwarding function is used for training mode.
 *       This will be implemented ASAP.
 * @date 2024-09-02
 */
void MHACoreLayer::forwarding(nntrainer::RunLayerContext &context,
                              bool training) {}

/**
 * @note This incremental_forwarding method is invoked for inference mode.
 *       Please note that Transformer Decoder's MHA takes only one sequence at a
 * step. Incremental forwarding function is used for this.
 */
void MHACoreLayer::incremental_forwarding(nntrainer::RunLayerContext &context,
                                          unsigned int _from, unsigned int _to,
                                          bool training) {

  if (_from && (_to - _from != 1)) {
    throw std::invalid_argument(
      "if it is not initial forwarding, then step size(difference between to "
      "and from) should be 1");
  }

  unsigned int max_timestep =
    std::get<nntrainer::props::MaxTimestep>(mha_core_props).get();

  unsigned int from = _from;
  unsigned int to = _to;

  if (to >= max_timestep) {
    // initial forwarding
    if (!_from) {
      throw std::invalid_argument(
        "to shouldn't greater than max_timestep for initial forwarding");
    } else {
      // exceeds the kv_cache size
      // KV_cache is shifted!
      cache_shift = true;
      from = max_timestep - 1;
      to = max_timestep;
    }
  }

  // util fn to compute tensor dimension for one step.
  auto get_step_dim = [to, from](const ml::train::TensorDim &dim) {
    auto step_dim = dim;
    step_dim.batch(1);
    step_dim.height(to - from); // One is expected.
    return step_dim;
  };

  /** incremental forwarding for each batch */
  nntrainer::Tensor &query =
    context.getInput(INOUT_INDEX::QUERY); // projected query
  nntrainer::Tensor &key = context.getInput(INOUT_INDEX::KEY); // projected key
  nntrainer::Tensor &value =
    context.getInput(INOUT_INDEX::VALUE); // projected value
  nntrainer::Tensor &output =
    context.getOutput(INOUT_INDEX::OUTPUT); // output to be projected

  nntrainer::Tensor &cache_key =
    context.getTensor(weight_idx[AttentionParams::cache_key]);
  nntrainer::Tensor &cache_value =
    context.getTensor(weight_idx[AttentionParams::cache_value]);

  const unsigned int num_heads_Q =
    std::get<nntrainer::props::NumHeads>(mha_core_props).get();

  ml::train::TensorDim query_dim =
    query.getDim(); // (B, 1, seq_len, n_heads_Q * head_dim)
  ml::train::TensorDim key_dim =
    key.getDim(); // (B, 1, seq_len, n_heads_KV * head_dim)
  ml::train::TensorDim value_dim =
    value.getDim(); // (B, 1, seq_len, n_heads_KV * head_dim)
  ml::train::TensorDim output_dim =
    output.getDim(); // (B, 1, seq_len, n_heads_Q * head_dim)
  ml::train::TensorDim cache_key_dim =
    cache_key.getDim(); // (B, 1, max_seq_len, n_heads_KV * head_dim)
  ml::train::TensorDim cache_value_dim =
    cache_value.getDim(); // (B, 1, max_seq_len, n_heads_KV * head_dim)

  ml::train::TensorDim query_step_dim =
    get_step_dim(query_dim); // (B, 1, from-to, n_heads_Q * head_dim)
  ml::train::TensorDim output_step_dim =
    get_step_dim(output_dim); // (B, 1, from-to, n_heads_Q * head_dim)
  ml::train::TensorDim cache_key_step_dim =
    get_step_dim(cache_key_dim); // (B, 1, from-to, n_heads_KV * head_dim)

  ml::train::TensorDim cache_value_step_dim =
    get_step_dim(cache_value_dim); // (B, 1, from-to, n_heads_KV * head_dim)

  unsigned int batch_size = (_from) ? 1 : query_dim.batch();
  // auto start_time = std::chrono::high_resolution_clock::now();
  // do the incremental forwarding
  for (unsigned int batch = 0; batch < batch_size; ++batch) {
    one_batch_incremental_forwarding(
      batch, _from, from, to, query, key, value, output, cache_key, cache_value,
      query_dim, query_step_dim, key_dim, value_dim, cache_key_dim,
      cache_key_step_dim, cache_value_dim, cache_value_step_dim, output_dim,
      output_step_dim);
  }
  if (!_from) {
    batch_size = query_dim.batch();
    nntrainer::Tensor cache_key_0_step =
      cache_key.getSharedDataTensor(cache_key_step_dim, 0, true);
    nntrainer::Tensor cache_value_0_step =
      cache_value.getSharedDataTensor(cache_value_step_dim, 0, true);

    for (unsigned int batch = 1; batch < batch_size; ++batch) {
      nntrainer::Tensor cache_key_nth_step = cache_key.getSharedDataTensor(
        cache_key_step_dim,
        batch * cache_key_dim.getFeatureLen() + from * cache_key_dim.width(),
        true);
      nntrainer::Tensor cache_value_nth_step =
        cache_key.getSharedDataTensor(cache_value_step_dim,
                                      batch * cache_value_dim.getFeatureLen() +
                                        from * cache_value_dim.width(),
                                      true);

      cache_key_nth_step.copyData(cache_key_0_step);
      cache_key_nth_step.copyData(cache_value_0_step);
    }
  }
}

void MHACoreLayer::compute_kcaches(
  nntrainer::Tensor &in, nntrainer::Tensor &cache, nntrainer::Tensor &out,
  unsigned int from, size_t sequence_len, unsigned int num_head,
  unsigned int group_size, unsigned int head_dim, BS::thread_pool<> &pool) {

  if (from) {
    nntrainer::compute_kcaches<uint16_t>(
      in.getData<float>(), cache.getData<uint16_t>(), out.getData<float>(),
      from + 1, num_head / group_size, head_dim, group_size, 16);
  } else {
    std::vector<std::future<void>> futures;
    for (unsigned int i = 0; i < sequence_len; ++i) {
      float *input_addr = in.getData<float>() + num_head * head_dim * i;
      uint16_t *cache_addr = cache.getData<uint16_t>();
      int row_to_compute = i + 1;
      size_t out_start_row = (i + 1) * i / 2;

      float *output_addr = out.getData<float>() + out_start_row * num_head;

      futures.emplace_back(pool.submit_task([=]() {
        nntrainer::compute_kcaches<uint16_t>(
          input_addr, cache_addr, output_addr, row_to_compute,
          num_head / group_size, head_dim, group_size, 16);
      }));
    }
    for (auto &fut : futures)
      fut.get();
  }
}

void MHACoreLayer::one_batch_incremental_forwarding(
  const unsigned int batch, const unsigned int _from, const unsigned int from,
  const unsigned int to, nntrainer::Tensor &query, nntrainer::Tensor &key,
  nntrainer::Tensor &value, nntrainer::Tensor &output,
  nntrainer::Tensor &cache_key, nntrainer::Tensor &cache_value,
  ml::train::TensorDim &query_dim, ml::train::TensorDim &query_step_dim,
  ml::train::TensorDim &key_dim, ml::train::TensorDim &value_dim,
  ml::train::TensorDim &cache_key_dim, ml::train::TensorDim &cache_key_step_dim,
  ml::train::TensorDim &cache_value_dim,
  ml::train::TensorDim &cache_value_step_dim, ml::train::TensorDim &output_dim,
  ml::train::TensorDim &output_step_dim) {

  /**
   *  cache_key
   *  +--------+                        ->
   *  |        |                        ->
   *  |        |                        ->
   *  |........| from                   ->
   *  |........| to -> b_cache_key_step -> b_cached_key
   *  |        |
   *  +--------+
   *
   */

  /** 1. Load Input Tensors of this batch : b_ denotes a Tensor for this batch
   * **/
  auto &pool = nntrainer::ThreadPoolManager::Global().getThreadPool();

  std::vector<std::future<void>> p_futures;

  nntrainer::Tensor b_projected_query_step = query.getSharedDataTensor(
    query_step_dim, batch * query_dim.getFeatureLen(), true);

  apply_rotary_emb_tensor_v2(b_projected_query_step, b_projected_query_step,
                             head_dim, _from, false);
  nntrainer::Tensor b_cache_key_step = cache_key.getSharedDataTensor(
    cache_key_step_dim,
    batch * cache_key_dim.getFeatureLen() + from * cache_key_dim.width(), true);

  ml::train::TensorDim key_step_dim = key.getDim();
  key_step_dim.height(to - from);

  nntrainer::Tensor key_step = key.getSharedDataTensor(
    key_step_dim, batch * key_dim.getFeatureLen(), true);

  apply_rotary_emb_tensor_v2(key_step, b_cache_key_step, head_dim, _from,
                             false);

  nntrainer::Tensor b_cache_value_step = cache_value.getSharedDataTensor(
    cache_value_step_dim,
    batch * cache_value_dim.getFeatureLen() + from * cache_value_dim.width(),
    true);

  nntrainer::Tensor value_step = value.getSharedDataTensor(
    key_step_dim, batch * value_dim.getFeatureLen(), true);

  apply_rotary_emb_tensor_v2(value_step, b_cache_value_step, head_dim, _from,
                             true);
  ml::train::TensorDim cached_key_dim = cache_key_dim;
  ml::train::TensorDim cached_value_dim = cache_value_dim;
  cached_key_dim.height(to);
  cached_value_dim.height(to);

  nntrainer::Tensor b_cached_key = cache_key.getSharedDataTensor(
    cached_key_dim, batch * cache_key_dim.getFeatureLen(), true);
  nntrainer::Tensor b_cached_value = cache_value.getSharedDataTensor(
    cached_value_dim, batch * cache_value_dim.getFeatureLen(), true);

  nntrainer::Tensor attention_output_step = output.getSharedDataTensor(
    output_step_dim, batch * output_dim.getFeatureLen(), true);

  nntrainer::Tensor out_(1, 1,
                         (from) ? to : ((to - from) * (to - from + 1) / 2),
                         num_heads_Q, b_projected_query_step.getTensorType());

  unsigned int gqa_size = num_heads_Q / num_heads_KV;

  compute_kcaches(b_projected_query_step, b_cached_key, out_, _from, to - from,
                  num_heads_Q, gqa_size, head_dim, pool);

  softmax_triangle(out_, to - from, num_heads_Q, from, pool);

  compute_fp16vcache_fp32_transposed(
    out_.getData<float>(), b_cached_value.getData<uint16_t>(),
    attention_output_step.getData<float>(), to, num_heads_KV, gqa_size,
    head_dim, (from) ? false : true, pool);
}

/************************************************************** */

/**
 * @brief rotary embedding-related member function
 * @note seq_len -> max_position_embeddings
 */
void MHACoreLayer::precompute_freqs(int head_dim, unsigned int seq_len,
                                    float theta) {
  // compute the freqs only when it is the first time to call this function
  if (freqs_cos != nullptr && freqs_cos->size() == seq_len)
    return;

  // theta_i = 10000^(-2(i-1)/dim) for i = [1, 2, ... , dim/2]
  // head_dim should be divisible by 2
  unsigned int half_ = head_dim / 2;
  for (unsigned int i = 0; i < half_; ++i) {
    thetas.push_back(1.0 /
                     (std::pow(theta, (2 * i) / static_cast<float>(head_dim))));
  }

  // cos / sin
  auto cos = new std::vector<std::vector<float>>();
  cos->assign(seq_len, std::vector<float>(head_dim, 0));
  auto sin = new std::vector<std::vector<float>>();
  sin->assign(seq_len, std::vector<float>(head_dim, 0));

  // update cos / sin frequency
  for (unsigned int i = 0; i < seq_len; ++i) {

#ifdef USE_NEON
    nntrainer::calc_trigonometric_vals_dup(
      half_, thetas.data(), (*cos)[i].data(), (*sin)[i].data(), i);
#else
    for (unsigned int j = 0; j < half_; ++j) {
      float angle = i * thetas[j];
      (*cos)[i][j] = std::cos(angle);
      (*cos)[i][j + half_] = std::cos(angle); // repeated 2 times

      (*sin)[i][j] = std::sin(angle);
      (*sin)[i][j + half_] = std::sin(angle); // repeated 2 times
    }
#endif
  }
  freqs_cos = cos;
  freqs_sin = sin;
};

void MHACoreLayer::apply_rotary_emb_tensor_v2(nntrainer::Tensor &in,
                                              nntrainer::Tensor &out,
                                              unsigned int dim,
                                              unsigned int from,
                                              bool convert_only) {
  unsigned int half_ = dim / 2;
  unsigned int max_timestep =
    std::get<nntrainer::props::MaxTimestep>(mha_core_props).get();

  std::vector<float> *cos_ = nullptr;
  std::vector<float> *sin_ = nullptr;

  for (unsigned int b = 0; b < in.batch(); b++) {
    for (unsigned int c = 0; c < in.channel(); c++) {
      for (unsigned int h = 0; h < in.height(); h++) {
        if (from < max_timestep) {
          cos_ = &(*freqs_cos)[from + h];
          sin_ = &(*freqs_sin)[from + h];
        }
        float *in_ptr = in.getData<float>() +
                        b * in.channel() * in.height() * in.width() +
                        c * in.height() * in.width() + h * in.width();

        if (out.getDataType() == ml::train::TensorDim::DataType::FP32) {

          nntrainer::compute_rotary_emb_value(in.width(), dim, half_, in_ptr,
                                              nullptr, cos_->data(),
                                              sin_->data(), convert_only);
        } else if (out.getDataType() ==
                   ml::train::TensorDim::DataType::UINT16) {
          uint16_t *out_ptr = out.getData<uint16_t>() +
                              b * out.channel() * out.height() * out.width() +
                              c * out.height() * out.width() + h * out.width();

          nntrainer::compute_rotary_emb_value(in.width(), dim, half_, in_ptr,
                                              out_ptr, cos_->data(),
                                              sin_->data(), convert_only);
        }
      }
    }
  }
}

void MHACoreLayer::softmax_triangle(nntrainer::Tensor &qk_out, size_t row,
                                    size_t num_head, unsigned int from,
                                    BS::thread_pool<> &pool) {

  float *qk_out_ = qk_out.getData<float>();

  if (from) {
    size_t start_row = 0;
    size_t end_row = from + 1;
    nntrainer::softmax_row_inplace(qk_out_, start_row, end_row, num_head);
  } else {
    std::vector<std::future<void>> futures;
    for (size_t i = 0; i < row; ++i) {
      size_t start_row = (i * (i + 1)) / 2;
      size_t end_row = ((i + 1) * (i + 2)) / 2;
      futures.push_back(pool.submit_task([=]() {
        nntrainer::softmax_row(qk_out_, start_row, end_row, num_head);
      }));
    }
    for (auto &fut : futures) {
      fut.get();
    }
  }
}

void MHACoreLayer::compute_fp16vcache_fp32_transposed(
  const float *in, const uint16_t *vcache, float *output, int seq,
  int num_cache_head, int gqa_size, int head_dim, bool process_all,
  BS::thread_pool<> &pool) {

  if (process_all) {
    std::vector<std::future<void>> futures;
    futures.reserve(seq);

    for (int i = 0; i < seq; ++i) {
      futures.push_back(pool.submit_task([=]() {
        const float *input =
          in + ((i * (i + 1)) / 2) * num_cache_head * gqa_size;
        float *out = output + i * (num_cache_head * gqa_size * head_dim);
        nntrainer::compute_fp16vcache_fp32_transposed(
          i, input, vcache, out, num_cache_head, gqa_size, head_dim);
      }));
    }
    for (auto &fut : futures)
      fut.get();
  } else {
    nntrainer::compute_fp16vcache_fp32_transposed(
      seq - 1, in, vcache, output, num_cache_head, gqa_size, head_dim);
  }
}

/**
 * @brief rotary embedding-related member function
 */
void MHACoreLayer::apply_rotary_emb_tensor(nntrainer::Tensor &in,
                                           unsigned int dim,
                                           unsigned int from) {
  nntrainer::Tensor out(in.getDim());
  float value = 0;
  float transformed_value = 0.0;
  unsigned int half_ = dim / 2;
  unsigned int max_timestep =
    std::get<nntrainer::props::MaxTimestep>(mha_core_props).get();

  std::vector<float> *cos_ = nullptr;
  std::vector<float> *sin_ = nullptr;

  if (from >= max_timestep) {
    cos_ = new std::vector<float>(dim);
    sin_ = new std::vector<float>(dim);
#ifdef USE_NEON
    nntrainer::calc_trigonometric_vals_dup(half_, thetas.data(), cos_->data(),
                                           sin_->data(), from);
#else
    for (unsigned int i = 0; i < half_; ++i) {
      float angle = from * thetas[i];
      (*cos_)[i] = std::cos(angle);
      (*cos_)[i + half_] = std::cos(angle); // repeated 2 times

      (*sin_)[i] = std::sin(angle);
      (*sin_)[i + half_] = std::sin(angle); // repeated 2 times
    }
#endif
  }

  if (in.getDataType() == ml::train::TensorDim::DataType::FP32) {
    for (unsigned int b = 0; b < in.batch(); b++) {
      for (unsigned int c = 0; c < in.channel(); c++) {
        for (unsigned int h = 0; h < in.height(); h++) {
          if (from < max_timestep) {
            cos_ = &(*freqs_cos)[from + h];
            sin_ = &(*freqs_sin)[from + h];
          }

          for (unsigned int w = 0; w < in.width(); w = w + dim) {
            for (unsigned int k = 0; k < dim; k++) {
              unsigned int span = w + k;
              value = in.getValue<float>(b, c, h, span);

              if (k < half_) {
                transformed_value =
                  -1.0 * in.getValue<float>(b, c, h, span + half_);
              } else {
                transformed_value = in.getValue<float>(b, c, h, span - half_);
              }
              value = value * (*cos_)[k] + transformed_value * (*sin_)[k];
              out.setValue(b, c, h, span, value);
            }
          }
        }
      }
    }
  } else if (in.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    for (unsigned int b = 0; b < in.batch(); b++) {
      for (unsigned int c = 0; c < in.channel(); c++) {
        for (unsigned int h = 0; h < in.height(); h++) {
          if (from < max_timestep) {
            cos_ = &(*freqs_cos)[from + h];
            sin_ = &(*freqs_sin)[from + h];
          }
          for (unsigned int w = 0; w < in.width(); w = w + dim) {
#ifdef USE_NEON
            nntrainer::compute_rotary_embedding_value(
              dim, half_, w, in.getData<_FP16>() + in.getIndex(b, c, h, 0),
              out.getData<_FP16>() + out.getIndex(b, c, h, 0), cos_->data(),
              sin_->data());
#else
            for (unsigned int k = 0; k < dim; k++) {
              unsigned int span = w + k;
              value = static_cast<float>(in.getValue<_FP16>(b, c, h, span));

              if (k < half_) {
                transformed_value =
                  -1.0 *
                  static_cast<float>(in.getValue<_FP16>(b, c, h, half_ + span));
              } else {
                transformed_value =
                  static_cast<float>(in.getValue<_FP16>(b, c, h, span - half_));
              }
              out.setValue(b, c, h, span,
                           static_cast<_FP16>(value * (*cos_)[k] +
                                              transformed_value * (*sin_)[k]));
            }
#endif
          }
        }
      }
    }
#else
    throw std::invalid_argument("Error: enable-fp16 is not enabled");
#endif
  }

  if (from >= max_timestep) {
    delete cos_;
    delete sin_;
  }
  in.copy(out);
}

void MHACoreLayer::setBatch(nntrainer::RunLayerContext &context,
                            unsigned int batch) {

  const float dropout_rate =
    std::get<nntrainer::props::DropOutRate>(mha_core_props).get();
  context.updateTensor(weight_idx[AttentionParams::cache_key], batch);
  context.updateTensor(weight_idx[AttentionParams::cache_value], batch);
  // context.updateTensor(weight_idx[AttentionParams::attention_weight], batch);
  if (dropout_rate > epsilon) {
    context.updateTensor(weight_idx[AttentionParams::dropout_mask], batch);
  }
}

void MHACoreLayer::updateTensorsByInputDimensions(
  nntrainer::RunLayerContext &context,
  std::vector<nntrainer::TensorDim> input_dimensions) {
  unsigned int height = input_dimensions[0].height();
  unsigned int &max_timestep =
    std::get<nntrainer::props::MaxTimestep>(mha_core_props).get();
  unsigned int &max_new_tokens =
    std::get<props::MaxNewTokens>(mha_core_props).get();
  unsigned int &max_position_embeddings =
    std::get<props::MaxPositionEmbeddings>(mha_core_props).get();
  max_timestep = height + max_new_tokens;

  ml::train::TensorDim kv_dim = input_dimensions[0];
  kv_dim.width(kv_dim.width() / (num_heads_Q / num_heads_KV));

  ml::train::TensorDim kv_cache_dim = kv_dim;
  kv_cache_dim.setDataType(ml::train::TensorDim::DataType::UINT16);
  kv_cache_dim.height(max_timestep);

  precompute_freqs(head_dim, max_position_embeddings, theta);

  context.updateInput(INOUT_INDEX::QUERY, input_dimensions[0]);
  context.updateInput(INOUT_INDEX::KEY, kv_dim);
  context.updateInput(INOUT_INDEX::VALUE, kv_dim);
  context.updateOutput(0, input_dimensions[0]);

  context.updateTensor(weight_idx[AttentionParams::cache_key], kv_cache_dim);
  context.updateTensor(weight_idx[AttentionParams::cache_value], kv_cache_dim);
}

void MHACoreLayer::calcDerivative(nntrainer::RunLayerContext &context) {}

void MHACoreLayer::calcGradient(nntrainer::RunLayerContext &context) {}

void MHACoreLayer::exportTo(nntrainer::Exporter &exporter,
                            const ml::train::ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
  exporter.saveResult(mha_core_props, method, this);
}

void MHACoreLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, mha_core_props);
  LayerImpl::setProperty(remain_props);
}

#ifdef PLUGGABLE

nntrainer::Layer *create_mha_core_layer() {
  auto layer = new MHACoreLayer();
  return layer;
}

void destroy_mha_core_layer(nntrainer::Layer *layer) { delete layer; }

extern "C" {
nntrainer::LayerPluggable ml_train_layer_pluggable{create_mha_core_layer,
                                                   destroy_mha_core_layer};
}

#endif

} // namespace causallm
