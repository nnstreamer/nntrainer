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
#include <immintrin.h>
#include <type_traits>

#ifdef _WIN32
#define COMPUTE_FP16_TO_FP32(x)                                                \
  _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(x)))
#define COMPUTE_FP32_TO_FP16(x)                                                \
  _mm_extract_epi16(_mm_cvtps_ph(_mm_set_ss(x), 0), 0)
#else
#define COMPUTE_FP16_TO_FP32(x) _cvtsh_ss(x)
#define COMPUTE_FP32_TO_FP16(x) _cvtss_sh(x, 0)
#endif

inline float convert_scalar(uint16_t h) {
  return nntrainer::compute_fp16_to_fp32(h);
}

inline __m256 load_fp16_8(const uint16_t *src) {
  float tmp[8];
  for (int i = 0; i < 8; ++i)
    tmp[i] = nntrainer::compute_fp16_to_fp32(src[i]);
  return _mm256_loadu_ps(tmp);
}

inline __m256 load_fp16_8_avx(const uint16_t *src) {
  __m128i in = _mm_loadu_si128(reinterpret_cast<const __m128i *>(src));
  return _mm256_cvtph_ps(in);
}

inline void load_fp16_8_to_chunk(const uint16_t *b_src, float *temp_row,
                                 int chunk_size) {
  int i = 0;
  for (; i + 8 <= chunk_size; i += 8) {
    __m256 f32 = load_fp16_8_avx((const uint16_t *)(b_src + i));
    _mm256_storeu_ps(temp_row + i, f32);
  }
  for (; i < chunk_size; ++i) {
    temp_row[i] = convert_scalar(b_src[i]);
  }
}

inline void convert_fp16_to_fp32_avx2(float *dst, const uint16_t *src,
                                      int len) {
  int i = 0;
  for (; i + 7 < len; i += 8) {
    __m128i half = _mm_loadu_si128(reinterpret_cast<const __m128i *>(src + i));
    __m256 full = _mm256_cvtph_ps(half);
    _mm256_storeu_ps(dst + i, full);
  }

  for (; i < len; ++i) {
    dst[i] = COMPUTE_FP16_TO_FP32(src[i]);
  }
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

void MHACoreLayer::compute_kcaches(const float *in, const uint16_t *kcache,
                                   float *output, int seq, int num_cache_head,
                                   int gqa_size, int head_dim, bool process_all,
                                   BS::thread_pool<> &pool) {

  size_t local_window_size =
    std::get<props::SlidingWindow>(mha_core_props).get();

  int seq_start = 0;
  int seq_end = process_all ? seq : 1;

  unsigned n_threads = 4;
#pragma omp parallel for num_threads(n_threads)
  for (int i = seq_end - 1; i >= seq_start; --i) {
    std::vector<float> tmp_fp32(head_dim);

    int num_rows = process_all ? i + 1 : seq;
    size_t start_row =
      local_window_size < num_rows ? num_rows - local_window_size : 0;
    int out_offset = calc_attn_index(i) * num_cache_head * gqa_size;

    for (int n = 0; n < num_cache_head; ++n) {

      for (int row = start_row; row < num_rows; ++row) {
        if (row + 1 < num_rows) {
          const uint16_t *next_kptr =
            kcache + ((row + 1) * num_cache_head + n) * head_dim;
          _mm_prefetch(reinterpret_cast<const char *>(next_kptr), _MM_HINT_T0);
        }

        const uint16_t *kptr = kcache + (row * num_cache_head + n) * head_dim;

        int d0 = 0;
        for (; d0 + 8 <= head_dim; d0 += 8) {
          __m128i half =
            _mm_loadu_si128(reinterpret_cast<const __m128i *>(kptr + d0));
          __m256 f32 = _mm256_cvtph_ps(half);
          _mm256_storeu_ps(&tmp_fp32[d0], f32);
        }
        for (; d0 < head_dim; ++d0) {
          tmp_fp32[d0] = convert_scalar(kptr[d0]);
        }

        for (int g = 0; g < gqa_size; ++g) {
          const float *a_ptr = in + num_cache_head * gqa_size * head_dim * i +
                               n * gqa_size * head_dim + g * head_dim;
          const float *b_row;
          if constexpr (std::is_same<uint16_t, float>::value) {
            b_row = reinterpret_cast<const float *>(kcache + n * head_dim);
          } else {
            b_row = tmp_fp32.data();
          }

          float sum = 0.0f;
          int i = 0;
          __m256 acc = _mm256_setzero_ps();
          for (; i + 8 <= head_dim; i += 8) {
            __m256 va = _mm256_loadu_ps(a_ptr + i);
            __m256 vb = _mm256_loadu_ps(b_row + i);
            acc = _mm256_fmadd_ps(va, vb, acc);
          }

          __m128 low = _mm256_castps256_ps128(acc);
          __m128 high = _mm256_extractf128_ps(acc, 1);
          __m128 sum128 = _mm_add_ps(low, high);
          sum128 = _mm_hadd_ps(sum128, sum128);
          sum128 = _mm_hadd_ps(sum128, sum128);
          sum += _mm_cvtss_f32(sum128);

          for (; i < head_dim; ++i)
            sum += a_ptr[i] * b_row[i];

          output[out_offset +
                 (local_window_size == UINT_MAX || num_rows < local_window_size
                    ? row
                    : row - (num_rows - local_window_size)) *
                   num_cache_head * gqa_size +
                 n * gqa_size + g] = sum / sqrt((float)head_dim);
        }
      }
    }
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
  auto &pool = nntrainer::ThreadPoolManager::getInstance();

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

  nntrainer::Tensor out_(1, 1, (from) ? to : calc_attn_index(to), num_heads_Q,
                         b_projected_query_step.getTensorType());

  unsigned int gqa_size = num_heads_Q / num_heads_KV;

  compute_kcaches(b_projected_query_step.getData<float>(),
                  b_cached_key.getData<uint16_t>(), out_.getData<float>(), to,
                  num_heads_KV, gqa_size, head_dim, (from) ? false : true,
                  pool);

  softmax_triangle_AVX2(out_, to - from, num_heads_Q, from, pool);

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

enum class OutputType { FLOAT32, FP16 };

inline void compute_rotary_embedding_value(unsigned int width, unsigned int dim,
                                           unsigned int half_, float *inout,
                                           void *output, const float *cos_,
                                           const float *sin_,
                                           bool only_convert_to_fp16) {

  using OutputType = ml::train::TensorDim::DataType;
  OutputType out_type = OutputType::FP32;

  if (output != nullptr)
    out_type = OutputType::UINT16;

  for (unsigned int w = 0; w < width; w += dim) {
    unsigned int k = 0;

    for (; k + 7 < half_; k += 8) {
      unsigned int i0 = w + k;
      unsigned int i1 = w + k + half_;

      __m256 a = _mm256_loadu_ps(&inout[i0]);
      __m256 b = _mm256_loadu_ps(&inout[i1]);

      if (only_convert_to_fp16) {
        if (out_type == OutputType::UINT16) {
          __m128i a_fp16 = _mm256_cvtps_ph(a, 0);
          __m128i b_fp16 = _mm256_cvtps_ph(b, 0);

          _mm_storeu_si128(
            reinterpret_cast<__m128i *>(static_cast<uint16_t *>(output) + i0),
            a_fp16);
          _mm_storeu_si128(
            reinterpret_cast<__m128i *>(static_cast<uint16_t *>(output) + i1),
            b_fp16);
        }

      } else {
        __m256 cos_v = _mm256_loadu_ps(&cos_[k]);
        __m256 sin_v = _mm256_loadu_ps(&sin_[k]);

        __m256 out0 =
          _mm256_sub_ps(_mm256_mul_ps(a, cos_v), _mm256_mul_ps(b, sin_v));
        __m256 out1 =
          _mm256_add_ps(_mm256_mul_ps(a, sin_v), _mm256_mul_ps(b, cos_v));

        if (out_type == OutputType::UINT16) {
          __m128i out0_fp16 = _mm256_cvtps_ph(out0, 0);
          __m128i out1_fp16 = _mm256_cvtps_ph(out1, 0);

          _mm_storeu_si128(
            reinterpret_cast<__m128i *>(static_cast<uint16_t *>(output) + i0),
            out0_fp16);
          _mm_storeu_si128(
            reinterpret_cast<__m128i *>(static_cast<uint16_t *>(output) + i1),
            out1_fp16);

        } else if (out_type == OutputType::FP32) {
          _mm256_storeu_ps(&inout[i0], out0);
          _mm256_storeu_ps(&inout[i1], out1);
        }
      }
    }

    for (; k < half_; ++k) {
      unsigned int i0 = w + k;
      unsigned int i1 = w + k + half_;
      // assert(i1 < width && "Scalar i1 overflow!");
      float a = inout[i0];
      float b = inout[i1];

      if (only_convert_to_fp16) {
        static_cast<uint16_t *>(output)[i0] = COMPUTE_FP32_TO_FP16(a);
        static_cast<uint16_t *>(output)[i1] = COMPUTE_FP32_TO_FP16(b);

      } else {

        float c = cos_[k];
        float s = sin_[k];

        float out0 = a * c - b * s;
        float out1 = a * s + b * c;

        if (out_type == OutputType::UINT16) {
          static_cast<uint16_t *>(output)[i0] = COMPUTE_FP32_TO_FP16(out0);
          static_cast<uint16_t *>(output)[i1] = COMPUTE_FP32_TO_FP16(out1);
        } else if (out_type == OutputType::FP32) {
          inout[i0] = out0;
          inout[i1] = out1;
        }
      }
    }
  }
}

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

          compute_rotary_embedding_value(in.width(), dim, half_, in_ptr,
                                         nullptr, cos_->data(), sin_->data(),
                                         convert_only);
        } else if (out.getDataType() ==
                   ml::train::TensorDim::DataType::UINT16) {

          uint16_t *out_ptr = out.getData<uint16_t>() +
                              b * out.channel() * out.height() * out.width() +
                              c * out.height() * out.width() + h * out.width();

          compute_rotary_embedding_value(in.width(), dim, half_, in_ptr,
                                         out_ptr, cos_->data(), sin_->data(),
                                         convert_only);
        }
      }
    }
  }
}

inline __m256 exp256_ps(__m256 x) {
  /* Low-Precision Versino III */
  const __m256 LOG2EF = _mm256_set1_ps(1.44269504088896341f); // 1 / ln(2)
  const __m256 LN2 = _mm256_set1_ps(0.6931471805599453f);     // ln(2)

  // Clamp input to range to prevent overflow/underflow
  const __m256 max_x = _mm256_set1_ps(88.3762626647949f);  // log(FLT_MAX)
  const __m256 min_x = _mm256_set1_ps(-88.3762626647949f); // log(FLT_MIN)
  x = _mm256_max_ps(min_x, _mm256_min_ps(max_x, x));

  // Range reduction: x = n * ln2 + r
  __m256 fx = _mm256_mul_ps(x, LOG2EF); // x * (1/ln(2))
  fx = _mm256_floor_ps(_mm256_add_ps(fx, _mm256_set1_ps(0.5f)));

  __m256 tmp = _mm256_mul_ps(fx, LN2); // n * ln(2)
  __m256 r = _mm256_sub_ps(x, tmp);    // r = x - n * ln2

  // Compute exp(r) using 10th-order polynomial (Horner's method)
  const __m256 c0 = _mm256_set1_ps(1.0f);
  const __m256 c1 = _mm256_set1_ps(1.0f);
  const __m256 c2 = _mm256_set1_ps(0.5f);
  const __m256 c3 = _mm256_set1_ps(1.0f / 6.0f);
  const __m256 c4 = _mm256_set1_ps(1.0f / 24.0f);
  const __m256 c5 = _mm256_set1_ps(1.0f / 120.0f);
  const __m256 c6 = _mm256_set1_ps(1.0f / 720.0f);
  const __m256 c7 = _mm256_set1_ps(1.0f / 5040.0f);
  const __m256 c8 = _mm256_set1_ps(1.0f / 40320.0f);
  const __m256 c9 = _mm256_set1_ps(1.0f / 362880.0f);
  const __m256 c10 = _mm256_set1_ps(1.0f / 3628800.0f);

  __m256 y = c10;
  y = _mm256_fmadd_ps(y, r, c9);
  y = _mm256_fmadd_ps(y, r, c8);
  y = _mm256_fmadd_ps(y, r, c7);
  y = _mm256_fmadd_ps(y, r, c6);
  y = _mm256_fmadd_ps(y, r, c5);
  y = _mm256_fmadd_ps(y, r, c4);
  y = _mm256_fmadd_ps(y, r, c3);
  y = _mm256_fmadd_ps(y, r, c2);
  y = _mm256_fmadd_ps(y, r, c1);
  y = _mm256_fmadd_ps(y, r, c0); // final y = (((...r+...)*r+...)*r + 1)

  // Reconstruct exp(x) = 2^n * exp(r)
  __m256i emm0 = _mm256_cvtps_epi32(fx);
  emm0 = _mm256_add_epi32(emm0, _mm256_set1_epi32(127));
  emm0 = _mm256_slli_epi32(emm0, 23);
  __m256 pow2n = _mm256_castsi256_ps(emm0);

  return _mm256_mul_ps(y, pow2n);
}

void MHACoreLayer::softmax_row_AVX2_inplace(float *qk_out, size_t start_row,
                                            size_t end_row, size_t num_heads) {
  size_t row_range = end_row - start_row;
  const size_t full_blocks = (num_heads / 8) * 8;
  // const size_t remainder = num_heads % 8;

  float *max_vals = new float[num_heads];
  float *sum_vals = new float[num_heads];
  // 1. max
  for (size_t c = 0; c < num_heads; ++c) {
    float max_val = -INFINITY;
    for (size_t r = start_row; r < end_row; ++r)
      max_val = std::max(max_val, qk_out[r * num_heads + c]);
    max_vals[c] = max_val;
  }

  // 2. inplace exp + sum
  for (size_t c = 0; c < full_blocks; c += 8) {
    __m256 maxv = _mm256_loadu_ps(&max_vals[c]);
    __m256 sum = _mm256_setzero_ps();
    for (size_t r = 0; r < row_range; ++r) {
      float *ptr = &qk_out[(start_row + r) * num_heads + c];
      __m256 val = _mm256_loadu_ps(ptr);
      __m256 e = exp256_ps(_mm256_sub_ps(val, maxv));
      _mm256_storeu_ps(ptr, e); // overwrite qk_out
      sum = _mm256_add_ps(sum, e);
    }
    _mm256_storeu_ps(&sum_vals[c], sum);
  }

  for (size_t c = full_blocks; c < num_heads; ++c) {
    float sum = 0.0f;
    float maxv = max_vals[c];
    for (size_t r = 0; r < row_range; ++r) {
      float &a = qk_out[(start_row + r) * num_heads + c];
      a = std::exp(a - maxv); // overwrite qk_out
      sum += a;
    }
    sum_vals[c] = sum;
  }
  // 3. softmax = exp / sum (inplace)
  for (size_t r = 0; r < row_range; ++r) {
    for (size_t c = 0; c < full_blocks; c += 8) {
      float *ptr = &qk_out[(start_row + r) * num_heads + c];
      __m256 val = _mm256_loadu_ps(ptr); // already exp(x - max)
      __m256 sumv = _mm256_loadu_ps(&sum_vals[c]);
      __m256 soft = _mm256_div_ps(val, sumv);
      _mm256_storeu_ps(ptr, soft);
    }
    for (size_t c = full_blocks; c < num_heads; ++c) {
      qk_out[(start_row + r) * num_heads + c] /= sum_vals[c];
    }
  }

  delete[] max_vals;
  delete[] sum_vals;
}

void MHACoreLayer::softmax_row_AVX2(float *qk_out, size_t start_row,
                                    size_t end_row, size_t num_heads) {

  const size_t full_block = (num_heads / 8) * 8;

  float *max_vals = new float[num_heads];
  float *sum_vals = new float[num_heads];

  // 1. Find Max along with col
  for (size_t c = 0; c < num_heads; ++c) {
    float max_val = -INFINITY;
    for (size_t r = start_row; r < end_row; ++r) {
      max_val = std::max(max_val, qk_out[r * num_heads + c]);
    }
    max_vals[c] = max_val;
  }

  // 2. Compute sum along with col (exp vectorized)
  for (size_t c = 0; c < full_block; c += 8) {
    __m256 sum = _mm256_setzero_ps();
    for (size_t r = start_row; r < end_row; ++r) {
      __m256 val = _mm256_loadu_ps(&qk_out[r * num_heads + c]);
      __m256 maxv = _mm256_loadu_ps(&max_vals[c]);
      __m256 sub = _mm256_sub_ps(val, maxv);
      __m256 e = exp256_ps(sub);
      sum = _mm256_add_ps(sum, e);
    }
    _mm256_storeu_ps(&sum_vals[c], sum);
  }

  for (size_t c = full_block; c < num_heads; ++c) {
    float sum = 0.0f;
    for (size_t r = start_row; r < end_row; ++r) {
      sum += std::exp(qk_out[r * num_heads + c] - max_vals[c]);
    }
    sum_vals[c] = sum;
  }

  // 3. apply softmax
  for (size_t r = start_row; r < end_row; ++r) {
    for (size_t c = 0; c < full_block; c += 8) {
      __m256 val = _mm256_loadu_ps(&qk_out[r * num_heads + c]);
      __m256 maxv = _mm256_loadu_ps(&max_vals[c]);
      __m256 sub = _mm256_sub_ps(val, maxv);
      __m256 e = exp256_ps(sub);
      __m256 sumv = _mm256_loadu_ps(&sum_vals[c]);
      __m256 softmax = _mm256_div_ps(e, sumv);
      _mm256_storeu_ps(&qk_out[r * num_heads + c], softmax);
    }
    for (size_t c = full_block; c < num_heads; ++c) {
      qk_out[r * num_heads + c] =
        std::exp(qk_out[r * num_heads + c] - max_vals[c]) / sum_vals[c];
    }
  }

  delete[] max_vals;
  delete[] sum_vals;
}

void MHACoreLayer::softmax_triangle_AVX2(nntrainer::Tensor &qk_out, size_t row,
                                         size_t num_head, unsigned int from,
                                         BS::thread_pool<> &pool) {

  float *qk_out_ = qk_out.getData<float>();

  size_t local_window_size =
    std::get<props::SlidingWindow>(mha_core_props).get();

  if (from) {
    size_t start_row = 0;
    size_t end_row =
      local_window_size == UINT_MAX ? from + 1 : local_window_size;
    softmax_row_AVX2_inplace(qk_out_, start_row, end_row, num_head);
  } else {
    std::vector<std::future<void>> futures;
    unsigned n_threads = 4;
#pragma omp parallel for num_threads(n_threads)
    for (int i = row - 1; i >= 0; --i) {
      size_t start_row = calc_attn_index(i);
      size_t end_row = calc_attn_index(i + 1);

      const size_t full_block = (num_head / 8) * 8;

      float *max_vals = new float[num_head];
      float *sum_vals = new float[num_head];

      // 1. Find Max along with col
      for (size_t c = 0; c < num_head; ++c) {
        float max_val = -INFINITY;
        for (size_t r = start_row; r < end_row; ++r) {
          max_val = std::max(max_val, qk_out_[r * num_head + c]);
        }
        max_vals[c] = max_val;
      }

      // 2. Compute sum along with col (exp vectorized)
      for (size_t c = 0; c < full_block; c += 8) {
        __m256 sum = _mm256_setzero_ps();
        for (size_t r = start_row; r < end_row; ++r) {
          __m256 val = _mm256_loadu_ps(&qk_out_[r * num_head + c]);
          __m256 maxv = _mm256_loadu_ps(&max_vals[c]);
          __m256 sub = _mm256_sub_ps(val, maxv);
          __m256 e = exp256_ps(sub);
          sum = _mm256_add_ps(sum, e);
        }
        _mm256_storeu_ps(&sum_vals[c], sum);
      }

      for (size_t c = full_block; c < num_head; ++c) {
        float sum = 0.0f;
        for (size_t r = start_row; r < end_row; ++r) {
          sum += std::exp(qk_out_[r * num_head + c] - max_vals[c]);
        }
        sum_vals[c] = sum;
      }

      // 3. apply softmax
      for (size_t r = start_row; r < end_row; ++r) {
        for (size_t c = 0; c < full_block; c += 8) {
          __m256 val = _mm256_loadu_ps(&qk_out_[r * num_head + c]);
          __m256 maxv = _mm256_loadu_ps(&max_vals[c]);
          __m256 sub = _mm256_sub_ps(val, maxv);
          __m256 e = exp256_ps(sub);
          __m256 sumv = _mm256_loadu_ps(&sum_vals[c]);
          __m256 softmax = _mm256_div_ps(e, sumv);
          _mm256_storeu_ps(&qk_out_[r * num_head + c], softmax);
        }
        for (size_t c = full_block; c < num_head; ++c) {
          qk_out_[r * num_head + c] =
            std::exp(qk_out_[r * num_head + c] - max_vals[c]) / sum_vals[c];
        }
      }

      delete[] max_vals;
      delete[] sum_vals;
    }
  }
}

// FP16 #8 → FP32 #8
inline __m256 load_fp16_8_to_fp32(const uint16_t *src) {
  __m128i half = _mm_loadu_si128(reinterpret_cast<const __m128i *>(src));
  return _mm256_cvtph_ps(half);
}

inline __m256 load_fp16_block(const uint16_t *src) {
  __m128i half = _mm_loadu_si128(reinterpret_cast<const __m128i *>(src));
  return _mm256_cvtph_ps(half);
}

void MHACoreLayer::compute_fp16vcache_fp32_transposed(
  const float *in, const uint16_t *vcache, float *output, int seq,
  int num_cache_head, int gqa_size, int head_dim, bool process_all,
  BS::thread_pool<> &pool) {

  int seq_start = process_all ? 0 : (seq - 1);
  int seq_end = seq;

  size_t local_window_size =
    std::get<props::SlidingWindow>(mha_core_props).get();

  unsigned n_threads = 4;
#pragma omp parallel for num_threads(n_threads)
  for (int i = seq_end - 1; i >= seq_start; --i) {

    std::vector<float> tmp_fp32(head_dim);
    int a_row_start =
      process_all ? calc_attn_index(i) * num_cache_head * gqa_size : 0;
    int out_offset = process_all ? i : 0;

    for (int n = 0; n < num_cache_head; ++n) {
      int num_blocks = head_dim / 8;
      int rem = head_dim % 8;

      std::vector<__m256> sumVec(num_blocks * gqa_size, _mm256_setzero_ps());
      std::vector<float> sumRem(gqa_size * rem, 0.0f);

      for (int j = i < local_window_size ? 0 : i + 1 - local_window_size;
           j <= i; ++j) {
        if (j + 1 < seq) {
          const uint16_t *next_vptr =
            vcache + ((j + 1) * num_cache_head + n) * head_dim;
          _mm_prefetch(reinterpret_cast<const char *>(next_vptr), _MM_HINT_T0);
        }

        const uint16_t *vptr = vcache + (j * num_cache_head + n) * head_dim;

        int d0 = 0;
        for (; d0 + 8 <= head_dim; d0 += 8) {
          __m128i half =
            _mm_loadu_si128(reinterpret_cast<const __m128i *>(vptr + d0));
          __m256 f32 = _mm256_cvtph_ps(half);
          _mm256_storeu_ps(&tmp_fp32[d0], f32);
        }
        for (; d0 < head_dim; ++d0) {
          tmp_fp32[d0] = convert_scalar(vptr[d0]);
        }

        for (int h = 0; h < gqa_size; ++h) {
          /*             float a_val =
                        in[a_row_start + (j * gqa_size + h) * num_cache_head +
              n];
            */
          float a_val =
            in[a_row_start +
               ((local_window_size == UINT_MAX || i < local_window_size
                   ? j
                   : j - (i + 1 - local_window_size)) *
                  gqa_size * num_cache_head +
                n * gqa_size + h)];
          __m256 inVec = _mm256_set1_ps(a_val);

          for (int b = 0; b < num_blocks; ++b) {
            __m256 bVec = _mm256_loadu_ps(&tmp_fp32[b * 8]);
            sumVec[h * num_blocks + b] =
              _mm256_fmadd_ps(inVec, bVec, sumVec[h * num_blocks + b]);
          }

          float *remPtr = &sumRem.data()[h * rem];
          int base = num_blocks * 8;
          for (int r = 0; r < rem; ++r) {
            remPtr[r] += a_val * tmp_fp32[base + r];
          }
        }
      }

      for (int h = 0; h < gqa_size; ++h) {
        for (int b = 0; b < num_blocks; ++b) {
          int out_base =
            ((out_offset * num_cache_head + n) * gqa_size + h) * head_dim +
            b * 8;
          _mm256_storeu_ps(&output[out_base], sumVec[h * num_blocks + b]);
        }

        float *remPtr = &sumRem.data()[h * rem];
        // float *remPtr = &sumRem[h * rem];
        int base = num_blocks * 8;
        for (int r = 0; r < rem; ++r) {
          int out_idx =
            ((out_offset * num_cache_head + n) * gqa_size + h) * head_dim +
            base + r;
          output[out_idx] = remPtr[r];
        }
      }
    }
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

size_t MHACoreLayer::calc_attn_index(size_t i) {
  size_t local_window_size =
    std::get<props::SlidingWindow>(mha_core_props).get();

  if (local_window_size == UINT_MAX || i < local_window_size) {
    return (i * (i + 1)) / 2;
  } else {
    return (local_window_size * (local_window_size + 1)) / 2 +
           (i - local_window_size) * local_window_size;
  }
};

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
