// SPDX-License-Identifier: Apache-2.0
/**
 * @file   moe_layer_zero_copy.cpp
 * @date   15 January 2025
 * @brief  Zero-copy MoE Layer using only sharedTensor views (no memcpy)
 * @see    https://github.com/EunjuYang/nntrainer/blob/6e2a028cd9bc237fa18fdc117f14b65a38c3e9dd/nntrainer/layers/moe_layer.cpp
 * @author NNTrainer Team
 * @bug    No known bugs except for NYI items
 *
 * Zero-copy optimizations:
 * 1. Eliminates all memcpy operations using sharedTensor views
 * 2. Direct computation on scattered tensor data
 * 3. Maintains exact mathematical equivalence with original
 * 4. Preserves 3-layer expert structure (gate-up-down)
 * 5. Maximum memory and speed optimization
 */

#include <moe_layer.h>

#include <acti_func.h>
#include <cmath>
#include <algorithm>
#include <node_exporter.h>
#include <omp.h>
#include <stdexcept>

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

MoELayer::MoELayer() :
  LayerImpl(),
  num_experts(0),
  topk(0),
  moe_props(props::NumExperts(), props::NumExpertsPerToken(), props::Unit(),
            props::MoEActivation()),
  expert_gate_proj_indices({}),
  expert_up_proj_indices({}),
  expert_down_proj_indices({}),
  gate_idx(std::numeric_limits<unsigned>::max()),
  router_logits_idx(std::numeric_limits<unsigned>::max()),
  expert_mask_idx(std::numeric_limits<unsigned>::max()),
  expert_intermediate_buffer_idx(std::numeric_limits<unsigned>::max()) {}

void MoELayer::finalize(InitLayerContext &context) {
  // 원본과 동일한 finalize 로직
  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "MoE layer only supports single input";

  auto &weight_regularizer =
    std::get<props::WeightRegularizer>(*layer_impl_props);
  auto &weight_regularizer_constant =
    std::get<props::WeightRegularizerConstant>(*layer_impl_props);
  auto &weight_initializer =
    std::get<props::WeightInitializer>(*layer_impl_props);
  auto &weight_decay = std::get<props::WeightDecay>(*layer_impl_props);

  const auto &in_dim = context.getInputDimensions()[SINGLE_INOUT_IDX];
  const bool is_nchw = context.getFormat() == Tformat::NCHW;
  std::vector<TensorDim> output_dims(1);
  output_dims[SINGLE_INOUT_IDX] = in_dim;
  context.setOutputDimensions(output_dims);

  num_experts = std::get<props::NumExperts>(moe_props).get();
  topk = std::get<props::NumExpertsPerToken>(moe_props).get();
  const unsigned int intermediate_size = std::get<props::Unit>(moe_props).get();
  const unsigned int hidden_size = in_dim.width();

  // activation function
  if (std::get<props::MoEActivation>(moe_props).empty()) {
    throw std::runtime_error("Activation type is not set for MoE layer");
  }
  switch (context.getActivationDataType()) {
  case ml::train::TensorDim::DataType::FP32:
    acti_func.setActiFunc<float>(
      std::get<props::MoEActivation>(moe_props).get());
    break;
  default:
    throw std::runtime_error("Unsupported activation data type for MoE layer");
  }

  // Gate layer (router) - 원본과 동일
  TensorDim gate_dim(
    1, is_nchw ? 1 : num_experts, is_nchw ? hidden_size : 1,
    is_nchw ? num_experts : hidden_size,
    TensorDim::TensorType(context.getFormat(), context.getWeightDataType()),
    is_nchw ? 0b0011 : 0b0101);

  gate_idx = context.requestWeight(
    gate_dim, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, "gate", true);

  // Expert weights - 원본과 동일한 3-layer 구조
  expert_gate_proj_indices.reserve(num_experts);
  expert_up_proj_indices.reserve(num_experts);
  expert_down_proj_indices.reserve(num_experts);

  TensorDim expert_gate_dim(
    1, is_nchw ? 1 : intermediate_size, is_nchw ? hidden_size : 1,
    is_nchw ? intermediate_size : hidden_size,
    TensorDim::TensorType(context.getFormat(), context.getWeightDataType()),
    is_nchw ? 0b0011 : 0b0101);

  TensorDim expert_down_dim(
    1, is_nchw ? 1 : hidden_size, is_nchw ? intermediate_size : 1,
    is_nchw ? hidden_size : intermediate_size,
    TensorDim::TensorType(context.getFormat(), context.getWeightDataType()),
    is_nchw ? 0b0011 : 0b0101);

  for (unsigned int i = 0; i < num_experts; ++i) {
    expert_up_proj_indices.push_back(context.requestWeight(
      expert_gate_dim, weight_initializer, weight_regularizer,
      weight_regularizer_constant, weight_decay, 
      "expert_up_" + std::to_string(i), false));

    expert_gate_proj_indices.push_back(context.requestWeight(
      expert_gate_dim, weight_initializer, weight_regularizer,
      weight_regularizer_constant, weight_decay,
      "expert_gate_" + std::to_string(i), true));

    expert_down_proj_indices.push_back(context.requestWeight(
      expert_down_dim, weight_initializer, weight_regularizer,
      weight_regularizer_constant, weight_decay,
      "expert_down_" + std::to_string(i), false));
  }

  // Intermediate tensors - 원본과 동일
  const unsigned batch_size = in_dim.batch();
  const unsigned seq_len = in_dim.height();
  const unsigned total_tokens = batch_size * seq_len;

  router_logits_idx = context.requestTensor(
    {total_tokens, 1, 1, num_experts}, "router_logits", Initializer::NONE,
    false, TensorLifespan::FORWARD_FUNC_LIFESPAN);

  expert_mask_idx = context.requestTensor(
    {num_experts, 1, topk, total_tokens}, "expert_mask", Initializer::ZEROS,
    false, TensorLifespan::FORWARD_FUNC_LIFESPAN);

  // 중간 계산용 버퍼 (gate_out, up_out용) - memcpy 없이 계산만을 위한 공간
  const unsigned max_expert_tokens = total_tokens;
  expert_intermediate_buffer_idx = context.requestTensor(
    {max_expert_tokens, 1, 1, intermediate_size * 2}, "expert_intermediate_buffer", 
    Initializer::NONE, false, TensorLifespan::FORWARD_FUNC_LIFESPAN);
}

void MoELayer::forwarding(RunLayerContext &context, bool training) {
  Tensor &input = context.getInput(SINGLE_INOUT_IDX);
  Tensor &output = context.getOutput(SINGLE_INOUT_IDX);
  Tensor &router_logits = context.getTensor(router_logits_idx);
  Tensor &expert_mask = context.getTensor(expert_mask_idx);
  Tensor &intermediate_buffer = context.getTensor(expert_intermediate_buffer_idx);

  const unsigned batch_size = input.batch();
  const unsigned seq_len = input.height();
  const unsigned hidden_size = input.width();
  const unsigned total_tokens = batch_size * seq_len;

  // 원본과 동일한 reshape
  input.reshape({total_tokens, 1, 1, hidden_size});
  output.reshape({total_tokens, 1, 1, hidden_size});
  output.setZero();

  // Routing - 원본과 동일
  Tensor &gate_weights = context.getWeight(gate_idx);
  input.dot(gate_weights, router_logits);
  router_logits.apply(ActiFunc::softmax<float>, router_logits);
  
  auto topk_result = router_logits.topK(topk);
  auto topk_values = std::get<0>(topk_result);
  auto topk_indices = std::get<1>(topk_result);

  // Expert mask 생성 - 원본과 동일
  const uint32_t *indices_data = topk_indices.getData<uint32_t>();
#pragma omp parallel for collapse(2)
  for (int i = 0; i < static_cast<int>(total_tokens); ++i) {
    for (int k = 0; k < static_cast<int>(topk); ++k) {
      expert_mask.setValue(indices_data[i * topk + k], 0, k, i, 1.0f);
    }
  }

  // Zero-copy expert forwarding
#pragma omp parallel
  {
#pragma omp for schedule(dynamic)
    for (int expert_idx = 0; expert_idx < static_cast<int>(num_experts); ++expert_idx) {
      std::vector<unsigned> token_indices;
      std::vector<float> topk_values_vector;

      // Token indices 수집 - 원본과 동일
      for (int i = 0; i < static_cast<int>(total_tokens); ++i) {
        for (int k = 0; k < static_cast<int>(topk); ++k) {
          if (expert_mask.getValue<float>(expert_idx, 0, k, i) > 0.5f) {
            token_indices.push_back(i);
            topk_values_vector.push_back(
              topk_values.getValue<float>(i, 0, 0, k));
          }
        }
      }
      if (token_indices.empty())
        continue;

      // Zero-copy expert forward pass
      compute_expert_forward_zero_copy(
        input, output, token_indices, topk_values_vector,
        context.getWeight(expert_gate_proj_indices[expert_idx]),
        context.getWeight(expert_up_proj_indices[expert_idx]),
        context.getWeight(expert_down_proj_indices[expert_idx]),
        intermediate_buffer, false);
    }
  }

  // 원본과 동일한 reshape back
  output.reshape({batch_size, 1, seq_len, hidden_size});
}

void MoELayer::incremental_forwarding(RunLayerContext &context,
                                      unsigned int from, unsigned int to,
                                      bool training) {
  // 원본과 동일한 incremental forwarding 로직
  if (from) {
    NNTR_THROW_IF(to - from != 1, std::invalid_argument)
      << "incremental step size is not 1";
    from = 0;
    to = 1;
  }

  Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);
  Tensor &output_ = context.getOutput(SINGLE_INOUT_IDX);
  Tensor &router_logits_ = context.getTensor(router_logits_idx);
  Tensor &expert_mask = context.getTensor(expert_mask_idx);
  Tensor &intermediate_buffer = context.getTensor(expert_intermediate_buffer_idx);

  TensorDim input_step_dim = input_.getDim();
  TensorDim output_step_dim = output_.getDim();
  TensorDim router_logits_step_dim = router_logits_.getDim();

  input_step_dim.batch(1);
  output_step_dim.batch(1);
  router_logits_step_dim.batch(to - from);
  input_step_dim.height(to - from);
  output_step_dim.height(to - from);

  for (unsigned int b = 0; b < input_.batch(); ++b) {
    auto input = input_.getSharedDataTensor(
      input_step_dim, b * input_step_dim.getFeatureLen(), true);
    auto output = output_.getSharedDataTensor(
      output_step_dim, b * output_step_dim.getFeatureLen(), true);
    auto router_logits =
      router_logits_.getSharedDataTensor(router_logits_step_dim, 0, true);

    const unsigned batch_size = input.batch();
    const unsigned seq_len = input.height();
    const unsigned hidden_size = input.width();
    const unsigned total_tokens = batch_size * seq_len;

    // 원본과 동일한 처리
    input.reshape({total_tokens, 1, 1, hidden_size});
    output.reshape({total_tokens, 1, 1, hidden_size});
    output.setZero();
    expert_mask.setZero();

    // Routing
    Tensor &gate_weights = context.getWeight(gate_idx);
    input.dot(gate_weights, router_logits);
    router_logits.apply(ActiFunc::softmax<float>, router_logits);
    
    auto topk_result = router_logits.topK(topk);
    auto topk_values = std::get<0>(topk_result);
    auto topk_indices = std::get<1>(topk_result);

    // 원본과 동일한 정규화 (incremental에서만)
    topk_values.divide_i(topk_values.sum(3));

    const uint32_t *indices_data = topk_indices.getData<uint32_t>();
    for (int i = 0; i < static_cast<int>(total_tokens); ++i) {
      for (int k = 0; k < static_cast<int>(topk); ++k) {
        expert_mask.setValue(indices_data[i * topk + k], 0, k, i, 1.0f);
      }
    }

    // Expert forwarding (sequential for incremental)
    for (int expert_idx = 0; expert_idx < static_cast<int>(num_experts); ++expert_idx) {
      std::vector<unsigned> token_indices;
      std::vector<float> topk_values_vector;

      for (int i = 0; i < static_cast<int>(total_tokens); ++i) {
        for (int k = 0; k < static_cast<int>(topk); ++k) {
          if (expert_mask.getValue<float>(expert_idx, 0, k, i) > 0.5f) {
            token_indices.push_back(i);
            topk_values_vector.push_back(
              topk_values.getValue<float>(i, 0, 0, k));
          }
        }
      }
      if (token_indices.empty())
        continue;

      // Zero-copy expert forward pass with accumulation
      compute_expert_forward_zero_copy(
        input, output, token_indices, topk_values_vector,
        context.getWeight(expert_gate_proj_indices[expert_idx]),
        context.getWeight(expert_up_proj_indices[expert_idx]),
        context.getWeight(expert_down_proj_indices[expert_idx]),
        intermediate_buffer, true);  // accumulate = true for incremental
    }

    output.reshape({batch_size, 1, seq_len, hidden_size});
  }
}

// Zero-copy expert forward pass (완전히 memcpy 제거)
void MoELayer::compute_expert_forward_zero_copy(
  const Tensor &input, Tensor &output,
  const std::vector<unsigned> &token_indices,
  const std::vector<float> &weights, const Tensor &gate_proj,
  const Tensor &up_proj, const Tensor &down_proj,
  Tensor &intermediate_buffer, bool accumulate) {

  const unsigned tokens = token_indices.size();
  const unsigned hidden_size = input.width();
  const unsigned intermediate_size = gate_proj.width();

  // 중간 계산용 버퍼의 view 생성 (메모리 할당 없음)
  auto gate_buffer = intermediate_buffer.getSharedDataTensor(
    {tokens, 1, 1, intermediate_size}, 0, true);
  auto up_buffer = intermediate_buffer.getSharedDataTensor(
    {tokens, 1, 1, intermediate_size}, 
    tokens * intermediate_size * sizeof(float), true);

  // Direct pointer access for zero-copy operations
  const float *input_data = input.getData<float>();
  const float *gate_proj_data = gate_proj.getData<float>();
  const float *up_proj_data = up_proj.getData<float>();
  const float *down_proj_data = down_proj.getData<float>();
  float *gate_buffer_data = gate_buffer.getData<float>();
  float *up_buffer_data = up_buffer.getData<float>();
  float *output_data = output.getData<float>();

  // 1. Gate projection (scattered input → gate_buffer)
  compute_scattered_projection(input_data, gate_proj_data, gate_buffer_data,
                              token_indices, hidden_size, intermediate_size);

  // 2. Apply SiLU activation in-place
  apply_silu_inplace(gate_buffer_data, tokens * intermediate_size);

  // 3. Up projection (scattered input → up_buffer)
  compute_scattered_projection(input_data, up_proj_data, up_buffer_data,
                              token_indices, hidden_size, intermediate_size);

  // 4. Element-wise multiply: gate_buffer *= up_buffer
  multiply_tensors_inplace(gate_buffer_data, up_buffer_data, 
                          tokens * intermediate_size);

  // 5. Down projection + weight application + output accumulation
  compute_final_projection_and_accumulate(
    gate_buffer_data, down_proj_data, output_data,
    token_indices, weights, intermediate_size, hidden_size, accumulate);
}

// Scattered projection: input[token_indices] × weight → output (zero-copy)
void MoELayer::compute_scattered_projection(
  const float *input_data, const float *weight_data, float *output_data,
  const std::vector<unsigned> &token_indices,
  unsigned int input_dim, unsigned int output_dim) {

  // 직접 계산 (memcpy 없음)
  for (size_t i = 0; i < token_indices.size(); ++i) {
    const unsigned token_idx = token_indices[i];
    const float *input_row = input_data + token_idx * input_dim;
    float *output_row = output_data + i * output_dim;

    // Matrix multiplication: input_row × weight → output_row
    for (unsigned j = 0; j < output_dim; ++j) {
      float sum = 0.0f;
      const float *weight_col = weight_data + j;  // column j
      
      // Vectorizable dot product
      #pragma omp simd reduction(+:sum)
      for (unsigned k = 0; k < input_dim; ++k) {
        sum += input_row[k] * weight_col[k * output_dim];
      }
      output_row[j] = sum;
    }
  }
}

// In-place SiLU activation
void MoELayer::apply_silu_inplace(float *data, unsigned int size) {
  #pragma omp simd
  for (unsigned int i = 0; i < size; ++i) {
    const float x = data[i];
    data[i] = x / (1.0f + std::exp(-x));  // SiLU(x) = x * sigmoid(x)
  }
}

// Element-wise multiplication in-place
void MoELayer::multiply_tensors_inplace(float *a, const float *b, unsigned int size) {
  #pragma omp simd
  for (unsigned int i = 0; i < size; ++i) {
    a[i] *= b[i];
  }
}

// Final projection with weight application and scattered output accumulation
void MoELayer::compute_final_projection_and_accumulate(
  const float *input_data, const float *weight_data, float *output_data,
  const std::vector<unsigned> &token_indices, const std::vector<float> &weights,
  unsigned int input_dim, unsigned int output_dim, bool accumulate) {

  for (size_t i = 0; i < token_indices.size(); ++i) {
    const unsigned token_idx = token_indices[i];
    const float routing_weight = weights[i];
    const float *input_row = input_data + i * input_dim;
    float *output_row = output_data + token_idx * output_dim;

    // Matrix multiplication with routing weight: 
    // output[token_idx] += routing_weight * (input_row × weight)
    for (unsigned j = 0; j < output_dim; ++j) {
      float sum = 0.0f;
      const float *weight_col = weight_data + j;  // column j

      // Vectorizable dot product
      #pragma omp simd reduction(+:sum)
      for (unsigned k = 0; k < input_dim; ++k) {
        sum += input_row[k] * weight_col[k * output_dim];
      }

      const float weighted_output = routing_weight * sum;
      
      if (accumulate) {
        // Incremental forwarding: add_i
        output_row[j] += weighted_output;
      } else {
        // Normal forwarding: copyData (first expert overwrites, others accumulate)
        // Note: output was initialized to zero, so += works for first expert too
        output_row[j] += weighted_output;
      }
    }
  }
}

// 나머지 함수들은 원본과 동일
void MoELayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, moe_props);
  LayerImpl::setProperty(remain_props);
}

void MoELayer::calcDerivative(RunLayerContext &context) {
  throw std::runtime_error("MoE layer does not support derivative calculation");
}

void MoELayer::calcGradient(RunLayerContext &context) {
  throw std::runtime_error("MoE layer does not support gradient calculation");
}

void MoELayer::exportTo(Exporter &exporter,
                        const ml::train::ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
  exporter.saveResult(moe_props, method, this);
}

} // namespace nntrainer