/**
 * Copyright (C) 2020 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *   http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 *
 * @file	moe_layer.cpp
 * @date	09 June 2025
 * @brief	This is a Mixture of Expert Layer Class for Neural Network
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Eunju Yang <ej.yang@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#include <acti_func.h>
#include <cmath>
#include <moe_layer.h>
#include <node_exporter.h>
#include <omp.h>
#include <stdexcept>
#include <algorithm>
#include <cstring>

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
  temp_gate_out_indices(MAX_THREADS, std::numeric_limits<unsigned>::max()),
  temp_up_out_indices(MAX_THREADS, std::numeric_limits<unsigned>::max()),
  temp_intermediate_indices(MAX_THREADS, std::numeric_limits<unsigned>::max()),
  temp_expert_input_indices(MAX_THREADS, std::numeric_limits<unsigned>::max()),
  temp_expert_output_indices(MAX_THREADS, std::numeric_limits<unsigned>::max()) {}

void MoELayer::finalize(InitLayerContext &context) {

  // 1. Validate input/output dimensions
  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "MoE layer only supports single input";

  auto &weight_regularizer =
    std::get<props::WeightRegularizer>(*layer_impl_props);
  auto &weight_regularizer_constant =
    std::get<props::WeightRegularizerConstant>(*layer_impl_props);
  auto &weight_initializer =
    std::get<props::WeightInitializer>(*layer_impl_props);
  auto &weight_decay = std::get<props::WeightDecay>(*layer_impl_props);

  // 2. Set output dimensions (same as input)
  const auto &in_dim = context.getInputDimensions()[SINGLE_INOUT_IDX];
  const bool is_nchw = context.getFormat() == Tformat::NCHW;
  std::vector<TensorDim> output_dims(1);
  output_dims[SINGLE_INOUT_IDX] = in_dim;
  context.setOutputDimensions(output_dims);

  // 3. Get MoE properties
  num_experts = std::get<props::NumExperts>(moe_props).get();
  topk = std::get<props::NumExpertsPerToken>(moe_props).get();
  const unsigned int intermediate_size = std::get<props::Unit>(moe_props).get();
  const unsigned int hidden_size = in_dim.width(); // Feature dimension

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

  // 4. Initialize gate layer (router)
  TensorDim gate_dim(
    1, is_nchw ? 1 : num_experts, is_nchw ? hidden_size : 1,
    is_nchw ? num_experts : hidden_size,
    TensorDim::TensorType(context.getFormat(), context.getWeightDataType()),
    is_nchw ? 0b0011 : 0b0101);

  gate_idx = context.requestWeight(
    gate_dim, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, "gate", true);

  // 5. Initialize expert weights
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
    // Gate projection
    expert_gate_proj_indices.push_back(context.requestWeight(
      expert_gate_dim, weight_initializer, weight_regularizer,
      weight_regularizer_constant, weight_decay,
      "expert_gate_" + std::to_string(i), true));

    // Up projection
    expert_up_proj_indices.push_back(context.requestWeight(
      expert_gate_dim, // Same dimensions as gate projection
      weight_initializer, weight_regularizer, weight_regularizer_constant,
      weight_decay, "expert_up_" + std::to_string(i), false));

    // Down projection
    expert_down_proj_indices.push_back(context.requestWeight(
      expert_down_dim, weight_initializer, weight_regularizer,
      weight_regularizer_constant, weight_decay,
      "expert_down_" + std::to_string(i), false));
  }

  // 6. Request intermediate tensors
  const unsigned batch_size = in_dim.batch();
  const unsigned seq_len = in_dim.height();
  const unsigned total_tokens = batch_size * seq_len;

  // Router logits: [batch * seq, num_experts]
  router_logits_idx = context.requestTensor(
    {total_tokens, 1, 1, num_experts}, "router_logits", Initializer::NONE,
    false, TensorLifespan::FORWARD_FUNC_LIFESPAN);

  // Pre-allocate thread-local temporary tensors for efficient computation
  // Maximum tokens any single expert might process
  const unsigned max_expert_tokens = total_tokens;
  
  // Determine actual number of threads to use (avoid over-allocation)
  const int max_threads = std::min(MAX_THREADS, std::max(1, static_cast<int>(num_experts)));
  
  // Resize vectors to actual thread count
  temp_gate_out_indices.resize(max_threads);
  temp_up_out_indices.resize(max_threads);
  temp_intermediate_indices.resize(max_threads);
  temp_expert_input_indices.resize(max_threads);
  temp_expert_output_indices.resize(max_threads);
  
  // Allocate one set of temporary tensors for each thread
  for (int thread_id = 0; thread_id < max_threads; ++thread_id) {
    temp_gate_out_indices[thread_id] = context.requestTensor(
      {max_expert_tokens, 1, 1, intermediate_size}, 
      "temp_gate_out_" + std::to_string(thread_id), Initializer::NONE,
      false, TensorLifespan::FORWARD_FUNC_LIFESPAN);
      
    temp_up_out_indices[thread_id] = context.requestTensor(
      {max_expert_tokens, 1, 1, intermediate_size}, 
      "temp_up_out_" + std::to_string(thread_id), Initializer::NONE,
      false, TensorLifespan::FORWARD_FUNC_LIFESPAN);
      
    temp_intermediate_indices[thread_id] = context.requestTensor(
      {max_expert_tokens, 1, 1, intermediate_size}, 
      "temp_intermediate_" + std::to_string(thread_id), Initializer::NONE,
      false, TensorLifespan::FORWARD_FUNC_LIFESPAN);
      
    temp_expert_input_indices[thread_id] = context.requestTensor(
      {max_expert_tokens, 1, 1, hidden_size}, 
      "temp_expert_input_" + std::to_string(thread_id), Initializer::NONE,
      false, TensorLifespan::FORWARD_FUNC_LIFESPAN);
      
    temp_expert_output_indices[thread_id] = context.requestTensor(
      {max_expert_tokens, 1, 1, hidden_size}, 
      "temp_expert_output_" + std::to_string(thread_id), Initializer::NONE,
      false, TensorLifespan::FORWARD_FUNC_LIFESPAN);
  }

  // Initialize routing cache
  routing_cache.expert_token_indices.resize(num_experts);
  routing_cache.expert_token_weights.resize(num_experts);
  routing_cache.token_expert_counts.resize(total_tokens);
}

void MoELayer::forwarding(RunLayerContext &context, bool training) {
  Tensor &input = context.getInput(SINGLE_INOUT_IDX);
  Tensor &output = context.getOutput(SINGLE_INOUT_IDX);
  Tensor &router_logits = context.getTensor(router_logits_idx);

  const unsigned batch_size = input.batch();
  const unsigned seq_len = input.height();
  const unsigned hidden_size = input.width();
  const unsigned total_tokens = batch_size * seq_len;

  // Reshape input: [B,1,S,H] -> [B*S,1,1,H]
  input.reshape({total_tokens, 1, 1, hidden_size});
  output.reshape({total_tokens, 1, 1, hidden_size});
  output.setZero();

  // Compute router logits and apply softmax
  Tensor &gate_weights = context.getWeight(gate_idx);
  input.dot(gate_weights, router_logits);
  router_logits.apply(ActiFunc::softmax<float>, router_logits);

  // Compute optimized routing information
  compute_routing_optimized(router_logits, routing_cache);

  // Get raw data pointers for efficient access
  const float* input_data = input.getData<float>();
  float* output_data = output.getData<float>();

  // Process experts in parallel
#pragma omp parallel for schedule(dynamic)
  for (int expert_idx = 0; expert_idx < static_cast<int>(num_experts); ++expert_idx) {
    const auto& token_indices = routing_cache.expert_token_indices[expert_idx];
    const auto& token_weights = routing_cache.expert_token_weights[expert_idx];
    
    if (token_indices.empty()) continue;

    // Use optimized expert forward computation
    compute_expert_forward_optimized(
      input_data, output_data, token_indices, token_weights,
      context.getWeight(expert_gate_proj_indices[expert_idx]),
      context.getWeight(expert_up_proj_indices[expert_idx]),
      context.getWeight(expert_down_proj_indices[expert_idx]),
      context);
  }

  // Reshape output: [B*S,1,1,H] -> [B,1,S,H]
  output.reshape({batch_size, 1, seq_len, hidden_size});
}

void MoELayer::compute_routing_optimized(const Tensor& router_logits, RoutingInfo& routing_info) {
  const unsigned total_tokens = router_logits.batch();
  const float* logits_data = router_logits.getData<float>();
  
  // Clear previous routing information
  for (auto& indices : routing_info.expert_token_indices) {
    indices.clear();
  }
  for (auto& weights : routing_info.expert_token_weights) {
    weights.clear();
  }
  std::fill(routing_info.token_expert_counts.begin(), routing_info.token_expert_counts.end(), 0);

  // Reserve space to avoid frequent reallocations
  const size_t estimated_tokens_per_expert = (total_tokens * topk) / num_experts + 1;
  for (unsigned int i = 0; i < num_experts; ++i) {
    routing_info.expert_token_indices[i].reserve(estimated_tokens_per_expert);
    routing_info.expert_token_weights[i].reserve(estimated_tokens_per_expert);
  }

  // Process tokens efficiently without creating intermediate tensors
  for (unsigned int token_idx = 0; token_idx < total_tokens; ++token_idx) {
    const float* token_logits = logits_data + token_idx * num_experts;
    
    // Find top-k experts for this token using partial sort
    std::vector<std::pair<float, unsigned int>> expert_probs;
    expert_probs.reserve(num_experts);
    
    for (unsigned int expert_idx = 0; expert_idx < num_experts; ++expert_idx) {
      expert_probs.emplace_back(token_logits[expert_idx], expert_idx);
    }
    
    // Partial sort to get top-k
    std::nth_element(expert_probs.begin(), expert_probs.begin() + topk, expert_probs.end(),
                     [](const auto& a, const auto& b) { return a.first > b.first; });
    
    // Add token to selected experts
    for (unsigned int k = 0; k < topk; ++k) {
      const unsigned int expert_idx = expert_probs[k].second;
      const float weight = expert_probs[k].first;
      
      routing_info.expert_token_indices[expert_idx].push_back(token_idx);
      routing_info.expert_token_weights[expert_idx].push_back(weight);
    }
    
    routing_info.token_expert_counts[token_idx] = topk;
  }
}

void MoELayer::compute_expert_forward_optimized(
    const float* input_data,
    float* output_data,
    const std::vector<unsigned int>& token_indices,
    const std::vector<float>& token_weights,
    const Tensor& gate_proj,
    const Tensor& up_proj,
    const Tensor& down_proj,
    RunLayerContext& context) {
    
  const unsigned int num_tokens = token_indices.size();
  if (num_tokens == 0) return;
  
  const unsigned int hidden_size = gate_proj.height();
  const unsigned int intermediate_size = gate_proj.width();
  
  // Get thread-specific temporary tensors to avoid race conditions
  const int thread_id = omp_get_thread_num();
  const int safe_thread_id = std::min(thread_id, static_cast<int>(temp_gate_out_indices.size()) - 1);  // Bounds check
  
  Tensor& temp_gate_out = context.getTensor(temp_gate_out_indices[safe_thread_id]);
  Tensor& temp_up_out = context.getTensor(temp_up_out_indices[safe_thread_id]);  
  Tensor& temp_intermediate = context.getTensor(temp_intermediate_indices[safe_thread_id]);
  Tensor& temp_expert_input = context.getTensor(temp_expert_input_indices[safe_thread_id]);
  Tensor& temp_expert_output = context.getTensor(temp_expert_output_indices[safe_thread_id]);
  
  // Resize temporary tensors for current batch
  temp_gate_out.reshape({num_tokens, 1, 1, intermediate_size});
  temp_up_out.reshape({num_tokens, 1, 1, intermediate_size});
  temp_intermediate.reshape({num_tokens, 1, 1, intermediate_size});
  temp_expert_input.reshape({num_tokens, 1, 1, hidden_size});
  temp_expert_output.reshape({num_tokens, 1, 1, hidden_size});
  
  // Get raw data pointers
  float* gate_out_data = temp_gate_out.getData<float>();
  float* up_out_data = temp_up_out.getData<float>();
  float* intermediate_data = temp_intermediate.getData<float>();
  float* expert_input_data = temp_expert_input.getData<float>();
  float* expert_output_data = temp_expert_output.getData<float>();
  
  const float* gate_proj_data = gate_proj.getData<float>();
  const float* up_proj_data = up_proj.getData<float>();
  const float* down_proj_data = down_proj.getData<float>();

  // Gather input tokens efficiently
#pragma omp parallel for
  for (int i = 0; i < static_cast<int>(num_tokens); ++i) {
    const unsigned int token_idx = token_indices[i];
    const float* src = input_data + token_idx * hidden_size;
    float* dst = expert_input_data + i * hidden_size;
    std::memcpy(dst, src, hidden_size * sizeof(float));
  }

  // Batch GEMM operations for better cache utilization
  // Gate projection: input * gate_proj -> gate_out
  batched_gemm(temp_expert_input, gate_proj, temp_gate_out, 
               std::vector<unsigned int>(num_tokens));
  
  // Up projection: input * up_proj -> up_out  
  batched_gemm(temp_expert_input, up_proj, temp_up_out,
               std::vector<unsigned int>(num_tokens));

  // Apply activation and multiply: silu(gate_out) * up_out -> intermediate
#pragma omp parallel for
  for (int i = 0; i < static_cast<int>(num_tokens); ++i) {
    for (unsigned int j = 0; j < intermediate_size; ++j) {
      const unsigned int idx = i * intermediate_size + j;
      const float gate_val = gate_out_data[idx];
      // SiLU activation: x * sigmoid(x)
      const float activated = gate_val / (1.0f + std::exp(-gate_val));
      intermediate_data[idx] = activated * up_out_data[idx];
    }
  }

  // Down projection: intermediate * down_proj -> expert_output
  batched_gemm(temp_intermediate, down_proj, temp_expert_output,
               std::vector<unsigned int>(num_tokens));

  // Apply routing weights and scatter to output (with atomic operations for thread safety)
#pragma omp parallel for
  for (int i = 0; i < static_cast<int>(num_tokens); ++i) {
    const unsigned int token_idx = token_indices[i];
    const float weight = token_weights[i];
    
    const float* src = expert_output_data + i * hidden_size;
    float* dst = output_data + token_idx * hidden_size;
    
    // Apply weight and accumulate to output
    for (unsigned int j = 0; j < hidden_size; ++j) {
      const float weighted_val = src[j] * weight;
      #pragma omp atomic
      dst[j] += weighted_val;
    }
  }
}

void MoELayer::batched_gemm(const Tensor& input, const Tensor& weight, Tensor& output,
                           const std::vector<unsigned int>& token_indices) {
  // Use the tensor's built-in dot product which should be optimized
  input.dot(weight, output);
}

// Keep the original compute_expert_forward for compatibility
inline Tensor MoELayer::compute_expert_forward(const Tensor &input,
                                               const Tensor &weights,
                                               const Tensor &gate_proj,
                                               const Tensor &up_proj,
                                               const Tensor &down_proj) {

  const unsigned tokens = input.batch();
  const unsigned hidden_size = input.width();
  const unsigned intermediate_size = gate_proj.width();

  // Gate projection: [tokens,1,1,H] x [H,I] -> [tokens,1,1,I]
  Tensor gate_out(tokens, 1, 1, intermediate_size);
  Tensor acti_out(tokens, 1, 1, intermediate_size);
  input.dot(gate_proj, gate_out);

  // Apply activation (silu)
  acti_func.run_fn(gate_out, acti_out);

  // Up projection: [tokens,1,1,H] x [H,I] -> [tokens,1,1,I]
  Tensor up_out(tokens, 1, 1, intermediate_size);
  input.dot(up_proj, up_out);

  // Multiply: silu(gate_out) * up_out
  acti_out.multiply_i(up_out);

  // Down projection: [tokens,1,1,I] x [I,H] -> [tokens,1,1,H]
  Tensor expert_output(tokens, 1, 1, hidden_size);
  acti_out.dot(down_proj, expert_output);

  // Weight by routing scores (broadcast multiply)
  for (unsigned i = 0; i < tokens; ++i) {
    float weight_val = weights.getValue(i, 0, 0, 0);
    auto weighted_expert_output = expert_output.getBatchSlice(i, 1);
    weighted_expert_output.multiply_i(weight_val);
  }

  return expert_output;
}

void MoELayer::incremental_forwarding(RunLayerContext &context,
                                      unsigned int from, unsigned int to,
                                      bool training) {}

void MoELayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, moe_props);
  LayerImpl::setProperty(remain_props);
}

void MoELayer::calcDerivative(RunLayerContext &context) {
  // MoE layer does not support derivative calculation
  throw std::runtime_error("MoE layer does not support derivative calculation");
}

void MoELayer::calcGradient(RunLayerContext &context) {
  // MoE layer does not support gradient calculation
  throw std::runtime_error("MoE layer does not support gradient calculation");
}

void MoELayer::exportTo(Exporter &exporter,
                        const ml::train::ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
  exporter.saveResult(moe_props, method, this); // Save MoE specific properties
}

} // namespace nntrainer
