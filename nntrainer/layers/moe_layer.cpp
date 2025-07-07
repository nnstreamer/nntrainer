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
  expert_mask_idx(std::numeric_limits<unsigned>::max()) {}

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

  // 4. Initialie gate layer (router)
  TensorDim gate_dim(
    1, is_nchw ? 1 : num_experts, is_nchw ? hidden_size : 1,
    is_nchw ? num_experts : hidden_size,
    TensorDim::TensorType(context.getFormat(), context.getWeightDataType()),
    is_nchw ? 0b0011 : 0b0101);

  gate_idx = context.requestWeight(
    gate_dim, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, "gate", true);

  // 5. Initializer expert weights
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

  // Router logits :  [batch * seq, num_experts]
  router_logits_idx = context.requestTensor(
    {total_tokens, 1, 1, num_experts}, "router_logits", Initializer::NONE,
    false, TensorLifespan::FORWARD_FUNC_LIFESPAN);

  // Expert mask: [num_experts, batch*seq]
  expert_mask_idx = context.requestTensor(
    {num_experts, 1, topk, total_tokens}, "expert_mask", Initializer::ZEROS,
    false, TensorLifespan::FORWARD_FUNC_LIFESPAN);
}

void MoELayer::forwarding(RunLayerContext &context, bool training) {
  Tensor &input = context.getInput(SINGLE_INOUT_IDX);
  Tensor &output = context.getOutput(SINGLE_INOUT_IDX);

  Tensor &router_logits = context.getTensor(router_logits_idx);
  Tensor &expert_mask = context.getTensor(expert_mask_idx);

  const unsigned batch_size = input.batch();
  const unsigned seq_len = input.height();
  const unsigned hidden_size = input.width();
  const unsigned total_tokens = batch_size * seq_len;

  // reshape input: [B,1,S,H] -> [B*S,1,1,H]
  input.reshape({total_tokens, 1, 1, hidden_size});

  // reshape output: [B,1,S,H] -> [B*S,1,1,H]
  output.reshape({total_tokens, 1, 1, hidden_size});
  output.setZero();

  // routing
  Tensor &gate_weights = context.getWeight(gate_idx);
  input.dot(gate_weights, router_logits);
  router_logits.apply(ActiFunc::softmax<float>, router_logits);
  auto topk_result = router_logits.topK(topk);
  auto topk_values = std::get<0>(topk_result);
  auto topk_indices = std::get<1>(topk_result);

  const uint32_t *indices_data = topk_indices.getData<uint32_t>();
#pragma omp parallel for collapse(2)
  for (int i = 0; i < static_cast<int>(total_tokens); ++i) {
    for (int k = 0; k < static_cast<int>(topk); ++k) {
      expert_mask.setValue(indices_data[i * topk + k], 0, k, i, 1.0f);
    }
  }

// expert forwarding
#pragma omp parallel
  {
#pragma omp for schedule(dynamic)
    for (int expert_idx = 0; expert_idx < static_cast<int>(num_experts);
         ++expert_idx) {
      std::vector<unsigned> token_indices;

      // get token indices for this expert
      for (int i = 0; i < static_cast<int>(total_tokens); ++i) {
        for (int k = 0; k < static_cast<int>(topk); ++k) {
          if (expert_mask.getValue<float>(expert_idx, 0, k, i) > 0.5f) {
            token_indices.push_back(i);
          }
        }
      }
      if (token_indices.empty())
        continue;

      // slicing and computing expert forward pass
      Tensor expert_output = compute_expert_forward(
        input.getBatchSlice(token_indices),
        topk_values.getBatchSlice(token_indices),
        context.getWeight(expert_gate_proj_indices[expert_idx]),
        context.getWeight(expert_up_proj_indices[expert_idx]),
        context.getWeight(expert_down_proj_indices[expert_idx]));

// acuumulate expert output into the main output tensor
#pragma omp critical
      {
        for (int i = 0; i < static_cast<int>(token_indices.size()); ++i) {
          unsigned idx = token_indices[i];
          auto tgt_output = output.getBatchSlice(idx, 1);
          // tgt_output.add_i(expert_output.getBatchSlice(i, 1));
          tgt_output.copyData(expert_output.getBatchSlice(i, 1));
        }
      }
    }
  }

  // reshape output: [B*S,1,1,H] -> [B,1,S,H]
  output.reshape({batch_size, 1, seq_len, hidden_size});
}

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
  ///@todo acti_func.run_fn(gate_out, gate_out); -> this inplace operation
  /// doesn't work
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
