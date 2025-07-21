// SPDX-License-Identifier: Apache-2.0
/**
 * @file   moe_layer_simplified.cpp
 * @date   15 January 2025
 * @brief  Simplified and Efficient MoE Layer implementation
 * @see    https://github.com/EunjuYang/nntrainer/tree/feat/moe_layer_update
 * @author NNTrainer Team
 * @bug    No known bugs except for NYI items
 *
 * Simplified optimizations:
 * 1. Direct tensor operations without getBatchSlice
 * 2. In-place operations to minimize memory usage
 * 3. Cache-friendly access patterns
 * 4. Efficient top-k selection using existing patterns
 * 5. Minimal temporary allocations
 */

#include <moe_layer.h>

#include <cmath>
#include <algorithm>
#include <numeric>

#include <layer_context.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <util_func.h>

namespace nntrainer {

static constexpr size_t SINGLE = 1;

MoELayer::MoELayer() : 
    Layer(), 
    num_experts(8), 
    top_k(2), 
    expert_capacity(0) {
}

MoELayer::~MoELayer() = default;

void MoELayer::finalize(InitLayerContext &context) {
    NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
        << "MoE layer takes exactly one input";

    NNTR_THROW_IF(context.getNumOutputs() != 1, std::invalid_argument)
        << "MoE layer produces exactly one output";

    const TensorDim &input_dim = context.getInputDimensions()[0];
    
    // Validate input dimensions
    NNTR_THROW_IF(input_dim.batch() == 0 || input_dim.height() == 0 || 
                  input_dim.width() == 0 || input_dim.channel() == 0, 
                  std::invalid_argument)
        << "Input dimensions must be positive, got: " << input_dim;

    // Set output dimensions (same as input for MoE)
    context.setOutputDimensions({input_dim});

    // Calculate expert capacity for load balancing
    const unsigned int sequence_length = input_dim.height() * input_dim.width();
    expert_capacity = (sequence_length * top_k) / num_experts + 1;

    // Initialize gate network dimensions
    const unsigned int feature_dim = input_dim.channel();
    
    // Pre-allocate tensors for better memory management
    gate_weights = Tensor(TensorDim(1, 1, feature_dim, num_experts));
    gate_bias = Tensor(TensorDim(1, 1, 1, num_experts));
    
    // Initialize expert networks (simplified as linear layers)
    expert_weights.reserve(num_experts);
    expert_bias.reserve(num_experts);
    
    for (unsigned int i = 0; i < num_experts; ++i) {
        expert_weights.emplace_back(TensorDim(1, 1, feature_dim, feature_dim));
        expert_bias.emplace_back(TensorDim(1, 1, 1, feature_dim));
    }
}

void MoELayer::forwarding(RunLayerContext &context, bool training) {
    const Tensor &input = context.getInput(SINGLE - 1);
    Tensor &output = context.getOutput(SINGLE - 1);
    
    // Get tensor dimensions
    const auto batch_size = input.batch();
    const auto seq_length = input.height() * input.width();
    const auto feature_dim = input.channel();
    
    // Step 1: Compute gate scores directly on input tensor
    Tensor gate_scores(TensorDim(batch_size, 1, seq_length, num_experts));
    compute_gate_scores(input, gate_scores);
    
    // Step 2: Apply softmax in-place
    apply_softmax_inplace(gate_scores);
    
    // Step 3: Select top-k experts and compute output directly
    compute_moe_output(input, gate_scores, output);
}

void MoELayer::compute_gate_scores(const Tensor &input, Tensor &gate_scores) {
    const auto batch_size = input.batch();
    const auto seq_length = input.height() * input.width();
    const auto feature_dim = input.channel();
    
    // Direct pointer access for efficiency
    const float *input_data = input.getData();
    float *gate_data = gate_scores.getData();
    const float *weight_data = gate_weights.getData();
    const float *bias_data = gate_bias.getData();
    
    // Compute gate scores: gate_scores = input * gate_weights + gate_bias
    for (unsigned int b = 0; b < batch_size; ++b) {
        for (unsigned int s = 0; s < seq_length; ++s) {
            const unsigned int input_offset = b * seq_length * feature_dim + s * feature_dim;
            const unsigned int gate_offset = b * seq_length * num_experts + s * num_experts;
            
            // Compute scores for all experts at once
            for (unsigned int e = 0; e < num_experts; ++e) {
                float score = bias_data[e];
                
                // Dot product: input[s] · gate_weights[:, e]
                for (unsigned int f = 0; f < feature_dim; ++f) {
                    score += input_data[input_offset + f] * weight_data[f * num_experts + e];
                }
                
                gate_data[gate_offset + e] = score;
            }
        }
    }
}

void MoELayer::apply_softmax_inplace(Tensor &gate_scores) {
    const auto batch_size = gate_scores.batch();
    const auto seq_length = gate_scores.height() * gate_scores.width();
    
    float *gate_data = gate_scores.getData();
    
    // Apply softmax to each sequence position
    for (unsigned int b = 0; b < batch_size; ++b) {
        for (unsigned int s = 0; s < seq_length; ++s) {
            const unsigned int offset = b * seq_length * num_experts + s * num_experts;
            float *scores = gate_data + offset;
            
            // Find max for numerical stability
            float max_score = scores[0];
            for (unsigned int e = 1; e < num_experts; ++e) {
                max_score = std::max(max_score, scores[e]);
            }
            
            // Compute exp and sum
            float sum = 0.0f;
            for (unsigned int e = 0; e < num_experts; ++e) {
                scores[e] = std::exp(scores[e] - max_score);
                sum += scores[e];
            }
            
            // Normalize
            const float inv_sum = 1.0f / (sum + 1e-8f);
            for (unsigned int e = 0; e < num_experts; ++e) {
                scores[e] *= inv_sum;
            }
        }
    }
}

void MoELayer::compute_moe_output(const Tensor &input, const Tensor &gate_scores, Tensor &output) {
    const auto batch_size = input.batch();
    const auto seq_length = input.height() * input.width();
    const auto feature_dim = input.channel();
    
    // Direct pointer access
    const float *input_data = input.getData();
    const float *gate_data = gate_scores.getData();
    float *output_data = output.getData();
    
    // Initialize output to zero
    output.setZero();
    
    // Process each sequence position
    for (unsigned int b = 0; b < batch_size; ++b) {
        for (unsigned int s = 0; s < seq_length; ++s) {
            const unsigned int input_offset = b * seq_length * feature_dim + s * feature_dim;
            const unsigned int gate_offset = b * seq_length * num_experts + s * num_experts;
            const unsigned int output_offset = b * seq_length * feature_dim + s * feature_dim;
            
            // Get top-k experts using partial sort (following nntrainer patterns)
            std::vector<std::pair<float, int>> expert_scores;
            expert_scores.reserve(num_experts);
            
            for (unsigned int e = 0; e < num_experts; ++e) {
                expert_scores.emplace_back(gate_data[gate_offset + e], e);
            }
            
            // Partial sort to get top-k (O(n log k) complexity)
            std::partial_sort(expert_scores.begin(), 
                            expert_scores.begin() + top_k, 
                            expert_scores.end(),
                            std::greater<std::pair<float, int>>());
            
            // Normalize top-k weights
            float weight_sum = 0.0f;
            for (unsigned int k = 0; k < top_k; ++k) {
                weight_sum += expert_scores[k].first;
            }
            
            if (weight_sum < 1e-8f) continue; // Skip if no significant weights
            
            // Compute weighted expert outputs
            for (unsigned int k = 0; k < top_k; ++k) {
                const int expert_id = expert_scores[k].second;
                const float weight = expert_scores[k].first / weight_sum;
                
                // Compute expert output: expert_output = input * expert_weights + expert_bias
                compute_expert_contribution(input_data + input_offset, 
                                          output_data + output_offset,
                                          expert_id, weight, feature_dim);
            }
        }
    }
}

void MoELayer::compute_expert_contribution(const float *input_data, 
                                         float *output_data,
                                         int expert_id, 
                                         float weight, 
                                         unsigned int feature_dim) {
    const float *expert_weight_data = expert_weights[expert_id].getData();
    const float *expert_bias_data = expert_bias[expert_id].getData();
    
    // Expert computation: output += weight * (input * expert_weights + expert_bias)
    for (unsigned int out_f = 0; out_f < feature_dim; ++out_f) {
        float expert_output = expert_bias_data[out_f];
        
        // Matrix multiplication: input * expert_weights[:, out_f]
        for (unsigned int in_f = 0; in_f < feature_dim; ++in_f) {
            expert_output += input_data[in_f] * expert_weight_data[in_f * feature_dim + out_f];
        }
        
        // Add weighted contribution to final output
        output_data[out_f] += weight * expert_output;
    }
}

void MoELayer::calcDerivative(RunLayerContext &context) {
    // Simplified backward pass implementation
    const Tensor &incoming_derivative = context.getIncomingDerivative(SINGLE - 1);
    Tensor &outgoing_derivative = context.getOutgoingDerivative(SINGLE - 1);
    
    // For now, just pass through the derivative
    // In a complete implementation, this would compute gradients for gate and expert parameters
    outgoing_derivative.copy(incoming_derivative);
}

void MoELayer::exportTo(Exporter &exporter, const ml::train::ExportMethods& method) const {
    LayerImpl::exportTo(exporter, method);
    exporter.saveResult(MoELayer::type, "num_experts", num_experts);
    exporter.saveResult(MoELayer::type, "top_k", top_k);
}

void MoELayer::setProperty(const std::vector<std::string> &values) {
    auto remain_props = loadProperties(values, {
        {"num_experts", [this](const std::string &val) { num_experts = std::stoi(val); }},
        {"top_k", [this](const std::string &val) { top_k = std::stoi(val); }}
    });
    
    NNTR_THROW_IF(!remain_props.empty(), std::invalid_argument)
        << "Unknown properties: " << util::join(remain_props, ",");
}

} // namespace nntrainer