// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file main.cpp
 * @date 09 January 2025
 * @brief Qwen3 MoE Application Implementation
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Samsung Electronics
 * @bug No known bugs except for NYI items
 */

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <vector>

#include <app_context.h>
#include <dataset.h>
#include <layer.h>
#include <model.h>
#include <optimizer.h>

// Custom layers for Qwen3 MoE (implemented)
#include "silu_layer.h"
#include "rms_norm_layer.h"
#include "qwen3_moe_mlp_layer.h"

// TODO: Implement these layers in future phases
// #include "qwen3_moe_attention_layer.h"
// #include "qwen3_moe_sparse_block_layer.h"
// #include "grouped_query_attention_layer.h"

using namespace nntrainer;

/**
 * @brief Qwen3 MoE Configuration
 */
struct Qwen3MoeConfig {
  unsigned int vocab_size = 151936;
  unsigned int hidden_size = 2048;
  unsigned int intermediate_size = 6144;
  unsigned int num_hidden_layers = 48;
  unsigned int num_attention_heads = 32;
  unsigned int num_key_value_heads = 4;
  unsigned int max_position_embeddings = 32768;
  unsigned int num_experts = 128;
  unsigned int num_experts_per_tok = 8;
  unsigned int moe_intermediate_size = 768;
  float rms_norm_eps = 1e-6;
  float rope_theta = 10000.0;
  bool use_sliding_window = false;
  unsigned int sliding_window = 4096;
  unsigned int decoder_sparse_step = 1;
  float router_aux_loss_coef = 0.001;
  
  // For simplified demo version
  bool demo_mode = true;
  unsigned int demo_num_layers = 2;  // Reduced for demo
  unsigned int demo_num_experts = 4; // Reduced for demo
};

/**
 * @brief Create Qwen3 MoE Decoder Layer (Simplified)
 */
std::vector<std::shared_ptr<Layer>> createQwen3MoeDecoderLayer(
  const Qwen3MoeConfig& config, unsigned int layer_idx) {
  
  std::vector<std::shared_ptr<Layer>> layers;
  std::string layer_prefix = "decoder_" + std::to_string(layer_idx) + "_";
  
  // Input Layer Norm
  auto input_norm = createLayer("rms_norm", {
    "name=" + layer_prefix + "input_norm",
    "epsilon=" + std::to_string(config.rms_norm_eps)
  });
  layers.push_back(input_norm);
  
  // TODO: Replace with proper Grouped Query Attention
  // For now, use regular multi-head attention as placeholder
  auto attention = createLayer("multi_head_attention", {
    "name=" + layer_prefix + "attention",
    "num_heads=" + std::to_string(config.num_attention_heads),
    "projected_output_size=" + std::to_string(config.hidden_size),
    "output_shape=" + std::to_string(config.hidden_size)
  });
  layers.push_back(attention);
  
  // Post Attention Layer Norm
  auto post_attention_norm = createLayer("rms_norm", {
    "name=" + layer_prefix + "post_attention_norm",
    "epsilon=" + std::to_string(config.rms_norm_eps)
  });
  layers.push_back(post_attention_norm);
  
  // For demo mode, always use regular MLP
  // TODO: Implement sparse MoE block for full model
  auto mlp = createLayer("qwen3_moe_mlp", {
    "name=" + layer_prefix + "mlp",
    "hidden_size=" + std::to_string(config.hidden_size),
    "intermediate_size=" + std::to_string(config.demo_mode ? 
      config.moe_intermediate_size : config.intermediate_size)
  });
  layers.push_back(mlp);
  
  return layers;
}

/**
 * @brief Create Simplified Qwen3 MoE Model
 */
std::shared_ptr<Model> createQwen3MoeModel(const Qwen3MoeConfig& config) {
  auto model = createModel(ModelType::NEURAL_NETWORK);
  
  // Token Embedding
  auto embedding = createLayer("embedding", {
    "name=token_embedding",
    "in_dim=" + std::to_string(config.vocab_size),
    "out_dim=" + std::to_string(config.hidden_size)
  });
  model->addLayer(embedding);
  
  // Decoder Layers
  unsigned int num_layers = config.demo_mode ? config.demo_num_layers : config.num_hidden_layers;
  for (unsigned int i = 0; i < num_layers; ++i) {
    auto decoder_layers = createQwen3MoeDecoderLayer(config, i);
    for (auto& layer : decoder_layers) {
      model->addLayer(layer);
    }
  }
  
  // Final Layer Norm
  auto final_norm = createLayer("rms_norm", {
    "name=final_norm",
    "epsilon=" + std::to_string(config.rms_norm_eps)
  });
  model->addLayer(final_norm);
  
  // Language Model Head
  auto lm_head = createLayer("fully_connected", {
    "name=lm_head",
    "unit=" + std::to_string(config.vocab_size),
    "bias=false"
  });
  model->addLayer(lm_head);
  
  return model;
}

/**
 * @brief Simple text generation function
 */
std::vector<int> generateText(std::shared_ptr<Model> model, 
                            const std::vector<int>& input_ids,
                            unsigned int max_length = 50,
                            float temperature = 1.0) {
  std::vector<int> generated = input_ids;
  std::random_device rd;
  std::mt19937 gen(rd());
  
  std::cout << "Starting text generation..." << std::endl;
  
  for (unsigned int i = input_ids.size(); i < max_length; ++i) {
    // Create input tensor (simplified for demo)
    TensorDim input_dim({1, (unsigned int)generated.size(), 1});
    auto input_tensor = std::make_shared<Tensor>(input_dim);
    
    // Fill input tensor with token IDs
    float* data = input_tensor->getData();
    for (size_t j = 0; j < generated.size(); ++j) {
      data[j] = static_cast<float>(generated[j]);
    }
    
    // For demo, just generate random tokens
    // TODO: Implement proper forward pass when model is complete
    std::uniform_int_distribution<> token_dist(1, 1000);
    int next_token = token_dist(gen);
    
    generated.push_back(next_token);
    
    // Break on end token (assuming token 2 is EOS)
    if (next_token == 2) break;
  }
  
  return generated;
}

/**
 * @brief Load model weights from file
 */
bool loadModelWeights(std::shared_ptr<Model> model, const std::string& weight_path) {
  try {
    model->load(weight_path);
    std::cout << "Model weights loaded successfully from: " << weight_path << std::endl;
    return true;
  } catch (const std::exception& e) {
    std::cerr << "Failed to load model weights: " << e.what() << std::endl;
    return false;
  }
}

/**
 * @brief Print model summary
 */
void printModelSummary(std::shared_ptr<Model> model, const Qwen3MoeConfig& config) {
  std::cout << "\n=== Qwen3 MoE Model Summary ===" << std::endl;
  std::cout << "Vocabulary Size: " << config.vocab_size << std::endl;
  std::cout << "Hidden Size: " << config.hidden_size << std::endl;
  std::cout << "Number of Layers: " << (config.demo_mode ? config.demo_num_layers : config.num_hidden_layers) << std::endl;
  std::cout << "Attention Heads: " << config.num_attention_heads << std::endl;
  std::cout << "Key-Value Heads: " << config.num_key_value_heads << std::endl;
  std::cout << "Number of Experts: " << (config.demo_mode ? config.demo_num_experts : config.num_experts) << std::endl;
  std::cout << "Experts per Token: " << config.num_experts_per_tok << std::endl;
  std::cout << "Demo Mode: " << (config.demo_mode ? "ON" : "OFF") << std::endl;
  std::cout << "Model Type: Qwen3 MoE (Simplified Demo Version)" << std::endl;
  std::cout << "Implementation Status: Phase 1 - Basic Components" << std::endl;
  std::cout << "================================\n" << std::endl;
}

/**
 * @brief Main function
 */
int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cout << "Qwen3 MoE Application - nntrainer Implementation" << std::endl;
    std::cout << "Usage: " << argv[0] << " <weight_path> [demo_mode] [temperature]" << std::endl;
    std::cout << "  weight_path: Path to model weights file (.bin)" << std::endl;
    std::cout << "  demo_mode: 0 or 1 (default: 1 for simplified demo)" << std::endl;
    std::cout << "  temperature: Sampling temperature (default: 1.0)" << std::endl;
    std::cout << "\nNote: This is a Phase 1 implementation with basic components." << std::endl;
    std::cout << "Full MoE functionality will be available in future phases." << std::endl;
    return 1;
  }
  
  std::string weight_path = argv[1];
  bool demo_mode = (argc > 2) ? (std::stoi(argv[2]) != 0) : true;
  float temperature = (argc > 3) ? std::stof(argv[3]) : 1.0f;
  
  try {
    // Initialize nntrainer context
    auto app_context = AppContext::getGlobalAppContext();
    
    // Configure Qwen3 MoE
    Qwen3MoeConfig config;
    config.demo_mode = demo_mode;
    
    std::cout << "Initializing Qwen3 MoE Model..." << std::endl;
    
    // Create model
    auto model = createQwen3MoeModel(config);
    
    // Print model summary
    printModelSummary(model, config);
    
    // Load weights if provided
    if (!weight_path.empty() && weight_path != "dummy") {
      if (!loadModelWeights(model, weight_path)) {
        std::cerr << "Warning: Could not load weights. Using random initialization." << std::endl;
      }
    } else {
      std::cout << "Using random initialization (no weights file provided)." << std::endl;
    }
    
    // Initialize model
    model->compile();
    model->initialize();
    
    std::cout << "Model initialization complete!" << std::endl;
    
    // Simple demo: Generate text from prompt
    std::cout << "\n=== Text Generation Demo ===" << std::endl;
    std::vector<int> prompt = {1, 50, 25, 10}; // Example token IDs
    
    auto start_time = std::chrono::high_resolution_clock::now();
    auto generated = generateText(model, prompt, 20, temperature);
    auto end_time = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    std::cout << "Input tokens: ";
    for (int token : prompt) {
      std::cout << token << " ";
    }
    std::cout << std::endl;
    
    std::cout << "Generated tokens: ";
    for (size_t i = prompt.size(); i < generated.size(); ++i) {
      std::cout << generated[i] << " ";
    }
    std::cout << std::endl;
    
    std::cout << "Generation time: " << duration.count() << " ms" << std::endl;
    std::cout << "Tokens generated: " << (generated.size() - prompt.size()) << std::endl;
    
    std::cout << "\n=== Qwen3 MoE Demo Complete ===" << std::endl;
    std::cout << "This was a Phase 1 demo. Future phases will include:" << std::endl;
    std::cout << "- Grouped Query Attention (GQA)" << std::endl;
    std::cout << "- Complete Sparse MoE Block" << std::endl;
    std::cout << "- Rotary Position Embedding" << std::endl;
    std::cout << "- Load balancing and routing" << std::endl;
    
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
  
  return 0;
}