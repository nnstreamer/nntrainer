// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd. All Rights Reserved.
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
 * @file main.cpp
 * @date 25 January 2025
 * @brief Main entry point for CausalLM application
 * @author Samsung Electronics Co., Ltd.
 * @bug No known bugs except for NYI items
 */

#include <iostream>
#include <string>
#include <memory>

#include <causal_lm.h>
#include <qwen3_causallm.h>
#include <qwen3_moe_causallm.h>
#include <factory.h>

using namespace causallm;

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " <model_directory>" << std::endl;
    std::cout << std::endl;
    std::cout << "Arguments:" << std::endl;
    std::cout << "  model_directory  Path to directory containing model files:" << std::endl;
    std::cout << "                   - config.json" << std::endl;
    std::cout << "                   - generation_config.json" << std::endl;
    std::cout << "                   - nntr_config.json" << std::endl;
    std::cout << "                   - tokenizer.json" << std::endl;
    std::cout << "                   - model weight file (.bin)" << std::endl;
    std::cout << std::endl;
    std::cout << "Example:" << std::endl;
    std::cout << "  " << program_name << " /path/to/qwen3-4b/" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        print_usage(argv[0]);
        return 1;
    }

    std::string model_dir = argv[1];
    
    try {
        // Load configuration files
        std::string config_path = model_dir + "/config.json";
        std::string generation_config_path = model_dir + "/generation_config.json";
        std::string nntr_config_path = model_dir + "/nntr_config.json";
        
        json config = LoadJsonFile(config_path);
        json generation_config = LoadJsonFile(generation_config_path);
        json nntr_config = LoadJsonFile(nntr_config_path);
        
        // Determine model architecture and create appropriate model
        std::string architecture = config["architectures"][0];
        std::unique_ptr<CausalLM> model;
        
        std::cout << "Detected architecture: " << architecture << std::endl;
        
        if (architecture == "Qwen3ForCausalLM") {
            model = std::make_unique<Qwen3CausalLM>(config, generation_config, nntr_config);
        } else if (architecture == "Qwen3MoeForCausalLM") {
            model = std::make_unique<Qwen3MoECausalLM>(config, generation_config, nntr_config);
        } else {
            // Default to base CausalLM
            std::cout << "Using default CausalLM for architecture: " << architecture << std::endl;
            model = std::make_unique<CausalLM>(config, generation_config, nntr_config);
        }
        
        // Initialize model
        std::cout << "Initializing model..." << std::endl;
        model->initialize();
        
        // Load weights
        std::string weight_path = model_dir + "/" + nntr_config["model_file_name"].get<std::string>();
        std::cout << "Loading weights from: " << weight_path << std::endl;
        model->load_weight(weight_path);
        
        // Get sample input if available
        std::string sample_input = nntr_config.contains("sample_input") 
            ? nntr_config["sample_input"].get<std::string>()
            : "<|im_start|>user\nGive me a short introduction to large language model.<|im_end|>\n<|im_start|>assistant\n";
        
        std::cout << "\nModel initialized successfully!" << std::endl;
        std::cout << "Running inference with sample input..." << std::endl;
        std::cout << "Input: " << sample_input << std::endl;
        std::cout << "Output: ";
        
        // Run inference
        model->run(sample_input, false); // do_sample = false for deterministic output
        
        std::cout << std::endl;
        std::cout << "Inference completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}