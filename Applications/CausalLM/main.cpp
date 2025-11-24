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
 *
 * @file	main.cpp
 * @date	23 July 2025
 * @brief	This is a main file for CausalLM application
 * @see		https://github.com/nnstreamer/
 * @author	Eunju Yang <ej.yang@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "json.hpp"
#include <app_context.h>
#include <factory.h>

#include "causal_lm.h"
#include "gptoss_cached_slim_causallm.h"
#include "gptoss_causallm.h"
#include "nntr_qwen3_causallm.h"
#include "nntr_qwen3_moe_causallm.h"
#include "qwen3_cached_slim_moe_causallm.h"
#include "qwen3_causallm.h"
#include "qwen3_moe_causallm.h"
#include "qwen3_slim_moe_causallm.h"

#include "ernie_causallm.h"

#include <sys/resource.h>

#include <atomic>
#include <chrono>
#include <thread>

using json = nlohmann::json;


int main(int argc, char *argv[]) {

  causallm::Factory::Instance().registerModel(
    "Ernie4_5_MoeForCausalLM",
    [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<causallm::Ernie4_5_MoeForCausalLM>(
        cfg, generation_cfg, nntr_cfg);
    });

  causallm::Factory::Instance().registerModel(
    "Qwen3CachedSlimMoeForCausalLM",
    [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<causallm::Qwen3CachedSlimMoECausalLM>(
        cfg, generation_cfg, nntr_cfg);
    });

  const std::string model_path = argv[1];
  std::string input_text;
  std::string system_head_prompt = "";
  std::string system_tail_prompt = "";

  std::cout << model_path << std::endl;

  try {
    // Load configuration files
    json cfg = causallm::LoadJsonFile(model_path + "/config.json");
    json generation_cfg =
      causallm::LoadJsonFile(model_path + "/generation_config.json");
    json nntr_cfg = causallm::LoadJsonFile(model_path + "/nntr_config.json");


    input_text = nntr_cfg["sample_input"].get<std::string>();


    // Construct weight file path
    const std::string weight_file =
      model_path + "/" + nntr_cfg["model_file_name"].get<std::string>();

    std::cout << weight_file << std::endl;

    // Initialize and run model
    auto model = causallm::Factory::Instance().create(
      cfg["architectures"].get<std::vector<std::string>>()[0], cfg,
      generation_cfg, nntr_cfg);


    model->initialize();

    model->load_weight(weight_file);

    bool do_sample = generation_cfg.contains("do_sample")
                       ? generation_cfg["do_sample"].get<bool>()
                       : false;
    model->run(input_text, do_sample, system_head_prompt, system_tail_prompt);

  } catch (const std::exception &e) {
    std::cerr << "\n[!] FATAL ERROR: " << e.what() << "\n";
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
