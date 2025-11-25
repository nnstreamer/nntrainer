#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "json.hpp"
#include <app_context.h>
#include <factory.h>

#include "causal_lm.h"
#include "ernie_causallm.h"

#include <sys/resource.h>
#include <thread>

using json = nlohmann::json;


int main(int argc, char *argv[]) {

  causallm::Factory::Instance().registerModel(
    "Ernie4_5_MoeForCausalLM",
    [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<causallm::Ernie4_5_MoeForCausalLM>(
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
