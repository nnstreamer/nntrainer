#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "json.hpp"
#include <app_context.h>
#include <factory.h>

#include "causal_lm.h"
#include "qwen3_causallm.h"

using json = nlohmann::json;

int main(int argc, char *argv[]) {

  /** Register all runnable causallm models to factory */
  causallm::Factory::Instance().registerModel(
    "LlamaForCausalLM", [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<causallm::CausalLM>(cfg, generation_cfg,
                                                  nntr_cfg);
    });
  causallm::Factory::Instance().registerModel(
    "Qwen3ForCausalLM", [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<causallm::Qwen3CausalLM>(cfg, generation_cfg,
                                                       nntr_cfg);
    });

  // Validate arguments
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <model_path> [input_prompt]\n"
              << "  <model_path>   : Path to model directory\n"
              << "  [input_prompt] : Optional input text (uses sample_input if "
                 "omitted)\n";
    return EXIT_FAILURE;
  }

  const std::string model_path = argv[1];
  std::string input_text;

  std::cout << model_path << std::endl;

  try {
    // Load configuration files
    json cfg = causallm::LoadJsonFile(model_path + "/config.json");
    json generation_cfg =
      causallm::LoadJsonFile(model_path + "/generation_config.json");
    json nntr_cfg = causallm::LoadJsonFile(model_path + "/nntr_config.json");

    // Determine input text
    if (argc >= 3) {
      input_text = argv[2];
    } else {
      input_text = nntr_cfg["sample_input"].get<std::string>();
    }

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

#if defined(_WIN32)
    model->run(input_text.c_str(), generation_cfg["do_sample"]);
#else
    model->run(input_text, generation_cfg["do_sample"]);
#endif

  } catch (const std::exception &e) {
    std::cerr << "\n[!] FATAL ERROR: " << e.what() << "\n";
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
