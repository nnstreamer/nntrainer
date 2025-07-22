#include <llm_util.hpp>
#include <model.h>
#include <qwen3_moe_causallm.h>

#include <app_context.h>
#include <engine.h>
#include <qwen_moe_layer.h>

namespace causallm {

void Qwen3MoECausalLM::setupParameters(json &cfg, json &generation_cfg,
                                       json &nntr_cfg) {
  Qwen3CausalLM(cfg, generation_cfg, nntr_cfg);

  // parameters for Qwen3MoE model
  try {
    NUM_EXPERTS = cfg["num_experts"];
    NUM_EXPERTS_PER_TOK = cfg["num_experts_per_tok"];
    INTERMEDIATE_SIZE = cfg["moe_intermediate_size"];
  } catch (const std::exception &e) {
    throw std::runtime_error("Qwen3MoE: num_experts and num_experts_per_tok "
                             "are not specified in the config file");
  }
}

std::vector<LayerHandle> Qwen3MoECausalLM::createMlp(const int layer_id,
                                                     int dim, int hidden_dim,
                                                     std::string input_name) {

  std::vector<LayerHandle> layers;
  layers.push_back(createLayer(
    "qwen_moe",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_down"),
     withKey("input_layers", input_name), withKey("unit", hidden_dim),
     withKey("num_experts", NUM_EXPERTS),
     withKey("num_experts_per_token", NUM_EXPERTS_PER_TOK),
     withKey("moe_activation", "swish")}));

  return layers;
}

void Qwen3MoECausalLM::registerCustomLayers() {

  Qwen3CausalLM::registerCustomLayers();
  auto &ct_engine = nntrainer::Engine::Global();
  auto app_context =
    static_cast<nntrainer::AppContext *>(ct_engine.getRegisteredContext("cpu"));

  try {
    app_context->registerFactory(nntrainer::createLayer<causallm::MoELayer>);
  } catch (std::invalid_argument &e) {
    std::cerr << "failed to register factory, reason: " << e.what()
              << std::endl;
  }
}

} // namespace causallm
