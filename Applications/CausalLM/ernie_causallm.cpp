//
// Created by donghak on 25. 11. 13..
//

#include <ernie_causallm.h>
#include <llm_util.hpp>
#include <app_context.h>
#include <engine.h>
#include <ernie_moe_layer.h>

namespace causallm {

std::vector<LayerHandle> Ernie4_5_MoeForCausalLM::createMlp(const int layer_id,
                                                      int dim, int hidden_dim,
                                                      std::string input_name) {
  std::vector<LayerHandle> layers;
  if (layer_id == 0 ) {
      layers.push_back(createLayer(
        "fully_connected",
        {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_up"),
         withKey("unit", hidden_dim), withKey("disable_bias", "true"),
         withKey("input_layers", input_name),
         withKey("weight_initializer", "ones")}));

      layers.push_back(createLayer(
        "fully_connected",
        {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_gate"),
         withKey("unit", hidden_dim), withKey("disable_bias", "true"),
         withKey("input_layers", input_name),
         withKey("weight_initializer", "ones")}));

      layers.push_back(createLayer(
        "swiglu",
        {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_swiglu"),
         withKey("input_layers", "layer" + std::to_string(layer_id) + "_ffn_up," +
                                   "layer" + std::to_string(layer_id) +
                                   "_ffn_gate")}));

      layers.push_back(createLayer(
        "fully_connected",
        {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_down"),
         withKey("unit", dim), withKey("disable_bias", "true"),
         withKey("input_layers",
                 "layer" + std::to_string(layer_id) + "_ffn_swiglu"),
         withKey("weight_initializer", "ones")}));

  } else {
    layers.push_back(createLayer(
      "ernie_moe",
      {
        withKey("name", "layer" + std::to_string(layer_id) + "_ffn_down"),
        withKey("input_layers", input_name),
        withKey("unit", MOE_INTERMEDIATE_SIZE),
        withKey("num_experts", NUM_EXPERTS),
        withKey("num_experts_per_token", 8), withKey("moe_activation", "swish")
      }));
  }
  return layers;
}

void Ernie4_5_MoeForCausalLM::setupParameters(json &cfg, json &generation_cfg,
                                        json &nntr_cfg) {


  try {
    NUM_EXPERTS = cfg["moe_num_experts"].get<unsigned int>();
    NUM_EXPERTS_PER_TOK = 6; //shared + expert = 6
    MOE_INTERMEDIATE_SIZE = cfg["moe_intermediate_size"].get<unsigned int>();

  } catch (const std::exception &e) {
    throw std::runtime_error("Ernie Causallm: config parsing error");
  }
}

void Ernie4_5_MoeForCausalLM::registerCustomLayers() {
  CausalLM::registerCustomLayers();
  auto &ct_engine = nntrainer::Engine::Global();
  auto app_context =
    static_cast<nntrainer::AppContext *>(ct_engine.getRegisteredContext("cpu"));

  try {
    app_context->registerFactory(
      nntrainer::createLayer<causallm::ErnieMoELayer>);
  } catch (std::invalid_argument &e) {
    std::cerr << "failed to register factory, reason: " << e.what()
              << std::endl;
  }
}

} // namespace causallm