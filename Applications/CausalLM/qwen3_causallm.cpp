#include <llm_util.hpp>
#include <model.h>
#include <qwen3_causallm.h>

#include <app_context.h>
#include <engine.h>
#include <reshaped_rms_norm.h>

namespace causallm {

std::vector<LayerHandle>
Qwen3CausalLM::createAttention(const int layer_id, int seq_len, int n_heads,
                               int head_dim, std::string query_name,
                               std::string key_name, std::string value_name) {

  std::vector<LayerHandle> layers;
  auto Q = "layer" + std::to_string(layer_id) + "_wq";
  auto Q_norm = "layer" + std::to_string(layer_id) + "_q_norm";
  auto K = "layer" + std::to_string(layer_id) + "_wk";
  auto K_norm = "layer" + std::to_string(layer_id) + "_k_norm";
  auto V = "layer" + std::to_string(layer_id) + "_wv";
  auto A = "layer" + std::to_string(layer_id) + "_attention";
  auto O = "layer" + std::to_string(layer_id) + "_attention_out";

  // V layer
  std::vector<std::string> v_params = {
    withKey("name", V), withKey("unit", head_dim * n_heads / GQA_SIZE),
    withKey("disable_bias", "true"), withKey("input_layers", value_name),
    withKey("weight_initializer", "ones")};
  layers.push_back(createLayer("fully_connected", v_params));

  // K layer
  std::vector<std::string> k_params = {
    withKey("name", K), withKey("unit", head_dim * n_heads / GQA_SIZE),
    withKey("disable_bias", "true"), withKey("input_layers", key_name),
    withKey("weight_initializer", "ones")};
  layers.push_back(createLayer("fully_connected", k_params));

  // K-reshaped-norm layer
  // k_norm(k_proj.view(hidden_shape))
  std::vector<std::string> k_norm_params = {
    withKey("name", K_norm), withKey("input_layers", K),
    withKey("packed", "false"), withKey("epsilon", std::to_string(NORM_EPS)),
    withKey("feature_size", std::to_string(head_dim))};
  layers.push_back(createLayer("reshaped_rms_norm", k_norm_params));

  // Q layer
  std::vector<std::string> q_params = {
    withKey("name", Q), withKey("unit", head_dim * n_heads),
    withKey("disable_bias", "true"), withKey("input_layers", query_name),
    withKey("weight_initializer", "ones")};
  layers.push_back(createLayer("fully_connected", q_params));

  // Q-reshaped-norm layer
  // q_norm(q_proj.view(hidden_shape))
  std::vector<std::string> q_norm_params = {
    withKey("name", Q_norm), withKey("input_layers", Q),
    withKey("packed", "false"), withKey("epsilon", std::to_string(NORM_EPS)),
    withKey("feature_size", std::to_string(head_dim))};
  layers.push_back(createLayer("reshaped_rms_norm", q_norm_params));

  // Attention core layer
  std::vector<std::string> a_params = {
    withKey("name", A),
    withKey("num_heads", n_heads),
    withKey("num_heads_kv", n_heads / GQA_SIZE),
    withKey("max_timestep", std::to_string(INIT_SEQ_LEN + NUM_TO_GENERATE)),
    withKey("sliding_window", SLIDING_WINDOW),
    withKey("rope_theta", ROPE_THETA),
    withKey("max_position_embeddings", MAX_POSITION_EMBEDDINGS),
    withKey("max_new_tokens", std::to_string(NUM_TO_GENERATE)),
    withKey("input_layers", {Q_norm, K_norm, V})};
  layers.push_back(createLayer("mha_core", a_params));

  // O layer
  std::vector<std::string> o_params = {
    withKey("name", O), withKey("unit", DIM), withKey("disable_bias", "true"),
    withKey("input_layers", A), withKey("weight_initializer", "ones")};
  layers.push_back(createLayer("fully_connected", o_params));

  return layers;
}

void Qwen3CausalLM::registerCustomLayers() {
  CausalLM::registerCustomLayers();
  ///
  auto &ct_engine = nntrainer::Engine::Global();
  auto app_context =
    static_cast<nntrainer::AppContext *>(ct_engine.getRegisteredContext("cpu"));

  try {
    app_context->registerFactory(
      nntrainer::createLayer<causallm::ReshapedRMSNormLayer>);
  } catch (std::invalid_argument &e) {
    std::cerr << "failed to register factory, reason: " << e.what()
              << std::endl;
  }
}

} // namespace causallm
