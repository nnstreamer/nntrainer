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
 * @file	qwen3_causallm.cpp
 * @date	23 July 2025
 * @brief	This defines a qwen3 causal language model.
 * @see		https://github.com/nnstreamer/
 * @author	Eunju Yang <ej.yang@samsung.com>
 * @author	Pranjal Thapliyal <p.thapliyal@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */
#include <llm_util.hpp>
#include <model.h>
#include <qwen3_causallm.h>

#include <app_context.h>
#include <engine.h>
#include <reshaped_rms_norm.h>

namespace causallm {

Tensor Qwen3Transformer::createAttention(const int layer_id, int seq_len,
                                         int n_heads, int head_dim,
                                         Tensor query, Tensor key,
                                         Tensor value) {

  // Q layer
  // Q layer
  std::vector<std::string> wq_props = {
    withKey("name", "layer" + std::to_string(layer_id) + "_wq"),
    withKey("unit", head_dim * n_heads), withKey("disable_bias", "true"),
    withKey("weight_initializer", "ones"), withKey("engine", layerEngine())};
  if (hasLoRA("wq", layer_id))
    appendLoRAProps(wq_props);
  else if (LORA_RANK > 0)
    wq_props.push_back(withKey("trainable", "false"));
  LayerHandle wq(createLayer("fully_connected", wq_props));
  Tensor q = wq(query);

  // Q-reshaped-norm layer (q_norm(q_proj.view(hidden_shape))).
  // engine=GPU (mirror of Gemma4 S1.1): keeps q_norm output GPU_CLMEM-resident
  // (the layer is registered on the cl context in registerCustomLayers) instead
  // of draining q to the host for a CPU RMS norm every layer. Decode ~+20% on
  // Adreno, prefill neutral, token-identical. q/k norm carry gamma so they take
  // the GPU coop kernel directly (no gamma-free v_norm fallback concern).
  // Never a LoRA target; frozen when LoRA is active unless TRAIN_NORMS asks
  // for the "LoRA + norms" recipe.
  std::vector<std::string> q_norm_props = {
    withKey("name", "layer" + std::to_string(layer_id) + "_q_norm"),
    withKey("packed", "false"), withKey("epsilon", std::to_string(NORM_EPS)),
    withKey("feature_size", std::to_string(head_dim)),
    withKey("engine", layerEngine())};
  if (LORA_RANK > 0 && !TRAIN_NORMS)
    q_norm_props.push_back(withKey("trainable", "false"));
  LayerHandle q_norm(createLayer("reshaped_rms_norm", q_norm_props));
  Tensor q_normed = q_norm(q);

  // K layer
  std::vector<std::string> wk_props = {
    withKey("name", "layer" + std::to_string(layer_id) + "_wk"),
    withKey("unit", head_dim * n_heads / GQA_SIZE),
    withKey("disable_bias", "true"), withKey("weight_initializer", "ones"),
    withKey("engine", layerEngine())};
  if (hasLoRA("wk", layer_id))
    appendLoRAProps(wk_props);
  else if (LORA_RANK > 0)
    wk_props.push_back(withKey("trainable", "false"));
  LayerHandle wk(createLayer("fully_connected", wk_props));
  Tensor k = wk(key);

  // K-reshaped-norm layer (k_norm(k_proj.view(hidden_shape))). engine=GPU as
  // with q_norm above (GPU_CLMEM-resident, no per-layer host drain).
  std::vector<std::string> k_norm_props = {
    withKey("name", "layer" + std::to_string(layer_id) + "_k_norm"),
    withKey("packed", "false"), withKey("epsilon", std::to_string(NORM_EPS)),
    withKey("feature_size", std::to_string(head_dim)),
    withKey("engine", layerEngine())};
  if (LORA_RANK > 0 && !TRAIN_NORMS)
    k_norm_props.push_back(withKey("trainable", "false"));
  LayerHandle k_norm(createLayer("reshaped_rms_norm", k_norm_props));
  Tensor k_normed = k_norm(k);

  // V layer
  std::vector<std::string> wv_props = {
    withKey("name", "layer" + std::to_string(layer_id) + "_wv"),
    withKey("unit", head_dim * n_heads / GQA_SIZE),
    withKey("disable_bias", "true"), withKey("weight_initializer", "ones"),
    withKey("engine", layerEngine())};
  if (hasLoRA("wv", layer_id))
    appendLoRAProps(wv_props);
  else if (LORA_RANK > 0)
    wv_props.push_back(withKey("trainable", "false"));
  LayerHandle wv(createLayer("fully_connected", wv_props));
  Tensor v = wv(value);

  // KV cache wiring. Default is external (5-input mha) with FP16
  // placeholders owned by KVCacheManager. When NNTR_KV_INT8 is set (any
  // value), switch to 3-input mode so mha_core allocates an INT8 cache +
  // FP16 scale tensors internally - createKVCachePlaceholders only emits
  // FP16 tensors so it can't host the int8 path.
  static const bool _kv_int8_setup = std::getenv("NNTR_KV_INT8") != nullptr;

  // Attention core layer (frozen: GQA has no learnable weights beyond
  // Q/K/V/O, so there's nothing here for LoRA to target)
  LayerHandle mha(createLayer(
    "mha_core",
    {
      withKey("name", "layer" + std::to_string(layer_id) + "_attention"),
      withKey("num_heads", n_heads),
      withKey("num_heads_kv", n_heads / GQA_SIZE),
      withKey("max_timestep", std::to_string(MAX_SEQ_LEN)),
      withKey("init_seq_len", std::to_string(INIT_SEQ_LEN)),
      withKey("sliding_window", getLayerSlidingWindow(layer_id)),
      withKey("rope_theta", ROPE_THETA),
      withKey("max_position_embeddings", MAX_POSITION_EMBEDDINGS),
      withKey("max_new_tokens", std::to_string(NUM_TO_GENERATE)),
      withKey("is_causal", IS_CAUSAL ? "true" : "false"),
      // nntr_config "use_flash_attention" (default true) selects the flash
      // (GEMM) arm: the host one on the cpu engine, and on a GPU engine the
      // device attention (cuda_attention, or the OpenCL flash / image arm
      // under NNTR_MHA_GPU) for prefill and decode. false keeps the per-row
      // host path, which cannot read a device-only activation pool. q_norm /
      // k_norm above are separate layers, so every arm reads the same
      // normalised Q/K, and RoPE is applied inside mha_core before any of
      // them. See causallm_gemm_attention().
      withKey("use_gemm_attention",
              causallm_gemm_attention(USE_FLASH_ATTENTION) ? "true" : "false"),
      // Decode properties from getModelFeatures() (single source): qwen3
      // turns them on, as Gemma-4 does, so the decode RoPE runs on the device
      // (the fused rotate-and-commit on the Adreno image arm) instead of
      // reading Q and K back to the host in every layer. They do not decide
      // where decode attention runs once use_gemm_attention is on: the cuda
      // engine routes every step to its device arm, and with NNTR_MHA_GPU the
      // minimum flash step is 1 (x86, and ARM), so decode takes the OpenCL
      // arm too.
      withKey("gpu_decode_attn",
              getModelFeatures().decode_gpu ? "true" : "false"),
      withKey("gpu_decode_rope",
              getModelFeatures().decode_rope_gpu ? "true" : "false"),
      withKey("gpu_ohwi_rope",
              getModelFeatures().decode_gpu ? "true" : "false"),
    }));

  Tensor a;
  if (FOR_TRAINING) {
    // No KV-cache placeholders for training: mha_core's internal-cache
    // (3-input) construction is what its training-mode forward/backward
    // (trainForwarding()/calcDerivative() in mha_core.cpp) requires.
    a = mha({q_normed, k_normed, v});
  } else {
    a = wireAttentionKVCache(layer_id, n_heads, mha, q_normed, k_normed, v,
                             _kv_int8_setup);
  }

  // O layer
  std::vector<std::string> wo_props = {
    withKey("name", "layer" + std::to_string(layer_id) + "_attention_out"),
    withKey("unit", DIM), withKey("disable_bias", "true"),
    withKey("weight_initializer", "ones"), withKey("engine", layerEngine())};
  if (hasLoRA("wo", layer_id))
    appendLoRAProps(wo_props);
  else if (LORA_RANK > 0)
    wo_props.push_back(withKey("trainable", "false"));
  LayerHandle wo(createLayer("fully_connected", wo_props));
  return wo(a);
}

void Qwen3Transformer::registerCustomLayers() {
  ///
  auto &ct_engine = nntrainer::Engine::Global();

  try {
    ct_engine.registerLayerFactory(
      "cpu", nntrainer::createLayer<causallm::ReshapedRMSNormLayer>);
  } catch (std::invalid_argument &e) {
    std::cerr << "failed to register factory, reason: " << e.what()
              << std::endl;
  }
  // GPU-context registration of ReshapedRMSNormLayer is centralized in
  // CausalLM::registerCustomLayers (shared by all models); q/k norm above build
  // with engine=GPU and resolve there to stay GPU_CLMEM-resident.
  //
  // ...but that chokepoint is only reachable from the CausalLM hierarchy.
  // Qwen3Embedding is `SentenceTransformer + Qwen3Transformer` and never
  // inherits CausalLM, so under engine=gpu its q_norm/k_norm asked for a
  // "reshaped_rms_norm" that was registered on the cpu context only — and an
  // unregistered type on a live context THROWS ("Key is not found for the
  // object. Key: reshaped_rms_norm") instead of falling back to cpu.
  // Registering here as well covers every model that mixes in this class; the
  // duplicate registration from CausalLM's central call is caught below, so
  // whichever runs first wins and the other is a benign no-op.
  try {
    ct_engine.registerLayerFactory(
      "gpu", nntrainer::createLayer<causallm::ReshapedRMSNormLayer>);
  } catch (std::invalid_argument &e) {
    // no "gpu" context (CPU-only build) or already registered — both benign.
  }
}

void Qwen3CausalLM::registerCustomLayers() {
  CausalLM::registerCustomLayers();
  Qwen3Transformer::registerCustomLayers();
}

} // namespace causallm
