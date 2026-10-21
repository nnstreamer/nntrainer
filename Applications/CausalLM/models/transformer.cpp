// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Eunju Yang <ej.yang@samsung.com>
 *
 * @file   transformer.cpp
 * @date   10 July 2025
 * @see    https://github.com/nntrainer/nntrainer
 * @author Eunju Yang <ej.yang@samsung.com>
 * @author Pranjal Thapliyal <p.thapliyal@samsung.com>
 * @author Sumon Nath <sumon.nath@samsung.com>
 * @author Niket Agarwal <niket.a@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This file defines Transformer's basic actions
 */

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <mutex>
#include <unordered_map>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h> // SetThreadPriority (async tokenizer, below-normal)
#endif

#include <app_context.h>
#if defined(ENABLE_OPENCL)
#include <cl_context.h> // GPU registration goes via the Engine facade now; the
// header is OpenCL-only -> guard for the no-OpenCL build.
#endif
#include <engine.h>
#include <model.h>

#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
#include <cuda_context.h>
#include <cuda_rmsnorm_layer.h>
#include <per_layer_slice.h>
#endif

#include <llm_util.hpp>
#include <tokenizer_cache.h>
#include <tokenizers_cpp.h>
#include <transformer.h>

#include <cpu_backend.h>
#include <decision_head_loss_layer.h>
#include <embedding_layer.h>
#include <fc_layer.h>
#include <fp16.h>
#include <int4_utils.h>
#include <lm_head_loss_layer.h>
#include <mha_core.h>
#include <neuralnet.h>
#include <per_layer_slice_gpu.h>
#include <qs4cx_tensor.h>
#include <residency_policy.h>
#include <rms_norm.h>
#include <rms_norm_gpu.h>
#include <util_func.h>

namespace causallm {

/**
 * @brief Get model format from weight file extension.
 */
ml::train::ModelFormat
Transformer::formatFromExtension(const std::string &weight_path) {
  const auto dot = weight_path.find_last_of('.');
  if (dot != std::string::npos) {
    const std::string ext = weight_path.substr(dot + 1);
    if (ext == "safetensors")
      return ml::train::ModelFormat::MODEL_FORMAT_SAFETENSORS;
  }
  return ml::train::ModelFormat::MODEL_FORMAT_BIN;
}

/**
 * @brief Load a file as a binary string.
 */
std::string LoadBytesFromFile(const std::string &path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file: " + path);
  }
  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);

  std::string buffer(size, ' ');
  if (!file.read(&buffer[0], size)) {
    throw std::runtime_error("Failed to read file: " + path);
  }
  return buffer;
}

/**
 * @brief Convert model_type text from config to ModelType.
 */
ModelType strToModelType(std::string model_type) {

  std::string model_type_lower = model_type;
  std::transform(model_type_lower.begin(), model_type_lower.end(),
                 model_type_lower.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  static const std::unordered_map<std::string, ModelType> model_type_map = {
    {"model", ModelType::MODEL},
    {"causallm", ModelType::CAUSALLM},
    {"embedding", ModelType::EMBEDDING}};

  if (model_type_map.find(model_type_lower) == model_type_map.end()) {
    return ModelType::UNKNOWN;
  }

  return model_type_map.at(model_type_lower);
}

/**
 * @brief Construct a Transformer and initialize shared config state.
 */
Transformer::Transformer(json &cfg, json &generation_cfg, json &nntr_cfg,
                         ModelType model_type) {

  std::string config_model_type_str = "Model";
  if (nntr_cfg.contains("model_type")) {
    config_model_type_str = nntr_cfg["model_type"].get<std::string>();
  }

  ModelType config_model_type = strToModelType(config_model_type_str);

  if (model_type != config_model_type) {
    throw std::runtime_error("model_type mismatch. Class Type: " +
                             std::to_string(static_cast<int>(model_type)) +
                             ", Config Type: " + config_model_type_str);
  }

  const bool skip_tokenizer = nntr_cfg.contains("skip_tokenizer") &&
                              nntr_cfg["skip_tokenizer"].get<bool>();

  /** [determinism] The pack's own opt-out from the reproducible arm. The fast
   * arm is the default now, so the key is redundant and only kept so a pack
   * written against the previous contract keeps loading; there is still no
   * key to turn the reproducible arm ON from a pack. Read here, in the base
   * constructor, so it lands before any layer finalizes (which is where the
   * arm is decided) and so every model class gets it from one place rather
   * than each one re-reading a key. An explicit NNTR_DETERMINISTIC from the
   * caller still outranks it -- see causallm::detArm(). Absent or false leaves
   * the default alone; the flag is never cleared here, so one pack opting out
   * cannot be undone by the next pack in the same process, which is the safe
   * direction for a process-wide
   * answer. detArm() does not cache, so the flag takes effect wherever it is
   * read after this point -- and every arm decision (layer finalize) is after
   * it. */
  if (nntr_cfg.contains("allow_nondeterministic") &&
      nntr_cfg["allow_nondeterministic"].is_boolean() &&
      nntr_cfg["allow_nondeterministic"].get<bool>())
    causallm::packAllowsNondeterministic() = true;

  // Initialize the model with the provided configurations. Vision models such
  // as TimmViT defer this to their derived constructor because the base
  // Transformer setup expects text-model fields.
  if (!(skip_tokenizer && model_type == ModelType::MODEL)) {
    setupParameters(cfg, generation_cfg, nntr_cfg);
  }

  // Skip tokenizer if specified, or when no tokenizer_file is configured
  // (e.g. vision-encoder sub-models composed into a multimodal handle, whose
  // config carries no tokenizer). Avoids a json type_error on a null path.
  if (skip_tokenizer || !nntr_cfg.contains("tokenizer_file") ||
      nntr_cfg["tokenizer_file"].is_null()) {
    tokenizer = nullptr; // No tokenizer for this model
  } else {
    // The ~30MB tokenizer.json parse measured ~680ms
    // and is independent of graph compile + weight load -- run it on a side
    // thread and join at first use (ensureTokenizer / getTokenizer).
    const std::string tok_path = nntr_cfg["tokenizer_file"];
    tokenizer_future_ = std::async(std::launch::async, [tok_path]() {
    // Below-normal priority: the parse tail (~100-300ms) overlaps the
    // parallel v8c prebuild workers (round-13 ordering trace), so let
    // the load workers win contention -- the parse still finishes long
    // before its first use joins it.
#if defined(_WIN32)
      SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
#endif
      // Through the persistent snapshot cache (tokenizer_cache.cpp): a hit
      // rebuilds the BPE from the snapshot's vocab/merges tables instead of
      // re-parsing the JSON; a miss, a stale key or a corrupt file parses
      // exactly as before (same FromBlobJSON call, same bytes) and rewrites
      // the snapshot in the background. The side thread only hides the parse
      // while the rest of the load is longer than it, which a small model's
      // is not. NNTR_TOKENIZER_CACHE=0 opts out.
      auto tok = causallm::LoadTokenizerCached(tok_path);
      // [NNTR_INIT_TRACE] ordering marker: where in the load pipeline the
      // async parse actually completes (contention forensics -- if this
      // lands after "load_weight begins" the side thread competes with the
      // parallel v8c prebuild workers; observed: it completes during the
      // single-threaded compile/planner phase instead).
      if (std::getenv("NNTR_INIT_TRACE")) {
        std::fprintf(stderr, "[init-trace] tokenizer async parse done\n");
        std::fflush(stderr);
      }
      return tok;
    });
  }
};

/**
 * @brief Set common transformer parameters from model configs.
 */
void Transformer::setupParameters(json &cfg, json &generation_cfg,
                                  json &nntr_cfg) {

  /** Initialize nntr prameters */
  BATCH_SIZE = nntr_cfg["batch_size"].get<unsigned int>();
  MODEL_TENSOR_TYPE = nntr_cfg["model_tensor_type"].get<std::string>();
  INIT_SEQ_LEN = nntr_cfg["init_seq_len"];
  /** NNTR_INIT_SEQ_LEN overrides the pack's init_seq_len.
   *
   *  init_seq_len is not a capability: max_seq_len is. It is the HEIGHT of the
   *  activation plane the graph is built at, and therefore also the largest
   *  prefill chunk that can be fed in one forward (Transformer::prefillChunk
   *  clamps to it). Every activation the planner sizes is [init_seq_len x
   *  something], so on the GPU lane the plane -- shared and device both -- is
   *  linear in this number, and it is the one dial that moves the activation
   *  half of the GPU footprint at all.
   *
   *  Lowering it does NOT shorten the prompt a model can take: with chunked
   *  prefill on (the KV ring, which is opt-in via NNTR_KV_WINDOW_RING or a
   *  model's kvRingByDefault(), turns it on for OpenCL and CUDA) the
   *  prompt is bounded by the KV budget and fed in chunks of this height. It
   *  trades prefill throughput -- more, smaller launches -- for plane bytes,
   *  which is a measurement, not an assumption, so it is a dial rather than a
   *  new default here.
   *
   *  Refused below 8 (a chunk of 0 would disable the prefill loop entirely)
   *  or above max_seq_len. */
  /** The upper bound is max_seq_len, not the pack's own init_seq_len. The
   *  pack value is a DEFAULT (a plane height the pack ships with), while the
   *  capability is max_seq_len -- Transformer::prefillChunk() clamps the
   *  chunk to INIT_SEQ_LEN and the KV budget bounds the prompt, so any height
   *  in [8, max_seq_len] builds a graph the pack can run. Bounding the
   *  override by the pack value instead made a lowered default a one-way
   *  door: once a pack ships init_seq_len 512, no A/B could raise it back to
   *  1024 without editing the pack. */
  /** NNTR_MAX_SEQ_LEN overrides the pack's max_seq_len -- the capability
   *  itself (KV budget, prompt bound). This is the other half of the pair a
   *  host embedding this engine configures: the init_seq_len override below
   *  was read, this name was not, so a host asking for a 32K window through
   *  the environment ran in the pack's 2048 window and had its prompt
   *  silently clipped (1919 of 32439 tokens prefilled on the cell that found
   *  it). Written back into nntr_cfg so the readers below see one value.
   *  Bounded by the RoPE table (max_position_embeddings) when the model
   *  config carries one; init_seq_len follows down when it would exceed the
   *  new cap. */
  if (const char *msl = std::getenv("NNTR_MAX_SEQ_LEN")) {
    const long want = std::atol(msl);
    const unsigned int pack = nntr_cfg["max_seq_len"].get<unsigned int>();
    unsigned int cap = 0;
    if (cfg.contains("max_position_embeddings"))
      cap = cfg["max_position_embeddings"].get<unsigned int>();
    if (want >= 8) {
      unsigned int use = static_cast<unsigned int>(want);
      if (cap != 0 && use > cap) {
        std::fprintf(stderr,
                     "[max_seq_len] NNTR_MAX_SEQ_LEN=%ld exceeds "
                     "max_position_embeddings=%u; clamping\n",
                     want, cap);
        use = cap;
      }
      nntr_cfg["max_seq_len"] = use;
      if (nntr_cfg["init_seq_len"].get<unsigned int>() > use) {
        nntr_cfg["init_seq_len"] = use;
        INIT_SEQ_LEN = use;
      }
      std::fprintf(stderr, "[max_seq_len] pack %u -> %u (NNTR_MAX_SEQ_LEN)\n",
                   pack, use);
      std::fflush(stderr);
    } else {
      std::fprintf(stderr, "[max_seq_len] ignoring NNTR_MAX_SEQ_LEN=%s\n", msl);
      std::fflush(stderr);
    }
  }
  /** The pack's own max_seq_len gets the SAME bound as the env override above.
   *  The RoPE table is built for max_position_embeddings positions, so a pack
   *  asking for a longer window walks the host rotation past the end of that
   *  table (a segfault in compute_rotary_emb_value on the first long prompt,
   *  not a load error). Clamp, say so once, and let init_seq_len follow. */
  if (cfg.contains("max_position_embeddings") &&
      nntr_cfg.contains("max_seq_len")) {
    const unsigned int cap = cfg["max_position_embeddings"].get<unsigned int>();
    const unsigned int pack = nntr_cfg["max_seq_len"].get<unsigned int>();
    if (cap != 0 && pack > cap) {
      std::fprintf(stderr,
                   "[max_seq_len] nntr_config max_seq_len=%u exceeds "
                   "max_position_embeddings=%u; clamping to %u\n",
                   pack, cap, cap);
      std::fflush(stderr);
      nntr_cfg["max_seq_len"] = cap;
      if (nntr_cfg.contains("init_seq_len") &&
          nntr_cfg["init_seq_len"].get<unsigned int>() > cap) {
        nntr_cfg["init_seq_len"] = cap;
        INIT_SEQ_LEN = cap;
      }
    }
  }
  /** The pack's plane height, AFTER the max_seq_len / max_position_embeddings
   *  clamps above have had their say. This is the floor prefillPlaneFor() never
   *  goes below, so a pack that ships a tall plane keeps it and only a pack
   *  that ships a small one gets grown per request. */
  PACK_INIT_SEQ_LEN = INIT_SEQ_LEN;
  const unsigned int isl_cap = nntr_cfg["max_seq_len"].get<unsigned int>();
  if (const char *isl = std::getenv("NNTR_INIT_SEQ_LEN")) {
    const int want = std::atoi(isl);
    if (want >= 8 && want <= static_cast<int>(isl_cap)) {
      std::fprintf(stderr,
                   "[init_seq_len] pack %u -> %d (NNTR_INIT_SEQ_LEN); the "
                   "activation plane and the prefill chunk follow\n",
                   static_cast<unsigned int>(INIT_SEQ_LEN), want);
      std::fflush(stderr);
      INIT_SEQ_LEN = want;
      /** An explicit height is an A/B of exactly this number: do not then move
       *  it per request behind the experimenter's back. */
      plane_pinned_ = true;
    } else {
      std::fprintf(stderr,
                   "[init_seq_len] ignoring NNTR_INIT_SEQ_LEN=%s: outside "
                   "[8, %u]\n",
                   isl, isl_cap);
      std::fflush(stderr);
    }
  }
  MAX_SEQ_LEN = nntr_cfg["max_seq_len"];
  // num_to_generate is optional: absent (or <= 0) means "no explicit cap",
  // i.e. generate until EOS or until the context window runs out. 0 is the
  // sentinel every consumer below tests for.
  NUM_TO_GENERATE = nntr_cfg.value("num_to_generate", 0);
  // Any non-positive value is the supported "no explicit cap" request, not a
  // misconfiguration: normalize it to the same 0 sentinel silently. This lives
  // in the function that (re-)reads the key so it holds after every call --
  // a derived Transformer whose constructor calls setupParameters() again
  // reloads the raw value, which would silently drop a guard placed elsewhere.
  if (NUM_TO_GENERATE < 0)
    NUM_TO_GENERATE = 0;
  MODEL_TENSOR_TYPE = nntr_cfg["model_tensor_type"];
  MEMORY_SWAP = nntr_cfg.contains("fsu") ? nntr_cfg["fsu"].get<bool>() : false;
  FSU_LOOKAHEAD = nntr_cfg.contains("fsu_lookahead")
                    ? nntr_cfg["fsu_lookahead"].get<unsigned int>()
                    : 1;
  EMBEDDING_DTYPE = nntr_cfg["embedding_dtype"];
  FC_LAYER_DTYPE = nntr_cfg["fc_layer_dtype"];
  // Legacy QINT4 weights now load as the canonical QS4CX class;
  // remap the config dtype so the whole pipeline (tensor factory + runtime
  // consumers) is QS4CX. The on-disk bytes stay legacy QINT4 and are transcoded
  // losslessly at read time, keyed on model_tensor_type ("QINT4-*"), which is
  // intentionally NOT remapped so the loader still knows the on-disk format.
  if (FC_LAYER_DTYPE == "QINT4")
    FC_LAYER_DTYPE = "QS4CX";
  USE_FLASH_ATTENTION = nntr_cfg.contains("use_flash_attention")
                          ? nntr_cfg["use_flash_attention"].get<bool>()
                          : true;
  EMBEDDING_FILE_NAME = nntr_cfg.value("embedding_file_name", std::string());
  PLE_FILE_NAME = nntr_cfg.value("ple_file_name", std::string());
  PLE_SIDECAR_EXPORT = nntr_cfg.value("ple_sidecar_export", std::string());
  EMBD_SIDECAR_EXPORT = nntr_cfg.value("embd_sidecar_export", std::string());
  LMHEAD_UNTIE =
    nntr_cfg.contains("lmhead_untie") && nntr_cfg["lmhead_untie"].get<bool>();
  LMHEAD_TIE_LUT = nntr_cfg.contains("lmhead_tie_lut") &&
                   nntr_cfg["lmhead_tie_lut"].get<bool>();

  if (cfg.contains("is_causal")) {
    IS_CAUSAL = cfg["is_causal"].get<bool>();
  } else if (cfg.contains("use_bidirectional_attention") &&
             !cfg["use_bidirectional_attention"].is_null()) {
    IS_CAUSAL = !cfg["use_bidirectional_attention"].get<bool>();
  } else if (nntr_cfg.contains("model_type") &&
             strToModelType(nntr_cfg["model_type"].get<std::string>()) ==
               ModelType::EMBEDDING &&
             cfg.contains("architectures") && cfg["architectures"].is_array() &&
             !cfg["architectures"].empty() &&
             cfg["architectures"][0].get<std::string>() == "Qwen2Model") {
    IS_CAUSAL = false;
  }

  NUM_VOCAB = cfg["vocab_size"];
  DIM = cfg["hidden_size"];
  // Not every architecture that reaches this base parser declares an FFN
  // intermediate size (e.g. the encoder-only / conv-block models); default it
  // instead of throwing on a missing key.
  INTERMEDIATE_SIZE =
    cfg.contains("intermediate_size") ? cfg["intermediate_size"].get<int>() : 0;
  NUM_LAYERS = cfg["num_hidden_layers"];
  NUM_HEADS = cfg["num_attention_heads"];
  HEAD_DIM = cfg.contains("head_dim")
               ? cfg["head_dim"].get<int>()
               : DIM / NUM_HEADS; // default value is hidden_size / num_heads
  NUM_KEY_VALUE_HEADS = cfg.contains("num_key_value_heads")
                          ? cfg["num_key_value_heads"].get<int>()
                          : NUM_HEADS;
  SLIDING_WINDOW =
    cfg.contains("sliding_window") && !cfg["sliding_window"].is_null()
      ? cfg["sliding_window"].get<unsigned int>()
      : UINT_MAX;
  SLIDING_WINDOW_PATTERN = cfg.contains("sliding_window_pattern")
                             ? cfg["sliding_window_pattern"].get<unsigned int>()
                             : 1;
  MAX_POSITION_EMBEDDINGS = cfg["max_position_embeddings"].get<unsigned int>();
  if (cfg.contains("rope_theta")) {
    ROPE_THETA = cfg["rope_theta"].get<unsigned int>();
  } else if (cfg.contains("rope_parameters") &&
             cfg["rope_parameters"].contains("rope_theta")) {
    ROPE_THETA = cfg["rope_parameters"]["rope_theta"].get<unsigned int>();
  } else if (cfg.contains("rope_parameters") &&
             cfg["rope_parameters"].contains("sliding_attention")) {
    json &rope_cfg = cfg["rope_parameters"]["sliding_attention"];
    ROPE_THETA = rope_cfg.value("rope_theta", 10000);
  } else {
    ROPE_THETA = cfg.value("rope_theta", 10000);
  }
  TIE_WORD_EMBEDDINGS = cfg.contains("tie_word_embeddings")
                          ? cfg["tie_word_embeddings"].get<bool>()
                          : false;
  NORM_EPS =
    cfg.contains("rms_norm_eps") ? cfg["rms_norm_eps"].get<float>() : 1e-5;
  GQA_SIZE = NUM_HEADS / NUM_KEY_VALUE_HEADS;

  // Attention-logit soft-capping (Gemma family). Parsed here so every model
  // shares one path; gated on cfg presence, so models without the key keep the
  // 0.0f default (no soft-cap). Consolidated from the per-model
  // setupParameters overrides, which carried byte-identical copies of it.
  if (cfg.contains("attn_logit_softcapping") &&
      !cfg["attn_logit_softcapping"].is_null()) {
    ATTN_LOGIT_SOFTCAPPING = cfg["attn_logit_softcapping"].get<float>();
  }

  // [Adreno image-attn model vetting] none is needed: gemma4's
  // global_head_dim=512 runs correctly on the OHWI image kernels (see the note
  // in Gemma4Transformer::setupParameters). The sliding-window case is handled
  // IN the kernels since qk_matmul_f16_ohwi (+_img) grew a local_window
  // argument (n + W <= q_pos lower-bound mask), so window < max_seq_len no
  // longer forces the flash path.

  LORA_RANK = nntr_cfg.contains("lora_rank")
                ? nntr_cfg["lora_rank"].get<unsigned int>()
                : 0;
  LORA_ALPHA = nntr_cfg.contains("lora_alpha")
                 ? nntr_cfg["lora_alpha"].get<unsigned int>()
                 : 0;
  LORA_TARGET = nntr_cfg.contains("lora_target")
                  ? nntr_cfg["lora_target"].get<std::vector<std::string>>()
                  : std::vector<std::string>();
  {
    const int last_n = nntr_cfg.value("lora_last_n_layers", 0);
    LORA_LAYER_START =
      (last_n > 0 && last_n < NUM_LAYERS) ? NUM_LAYERS - last_n : 0;
  }
  LORA_CLIP_GRAD = nntr_cfg.contains("lora_clip_grad_by_norm")
                     ? nntr_cfg["lora_clip_grad_by_norm"].get<float>()
                     : 0.0f;
  TRAIN_NORMS = nntr_cfg.contains("lora_train_norms") &&
                nntr_cfg["lora_train_norms"].get<bool>();
  LORA_QAT = nntr_cfg.contains("lora_qat") && nntr_cfg["lora_qat"].get<bool>();
  QAT_EMA_MOMENTUM = nntr_cfg.value("lora_qat_ema_momentum", 0.1f);
  ACTIVATION_CKPT = nntr_cfg.value("activation_checkpointing", false);

  return;
};

bool Transformer::hasLoRA(const std::string &module_type, int layer_id) const {
  return LORA_RANK > 0 && layer_id >= LORA_LAYER_START &&
         std::find(LORA_TARGET.begin(), LORA_TARGET.end(), module_type) !=
           LORA_TARGET.end();
}

std::string Transformer::layerEngine() const {
  return (FOR_TRAINING || FORCE_CPU_ENGINE) ? std::string("cpu")
                                            : causallm_engine();
}

bool Transformer::trainGpuOffload() const {
#if defined(ENABLE_OPENCL)
  if (!FOR_TRAINING || LORA_RANK <= 0)
    return false;
  // Opt-in: the offload is correct but, until the device GEMM beats the CPU
  // table on the target, a default-on switch would slow training down.
  const char *e = std::getenv("NNTR_TRAIN_GPU");
  if (e == nullptr || e[0] == '0')
    return false;
  try {
    (void)nntrainer::Engine::Global().getRegisteredContext("gpu");
    return true;
  } catch (std::exception &) {
    return false; // NNTR_ENGINE=cpu: the gpu context was never brought up
  }
#else
  return false;
#endif
}

void Transformer::appendLoRAProps(std::vector<std::string> &props) const {
  props.emplace_back(withKey("lora_rank", std::to_string(LORA_RANK)));
  if (trainGpuOffload())
    props.emplace_back(withKey("lora_gpu", "true"));
  if (LORA_ALPHA > 0)
    props.emplace_back(withKey("lora_alpha", std::to_string(LORA_ALPHA)));
  if (LORA_CLIP_GRAD > 0.0f)
    props.emplace_back(
      withKey("clip_grad_by_norm", std::to_string(LORA_CLIP_GRAD)));
  if (LORA_QAT)
    props.emplace_back(withKey("lora_qat", "true"));
  if (LORA_QAT)
    props.emplace_back(
      withKey("lora_qat_ema_momentum", std::to_string(QAT_EMA_MOMENTUM)));
}

/**
 * @brief Join the async tokenizer load.
 */
void Transformer::ensureTokenizer() {
  std::lock_guard<std::mutex> lk(tokenizer_join_mtx_);
  if (tokenizer_future_.valid())
    tokenizer = tokenizer_future_.get();
}

/**
 * @brief Build and compile the symbolic transformer graph.
 */
void Transformer::initialize() {

  // [NNTR_INIT_TRACE] init-latency dissection (round-13 follow-up).
  static const bool init_trace = std::getenv("NNTR_INIT_TRACE") != nullptr;
  const auto _tt0 = std::chrono::steady_clock::now();
  auto _tlap = [&](const char *what) {
    if (!init_trace)
      return;
    std::fprintf(stderr, "[init-trace] %8.1f ms  %s\n",
                 std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now() - _tt0)
                   .count(),
                 what);
    std::fflush(stderr);
  };

  // RegisterCustomLayers
  registerCustomLayers();
  _tlap("registerCustomLayers");

  // create model and apply properties before compile()
  model = ml::train::createModel(ml::train::ModelType::NEURAL_NET);

  std::vector<std::string> model_props = {
    withKey("batch_size", BATCH_SIZE), withKey("epochs", "1"),
    withKey("model_tensor_type", MODEL_TENSOR_TYPE)};
  if (MEMORY_SWAP) {
    model_props.emplace_back(withKey("fsu", "true"));
    model_props.emplace_back(withKey("fsu_lookahead", FSU_LOOKAHEAD));
  }
  model->setProperty(model_props);

  // The model's activation-plane preference reaches the OpenCL pool through
  // the residency policy. Written for every model, before the graph is built,
  // so a model loaded after another one does not inherit its answer.
  nntrainer::ResidencyPolicy::global().prefer_private_device_plane =
    prefersPrivateActivationPlane();

  // build symbolic tensor graph and compile from (input, output)
  auto [x, y] = constructModel();
  _tlap("constructModel (symbolic graph)");

  if (model->compile(x, y, ml::train::ExecutionMode::INFERENCE)) {
    throw std::invalid_argument("Model compilation failed.");
  }
  _tlap("model->compile+initialize (ccapi)");

  // Hand the backend this model's graph-replay feed nodes, now that the graph
  // exists and the names are final. A model that declares none (the default)
  // leaves the list empty, and the CUDA decode-graph arm refuses to replay
  // without it -- so an unchecked model gets no behaviour change at all, and
  // the worst case of a wrong list is a missing speedup rather than wrong
  // tokens. Same static_cast the incremental-inference override already uses:
  // the ccapi Model has no passthrough for this and inventing one would widen
  // the public surface for a backend hint.
  if (auto feed = getGraphReplayFeedNodes(); !feed.empty()) {
    auto *nn = static_cast<nntrainer::NeuralNetwork *>(model.get());
    nn->setGraphReplayFeedNodes(std::move(feed));
    // And the position the capture stops being valid at, so the backend can
    // recapture instead of replaying a graph built for the other window regime.
    nn->setGraphReplayExpiryPosition(getGraphReplayExpiryPosition());
  }

  is_initialized = true;
#ifdef DEBUG
  model->summarize(std::cout, ML_TRAIN_SUMMARY_MODEL);
#endif
}

void Transformer::setDecisionTask(const std::vector<int32_t> &letter_ids,
                                  const std::string &objective,
                                  unsigned int group, unsigned int seed) {
  NNTR_THROW_IF(letter_ids.size() != 26, std::invalid_argument)
    << "a decision task needs 26 letter ids, got " << letter_ids.size();
  NNTR_THROW_IF(objective != "sft" && objective != "soft" &&
                  objective != "rlcd" && objective != "rlvr",
                std::invalid_argument)
    << "decision objective must be sft, soft, rlcd or rlvr, got " << objective;
  std::string ids;
  for (int32_t id : letter_ids)
    ids += (ids.empty() ? "" : ",") + std::to_string(id);
  TRAIN_TASK = "decision";
  DECISION_LETTER_IDS = ids;
  DECISION_OBJECTIVE = objective;
  DECISION_GROUP = group;
  DECISION_SEED = seed;
}

/**
 * @brief Build and compile the symbolic transformer graph for LoRA training.
 */
void Transformer::initializeForTraining(float lr, unsigned int epochs,
                                        float lr_decay_rate,
                                        float lr_decay_steps) {
  FOR_TRAINING = true;

  registerCustomLayers();

  model = ml::train::createModel(ml::train::ModelType::NEURAL_NET);

  std::vector<std::string> model_props = {
    withKey("batch_size", BATCH_SIZE),
    withKey("epochs", std::to_string(epochs)),
    withKey("model_tensor_type", MODEL_TENSOR_TYPE)};
  if (ACTIVATION_CKPT)
    model_props.push_back(withKey("activation_checkpointing", "true"));
  model->setProperty(model_props);

  std::vector<std::string> opt_props = {withKey("learning_rate", lr)};
  if (lr_decay_rate > 0.0f && lr_decay_steps > 0.0f) {
    // nntrainer's built-in exponential decay: lr *= decay_rate every
    // decay_steps global iterations. With 90 samples/epoch and
    // decay_steps 90 the lr decays once per epoch.
    opt_props.push_back(withKey("decay_rate", lr_decay_rate));
    opt_props.push_back(withKey("decay_steps", lr_decay_steps));
  }
  auto optimizer = ml::train::createOptimizer("adam", opt_props);
  if (model->setOptimizer(std::move(optimizer))) {
    throw std::invalid_argument("Failed to set optimizer.");
  }

  // The training graph already ends in its loss: the lm head of a training
  // graph is the lm_head_loss layer (see CausalLM::buildLmHeadOutput()).
  auto [x, y] = constructModel();

  // compile(Tensor, Tensor, mode) internally compiles + initializes.
  if (model->compile(x, y, ml::train::ExecutionMode::TRAIN)) {
    throw std::invalid_argument("Training model compilation failed.");
  }

  is_initialized = true;
}

/**
 * @brief Set the dataset used for training/validation.
 */
void Transformer::setDataset(const ml::train::DatasetModeType &mode,
                             std::shared_ptr<ml::train::Dataset> dataset) {
  if (model->setDataset(mode, dataset)) {
    throw std::invalid_argument("Failed to set dataset.");
  }
}

/**
 * @brief Run training for the epochs configured in initializeForTraining().
 */
void Transformer::train(
  std::function<void(void *)> epoch_complete_cb, void *epoch_data,
  std::function<bool(void *)> stop_cb, void *stop_data,
  std::function<void(unsigned int, float, void *)> iteration_cb,
  void *iteration_data) {
  if (!stop_cb)
    stop_cb = [](void *) { return false; };
  if (!epoch_complete_cb)
    epoch_complete_cb = [](void *) {};
  if (!iteration_cb)
    iteration_cb = [](unsigned int, float, void *) {};

  if (model->train({}, stop_cb, stop_data, epoch_complete_cb, epoch_data,
                   iteration_cb, iteration_data)) {
    throw std::runtime_error("Training failed.");
  }
}

ml::train::RunStats Transformer::getTrainingStats() {
  return model->getTrainingStats();
}

ml::train::RunStats Transformer::getValidStats() {
  return model->getValidStats();
}

void Transformer::forEachLayer(
  std::function<void(ml::train::Layer &, nntrainer::RunLayerContext &, void *)>
    fn,
  void *user_data) {
  model->forEachLayer(fn, user_data);
}

/**
 * @brief Save only the loraA/loraB weights (raw FP32) to a file.
 */
void Transformer::save_weight_lora(const std::string &lora_path) {
  std::ofstream file(lora_path, std::ios::binary);
  NNTR_THROW_IF(!file.is_open(), std::runtime_error)
    << "Failed to open lora output file: " << lora_path;

  std::function<void(ml::train::Layer &, nntrainer::RunLayerContext &, void *)>
    fn = [&file](ml::train::Layer &l, nntrainer::RunLayerContext &context,
                 void *) {
      if (l.getType() != "fully_connected")
        return;
      for (auto &w : context.getWeights()) {
        const std::string &name = w->getName();
        if (name.find(":loraA") != std::string::npos ||
            name.find(":loraB") != std::string::npos)
          w->getVariableRef().save(file);
      }
    };
  model->forEachLayer(fn, nullptr);
}

namespace {

/**
 * @brief Which of the two QS4CX record strides a weight file was written at.
 * @note Same three answers as the loader's own probe (see
 *       Qs4cxRecordStride in neuralnet/models/neuralnet.cpp, whose logic this
 *       mirrors); UNDECIDED means neither signal could judge it.
 */
enum class BaseQs4cxStride { PADDED, TRIMMED, UNDECIDED };

/**
 * @brief Value of a "Key:\t NNN kB" line of /proc/self/status, or -1.
 */
long procStatusKb(const std::string &key) {
  std::ifstream status("/proc/self/status");
  if (!status.is_open())
    return -1;
  std::string line;
  while (std::getline(status, line)) {
    if (line.rfind(key, 0) != 0)
      continue;
    std::istringstream values(line.substr(key.size()));
    long kb = -1;
    if (values >> kb)
      return kb;
    return -1;
  }
  return -1;
}

/**
 * @brief Print VmHWM / VmRSS around a load, to show what it costs.
 * @note NNTR_LOAD_PROF=1. VmHWM is a high-water mark, so the "after" line of a
 *       load that only ever *drops* memory still reports the peak -- which is
 *       the point: it says what the load left behind, and the "before" line
 *       says what the load itself reached.
 */
void reportLoadMemory(const char *phase, const std::string &path) {
  if (std::getenv("NNTR_LOAD_PROF") == nullptr)
    return;
  std::fprintf(stderr, "[load-prof] %s VmHWM=%ld kB VmRSS=%ld kB  %s\n", phase,
               procStatusKb("VmHWM:"), procStatusKb("VmRSS:"), path.c_str());
  std::fflush(stderr);
}

/**
 * @brief Probe the first QS4CX record of a file to read off its stride.
 *
 * Ported from the file-static qs4cxRecordStride() in
 * nntrainer/models/neuralnet.cpp (keep the two in step); it is duplicated
 * rather than called because that one is private to the loader's translation
 * unit. The rationale for reading the record instead of trusting its size is
 * that one there, in short: the two strides differ only by the floor(N/2) pad
 * bytes between a record's nibbles and its scales, which the padded layout
 * leaves zero and the trimmed layout fills with the first per-channel scale --
 * a strictly positive finite reciprocal. So four bytes at (offset + trimmed
 * nibble bytes) tell the two apart. N < 8 has fewer than four pad bytes and
 * answers UNDECIDED rather than guessing.
 */
BaseQs4cxStride probeQs4cxStride(const std::string &path, size_t offset,
                                 size_t K, size_t N) {
  const size_t trimmed_nibbles =
    nntrainer::QS4CX_Tensor::nibbleBytes(K, N, false);
  const size_t padded_nibbles =
    nntrainer::QS4CX_Tensor::nibbleBytes(K, N, true);
  if (padded_nibbles - trimmed_nibbles < sizeof(float))
    return BaseQs4cxStride::UNDECIDED;

  std::ifstream probe_file(path, std::ios::in | std::ios::binary);
  if (!probe_file.is_open())
    return BaseQs4cxStride::UNDECIDED;
  probe_file.seekg(static_cast<std::streamoff>(offset + trimmed_nibbles),
                   std::ios::beg);
  float probe = 0.0f;
  probe_file.read(reinterpret_cast<char *>(&probe), sizeof(probe));
  if (!probe_file)
    return BaseQs4cxStride::UNDECIDED;

  return (std::isnormal(probe) && probe > 0.0f) ? BaseQs4cxStride::TRIMMED
                                                : BaseQs4cxStride::PADDED;
}

/**
 * @brief One base-weight record of a checkpoint, as laid out over the file by
 *        walkBaseCheckpointRecords().
 */
struct BaseWeightRecord {
  nntrainer::Weight *weight = nullptr; ///< the training graph's weight
  std::string layer_name;              ///< for the error messages
  size_t offset = 0;                   ///< first byte of the record
  size_t size = 0;        ///< bytes the record occupies (advances the offset)
  size_t data_offset = 0; ///< first payload byte (offset + a qscheme prefix)
  size_t payload = 0;     ///< bytes the read consumes (size - the prefix)
  size_t in_memory = 0;   ///< dst.getMemoryBytes(), for the mismatch report
  bool qs4cx = false;     ///< sized by the per-file QS4CX stride choice
  bool deferred = false;  ///< reads no bytes of its own (see the head alias)
  size_t alias = 0;       ///< deferred: index of the record it shares
};

/**
 * @brief What one pass of walkBaseCheckpointRecords() learned about a file.
 */
struct BaseWalkResult {
  /// which file the walk sizes; only set by the caller, read when peeking
  std::string path;
  size_t total = 0;       ///< bytes the records occupy in file order
  size_t num_records = 0; ///< weights that contributed a record
  bool stride_differs =
    false;           ///< some QS4CX record sizes differently per stride
  bool exact = true; ///< every record size is one this loader can trust
  bool first_qs4cx_found = false;
  size_t first_qs4cx_offset = 0;
  size_t first_qs4cx_K = 0;
  size_t first_qs4cx_N = 0;
};

/**
 * @brief Lay a base checkpoint's records over the file by walking THIS graph.
 *
 * A .bin checkpoint carries no header, no names and no dtype: it is the raw
 * payloads of the graph's weights in graph order, so a record's offset is the
 * running sum of the record sizes before it, computable from the in-memory
 * weights alone. That is what makes a direct load possible -- and it is why
 * this walk is a mirror of the prescan in NeuralNetwork::load() (its
 * `walk_records` lambda) rather than an independent reading of the format: one
 * disagreement in either direction silently shifts every record after it.
 *
 * The filters that decide whether a weight contributes a record are all
 * load-bearing:
 *  - a shared weight (a tied head reaches the same Tensor as its embedding
 *    through requestOrExtend) is written at its FIRST occurrence only, deduped
 *    here by &getVariableRef() exactly as the loader dedupes;
 *  - a weight whose tensor type prefixes its payload with a uint16 qscheme
 *    (CharTensor / ShortTensor / UIntTensor / Uint4QTensor / BCQTensor: the
 *    types that override save()/read() to write those two bytes) occupies 2
 *    more bytes than its buffer, which the read then skips.
 *
 * Layers whose save() override writes nothing need no case of their own, and
 * adding one here would only put this walk out of step with the loader, which
 * has none: a shared_fully_connected in shared_mode_ and a tied lm_head both
 * reach the owner's Tensor, so the dedup already covers them, and an embedding
 * with quantized_lut_path set requests no weight at all -- its finalize()
 * returns before requestWeight, so there is no record to skip.
 *
 * The LoRA adapters are excluded by name: they are not in a pretrained
 * checkpoint at all, which is precisely why this path cannot call model->load()
 * -- the ordinary loader is strictly sequential in TRAIN mode (no seeking), so
 * the extra loraA/loraB records would shift everything after the first
 * adapted FC.
 *
 * @param nn           the compiled training model, for its graph only
 * @param qs4cx_padded stride to lay the QS4CX records out at
 * @param legacy_int4  model_tensor_type starts with "QINT4": its int4 records
 *                     are the legacy on-disk form, not QS4CX records
 * @param records      out: the records, in file order
 * @param res          out: totals and the facts the stride choice needs
 * @param peek         stream used to peek a legacy QINT4 container header
 * @param peek_tried   whether @p peek has been opened yet
 */
void walkBaseCheckpointRecords(nntrainer::NeuralNetwork *nn, bool qs4cx_padded,
                               bool legacy_int4,
                               std::vector<BaseWeightRecord> &records,
                               BaseWalkResult &res, std::ifstream &peek,
                               bool &peek_tried) {
  res = BaseWalkResult{};
  records.clear();
  // A legacy QINT4 container is per record and only readable from the file,
  // so the walk needs a peeking stream of its own (as the loader does).
  auto peek_scheme = [&](size_t at, uint16_t &out) {
    if (!peek_tried) {
      peek_tried = true;
      peek.open(res.path, std::ios::in | std::ios::binary);
    }
    if (!peek.is_open())
      return false;
    peek.clear();
    peek.seekg(static_cast<std::streamoff>(at), std::ios::beg);
    peek.read(reinterpret_cast<char *>(&out), sizeof(out));
    return peek.good();
  };

  std::unordered_set<const nntrainer::Tensor *> visited;
  std::function<void(ml::train::Layer &, nntrainer::RunLayerContext &, void *)>
    collect =
      [&](ml::train::Layer &l, nntrainer::RunLayerContext &context, void *) {
        for (auto *w : context.getWeights()) {
          const std::string &name = w->getName();
          if (name.find(":loraA") != std::string::npos ||
              name.find(":loraB") != std::string::npos)
            continue;
          if (!visited.insert(&w->getVariableRef()).second)
            continue; // shared weight: written at its first occurrence

          BaseWeightRecord rec;
          rec.weight = w;
          rec.layer_name = l.getName();
          rec.offset = res.total;
          const auto dtype = w->getDim().getDataType();
          rec.qs4cx = dtype == ml::train::TensorDim::DataType::QS4CX;
          // A QS4CX record is sized -- and later read -- at the stride the walk
          // assumes, and getMemoryBytes() reports whichever stride the tensor
          // carries, so choose before asking.
          if (rec.qs4cx && !legacy_int4)
            w->getVariableRef().setQs4cxRecordPadded(qs4cx_padded);
          rec.in_memory = w->getVariableRef().getMemoryBytes();
          rec.size = rec.in_memory;
          w->getVariableRef().setFileOffset(res.total);
          rec.data_offset = rec.offset;
          rec.payload = rec.size;

          if (dtype != ml::train::TensorDim::DataType::FP32 &&
              dtype != ml::train::TensorDim::DataType::FP16 &&
              dtype != ml::train::TensorDim::DataType::Q6_K &&
              dtype != ml::train::TensorDim::DataType::Q4_0 &&
              dtype != ml::train::TensorDim::DataType::QS4CX) {
            // A qscheme header, per the loader's identical test. The five types
            // that really do write one are exact here; Q4_K and UINT4/8/16/32
            // are not (Q4_K writes no header at all, UINT4 with the Q4_Kx8
            // scheme materialises as a Q4_K tensor) -- the loader marks those
            // inexact and skips its own size check for them. A direct load
            // cannot afford that: see `exact` below.
            rec.size += sizeof(uint16_t);
            rec.data_offset += sizeof(uint16_t);
            if (dtype == ml::train::TensorDim::DataType::Q4_K ||
                dtype == ml::train::TensorDim::DataType::UINT4 ||
                dtype == ml::train::TensorDim::DataType::UINT8 ||
                dtype == ml::train::TensorDim::DataType::UINT16 ||
                dtype == ml::train::TensorDim::DataType::UINT32)
              res.exact = false;
          }

          if (rec.qs4cx && legacy_int4) {
            // A legacy QINT4 model builds QS4CX weights but its int4 records
            // are the legacy on-disk form; flag the tensor so read() transcodes
            // and size the record by the peeked container.
            w->getVariableRef().setOnDiskLegacyQint4(true);
            uint16_t disk_scheme = 0xFFFF;
            if (!peek_scheme(rec.offset, disk_scheme))
              res.exact = false;
            const nntrainer::TensorDim &d = w->getDim();
            rec.size = sizeof(uint16_t);
            if (disk_scheme ==
                static_cast<uint16_t>(nntrainer::QScheme::PER_CHANNEL_AFFINE))
              rec.size += nntrainer::Int4Utils::plainRecordPayloadBytes(
                d.width(), d.height());
            else
              rec.size += nntrainer::Int4Utils::kaiNibblePayloadBytes(
                            d.width(), d.height()) +
                          static_cast<size_t>(d.width()) * sizeof(uint16_t);
            rec.data_offset = rec.offset; // the transcode reads from record 0
            rec.payload = rec.size;
          } else if (rec.qs4cx) {
            const nntrainer::TensorDim &d = w->getDim();
            if (nntrainer::QS4CX_Tensor::recordBytes(d.height(), d.width(),
                                                     !qs4cx_padded) != rec.size)
              res.stride_differs = true;
            if (!res.first_qs4cx_found) {
              res.first_qs4cx_found = true;
              res.first_qs4cx_offset = rec.offset;
              res.first_qs4cx_K = d.height();
              res.first_qs4cx_N = d.width();
            }
          }
          res.total += rec.size;
          records.push_back(std::move(rec));
        }
      };
  nn->forEachLayer(collect, nullptr);
}

} // namespace

/**
 * @brief Load a pretrained (non-LoRA) base checkpoint into a graph that has
 *        LoRA weights, then optionally overlay a saved LoRA adapter.
 */
void Transformer::load_weight_lora(const std::string &base_path,
                                   const std::string &lora_path) {
  NNTR_THROW_IF(!is_initialized, std::runtime_error)
    << "Transformer model is not initialized. Please call "
       "initializeForTraining() before load_weight_lora().";

  // Read the base checkpoint straight into this graph's weights: lay its
  // records over the file by walking THIS graph, then read each one into the
  // weight it was laid out for.
  //
  // This used to build a throwaway full no-LoRA inference model, load the
  // checkpoint into it with the ordinary loader, memcpy every weight across by
  // name and destroy the copy. On the S26U at batch 1 that copy WAS the
  // process peak -- 709 MB of VmHWM against 530-600 MB of training RSS, and
  // ~180 MB above even a checkpointed training run -- and it bought nothing but
  // a set of offsets, which are a prefix sum of record sizes this graph can
  // compute for itself. A .bin checkpoint is the payloads of the graph's
  // weights in graph order and nothing else, so the two graphs' layouts agree
  // by construction: the same layer requests the same weight at the same dtype
  // (weight_dtype is read from nntr_config, not from LoRA_RANK or FOR_TRAINING)
  // and shares it with the same neighbour. Where that stops being true the
  // size guard below refuses the file rather than training on shifted bytes.
  nntrainer::NeuralNetwork *nn =
    static_cast<nntrainer::NeuralNetwork *>(model.get());
  const std::string weight_path = base_path; // @todo support "ini;bin" pairs?
  const auto format = formatFromExtension(base_path);
  const std::string model_tensor_type_str = MODEL_TENSOR_TYPE;
  const bool legacy_int4 = model_tensor_type_str.rfind("QINT4", 0) == 0;

  // Only the .bin layout this walk reproduces. A .safetensors file names every
  // tensor and carries its own offsets, so laying records out by graph order
  // would read the wrong bytes without saying so -- and before this guard it
  // silently loaded nothing at all, training from whatever the initializers
  // left behind while reporting success.
  NNTR_THROW_IF(format != ml::train::ModelFormat::MODEL_FORMAT_BIN,
                std::runtime_error)
    << "load_weight_lora: '" << base_path
    << "' is not a .bin checkpoint. This loader locates each weight as the "
       "running sum of the record sizes before it, which is the .bin layout "
       "only; a format that names its tensors needs a different one.";

  reportLoadMemory("before", weight_path);

  std::error_code ec;
  const auto file_bytes = std::filesystem::file_size(weight_path, ec);
  NNTR_THROW_IF(ec, std::runtime_error) << "load_weight_lora: cannot stat '"
                                        << weight_path << "': " << ec.message();
  const size_t actual = static_cast<size_t>(file_bytes);

  std::vector<BaseWeightRecord> records;
  BaseWalkResult res;
  res.path = weight_path;
  std::ifstream peek;
  bool peek_tried = false;
  walkBaseCheckpointRecords(nn, /*qs4cx_padded=*/true, legacy_int4, records,
                            res, peek, peek_tried);

  // Tell the two QS4CX record strides apart, per FILE, exactly as the loader
  // does: the record names neither, but a file comes from one exporter.
  //   - size, when it can speak: trimmed_total <= actual < padded_total can
  //     only be trimmed, since no tail makes a padded file that short;
  //   - the first record's own bytes when size cannot, because save() appends
  //     an adam block and 8 bytes of training metadata after the weight
  //     bodies, and the two totals differ by only sum(floor(N/2)).
  // The loader falls back to padded (with a warning) when neither speaks;
  // this path refuses instead, because here there is no second signal -- a
  // plain "the file is N bytes longer" tail says nothing about the stride, and
  // a trimmed file read as padded slides every per-channel scale by
  // floor(N/2) bytes: plausible output, every channel scaled wrong.
  size_t trimmed_total = 0;
  // Which QS4CX record stride the file turns out to use. Padded until a file
  // says otherwise: it is what the first walk sized (and so stamped) every
  // record at.
  bool padded_choice = true;
  if (res.stride_differs) {
    std::vector<BaseWeightRecord> trimmed_records;
    BaseWalkResult trimmed_res;
    trimmed_res.path = weight_path;
    walkBaseCheckpointRecords(nn, /*qs4cx_padded=*/false, legacy_int4,
                              trimmed_records, trimmed_res, peek, peek_tried);
    trimmed_total = trimmed_res.total;
    NNTR_THROW_IF(!trimmed_res.stride_differs, std::runtime_error)
      << "load_weight_lora: internal inconsistency -- the QS4CX stride choice "
         "changed the total but not the per-record sizes";

    // A file no shorter than the padded total and no longer than the trimmed
    // one can only be trimmed: no tail makes a padded file that short.
    padded_choice = !(trimmed_total <= actual && actual < res.total);
    if (padded_choice && res.total <= actual && res.first_qs4cx_found) {
      switch (probeQs4cxStride(weight_path, res.first_qs4cx_offset,
                               res.first_qs4cx_K, res.first_qs4cx_N)) {
      case BaseQs4cxStride::TRIMMED:
        padded_choice = false;
        break;
      case BaseQs4cxStride::PADDED:
        break;
      case BaseQs4cxStride::UNDECIDED:
        throw std::runtime_error(
          "load_weight_lora: cannot tell which QS4CX record stride '" +
          weight_path + "' (" + std::to_string(actual) +
          " bytes) was written at: it fits neither stride exclusively (padded "
          "total " +
          std::to_string(res.total) + ", trimmed total " +
          std::to_string(trimmed_total) + ") and its first QS4CX record (" +
          std::to_string(res.first_qs4cx_K) + " x " +
          std::to_string(res.first_qs4cx_N) +
          ") is too narrow to read the stride off. Reading a trimmed file at "
          "the padded stride would scale every channel by its neighbour's "
          "factor, so the load is refused rather than guessed.");
      }
    }
    if (!padded_choice) {
      ml_logi("weight file '%s' matches the trimmed QS4CX record stride "
              "(%zu bytes) rather than the padded one (%zu); reading it as "
              "trimmed",
              weight_path.c_str(), trimmed_total, res.total);
      records = std::move(trimmed_records);
      res.total = trimmed_total;
      res.num_records = trimmed_res.num_records;
      res.exact = trimmed_res.exact;
    }
  }

  // Stamp the tensors with the layout that won, before anything reads them.
  //
  // A walk leaves every tensor it sized stamped for the stride IT used, and the
  // stride decision above may have run the other walk last -- so whichever
  // stride was chosen, the tensors have to be restamped from the records that
  // are about to be read, or a padded file would be read at the trimmed stride
  // (which does not shift the nibbles, it slides every per-channel scale by
  // floor(N/2) bytes: plausible output, each channel scaled by its neighbour's
  // factor). This is the two calls the walk itself makes per weight.
  for (const auto &rec : records) {
    nntrainer::Tensor &var = rec.weight->getVariableRef();
    if (rec.qs4cx && !legacy_int4)
      var.setQs4cxRecordPadded(padded_choice);
    var.setFileOffset(rec.offset);
  }

  // The on-disk contract the offsets above assume, checked before a single byte
  // is read. The .bin records no per-tensor dtype, so a weight's position is
  // nothing but this running sum: if any layer asks for a dtype or stride other
  // than the one the exporting graph wrote, every following weight is read at
  // the wrong offset, and that surfaces as garbage output far from the cause.
  // The ordinary loader compares the same two numbers (neuralnet.cpp, after its
  // own prescan); it can afford to accept a longer file because a save()
  // appends an adam block and 8 bytes of training metadata after the weight
  // bodies.
  //
  // One dtype makes the sum untrustworthy: Q4_K and UINT4/8/16/32 are counted
  // with a qscheme header the tensor does not actually write (only the five
  // types that override save()/read() do), so a graph holding one is skipped,
  // exactly as the loader skips its own check there.
  if (res.exact) {
    NNTR_THROW_IF(actual < res.total, std::runtime_error)
      << "load_weight_lora: weight file is smaller than this graph expects: '"
      << weight_path << "' holds " << actual << " bytes, but the "
      << res.num_records << " weights of this graph account for " << res.total
      << " (short by " << (res.total - actual)
      << "). The model file records no per-tensor dtype, so the first thing to "
         "suspect is a layer requesting a weight at a different dtype or "
         "stride "
         "than the package was exported with -- that shifts every later "
         "weight's offset. A truncated or mismatched file does it too.";
    if (actual > res.total)
      ml_logw("weight file '%s' is %zu bytes longer than the %zu bytes this "
              "graph accounts for; reading the leading %zu and ignoring the "
              "tail",
              weight_path.c_str(), actual - res.total, res.total, res.total);
  }

  // Read every record into the weight it was laid out for. The walk already
  // stamped each tensor's file offset, and Tensor::read() with no offset
  // argument picks that up -- the same entry point the ordinary loader reaches
  // through Layer::read(), so each tensor type keeps its own read()
  // (the qscheme-header types skip their own 2 bytes, a legacy QINT4 record
  // transcodes into its QS4CX tensor, and so on) rather than this path
  // reproducing any of it.
  //
  // The fp32 shadow is copied after each read exactly as Layer::read() does it,
  // because the mixed-precision GEMMs read that one and it would otherwise stay
  // at whatever the initializer left in it.
  //
  // Which weights have a record is taken from the walk itself, by Tensor
  // identity -- the same identity it deduped on. The tensors' file offsets
  // cannot answer it: an unstamped tensor reads back 0, and 0 is also where the
  // first record lives.
  std::unordered_set<const nntrainer::Tensor *> stamped;
  for (const auto &rec : records)
    stamped.insert(&rec.weight->getVariableRef());
  NNTR_THROW_IF(stamped.size() != records.size(), std::runtime_error)
    << "load_weight_lora: internal inconsistency -- " << records.size()
    << " records over " << stamped.size() << " distinct weights";

  std::ifstream model_file = nntrainer::checkedOpenStream<std::ifstream>(
    weight_path, std::ios::in | std::ios::binary);

  unsigned int loaded = 0;
  std::function<void(ml::train::Layer &, nntrainer::RunLayerContext &, void *)>
    apply =
      [&](ml::train::Layer &l, nntrainer::RunLayerContext &context, void *) {
        auto weights = context.getWeights();
        for (unsigned int i = 0; i < weights.size(); ++i) {
          nntrainer::Weight *w = weights[i];
          const std::string &name = w->getName();
          // The LoRA A/B are this graph's own and were never in a pretrained
          // checkpoint; a shared weight was laid out at its first occurrence
          // only, so it has no record here and keeps the bytes read for its
          // owner. Anything else without a record means the graph and the file
          // disagree, which is what the size check above is for -- but only a
          // per-weight report can say WHICH one, and training on an
          // uninitialized weight is silent, so name it and refuse.
          if (name.find(":loraA") != std::string::npos ||
              name.find(":loraB") != std::string::npos)
            continue;
          NNTR_THROW_IF(stamped.find(&w->getVariableRef()) == stamped.end(),
                        std::runtime_error)
            << "load_weight_lora: weight '" << name << "' of layer '"
            << l.getName() << "' got no record in '" << weight_path
            << "'. Its count or dtype differs from the graph the file was "
               "written "
               "from; loading would train on an uninitialized weight.";
          // Seek each record from its own stamped offset rather than reading
          // sequentially: start_offset = max() makes Tensor::read() substitute
          // the tensor's file offset, and read_from_offset is what makes it
          // seek (the ordinary loader's INFERENCE arm passes the same pair).
          // Sequential reading would couple every record to what the one before
          // it actually consumed -- and the two differ by 2 bytes on a
          // qscheme-header type, which drifts the rest of the file off by one
          // and ends in a short read.
          w->getVariableRef().read(model_file,
                                   std::numeric_limits<size_t>::max(), true);
          ++loaded;
          if (context.isMixedPrecision(i) && !context.getWeightFP32(i).empty())
            context.getWeightFP32(i).copyData(w->getVariableRef());
        }
      };
  model->forEachLayer(apply, nullptr);

  NNTR_THROW_IF(loaded == 0, std::runtime_error)
    << "load_weight_lora: no weights were read from " << weight_path;
  reportLoadMemory("after", weight_path);
  ml_logd("load_weight_lora: read %u base weight(s) from '%s' (%zu bytes)",
          loaded, weight_path.c_str(), res.total);

  // Overlay a previously saved LoRA adapter, if given.
  if (!lora_path.empty()) {
    std::ifstream lora_file(lora_path, std::ios::binary);
    NNTR_THROW_IF(!lora_file.is_open(), std::runtime_error)
      << "Failed to open lora file: " << lora_path;

    std::function<void(ml::train::Layer &, nntrainer::RunLayerContext &,
                       void *)>
      load_lora = [&lora_file](ml::train::Layer &l,
                               nntrainer::RunLayerContext &context, void *) {
        if (l.getType() != "fully_connected")
          return;
        for (auto &w : context.getWeights()) {
          const std::string &name = w->getName();
          if (name.find(":loraA") != std::string::npos ||
              name.find(":loraB") != std::string::npos)
            w->getVariableRef().read(lora_file);
        }
      };
    model->forEachLayer(load_lora, nullptr);
  }
}

namespace {

// ---------------------------------------------------------------------------
// qint4 LoRA adapter file -- the same per-channel int4 scheme as the QS4CX
// base weight: one fp32 scale per output channel (the weight's width, N),
// int4 codes in [-8, 7] stored as uint4 + 8, value = (code - 8) * scale[n].
//
//   u32 magic "NQI4", u32 version
//   per loraA / loraB, in graph order:
//     u32 K, u32 N
//     N * ceil(K/2) nibble bytes: row n of the (K x N) weight, even k in the
//         low nibble, odd k in the high nibble
//     N fp32 scales
//
// The record is the trimmed QS4CX plain layout, so an adapter is byte-valid
// as a base weight of the same shape. FullyConnectedLayer::fakeQuantizeQInt4
// calibrates exactly these scales during QAT, so the file reproduces
// bit-for-bit what training quantized against.
// ---------------------------------------------------------------------------
constexpr uint32_t QINT4_ADAPTER_MAGIC = 0x3449514E; // "NQI4"
constexpr uint32_t QINT4_ADAPTER_VERSION = 1;

/**
 * @brief Quantize + pack a (K x N) LoRA weight into a qint4 record (nibbles
 *        then per-channel scales). With @p channel_d (the QAT per-channel
 *        EMA scales, indexed by output channel) those scales are force-fed
 *        so the adapter is exactly what training quantized against;
 *        otherwise, and for a channel QAT never scaled, the natural scale
 *        d = amax/7 (PTQ).
 */
std::vector<uint8_t> buildQInt4Record(const float *data_KN, size_t K, size_t N,
                                      const std::vector<float> *channel_d) {
  const size_t row_bytes = (K + 1) / 2;
  std::vector<uint8_t> out(N * row_bytes + N * sizeof(float));
  for (size_t n = 0; n < N; ++n) {
    float amax = 0.0f;
    for (size_t k = 0; k < K; ++k) {
      const float v = data_KN[k * N + n];
      if (std::abs(v) > amax)
        amax = std::abs(v);
    }
    float d = amax / 7.0f;
    if (channel_d != nullptr && n < channel_d->size() &&
        (*channel_d)[n] > 1e-10f)
      d = (*channel_d)[n];
    std::memcpy(out.data() + N * row_bytes + n * sizeof(float), &d,
                sizeof(float));
    uint8_t *row = out.data() + n * row_bytes;
    auto code = [d](float x) {
      if (d < 1e-10f)
        return 8;
      return std::max(0, std::min(15, (int)std::round(x / d) + 8));
    };
    for (size_t k = 0; k < K; ++k) {
      const uint8_t c = static_cast<uint8_t>(code(data_KN[k * N + n]));
      if ((k & 1) == 0)
        row[k / 2] = c; // even k: low nibble
      else
        row[k / 2] |= static_cast<uint8_t>(c << 4); // odd k: high nibble
    }
  }
  return out;
}

/** @brief Dequantize a qint4 record (nibbles + per-channel scales) into a
 *        (K x N) weight. */
void dequantizeQInt4Record(const uint8_t *rec, size_t K, size_t N,
                           float *data_KN) {
  const size_t row_bytes = (K + 1) / 2;
  for (size_t n = 0; n < N; ++n) {
    float d;
    std::memcpy(&d, rec + N * row_bytes + n * sizeof(float), sizeof(float));
    const uint8_t *row = rec + n * row_bytes;
    for (size_t k = 0; k < K; ++k) {
      const int code = ((k & 1) == 0) ? (row[k / 2] & 0x0F) : (row[k / 2] >> 4);
      data_KN[k * N + n] = (code - 8) * d;
    }
  }
}

} // namespace

/**
 * @brief Save LoRA adapters as qint4. When LORA_QAT calibrated per-channel
 *        EMA scales during training, force-feed them; otherwise fall back to
 *        a natural (recomputed) per-channel scale, i.e. plain PTQ.
 */
void Transformer::save_weight_lora_qint4(const std::string &path) {
  NNTR_THROW_IF(!is_initialized, std::runtime_error)
    << "Model not initialized before save_weight_lora_qint4().";

  std::ofstream f(path, std::ios::binary);
  NNTR_THROW_IF(!f.is_open(), std::runtime_error)
    << "Failed to open " << path << " for writing.";
  f.write(reinterpret_cast<const char *>(&QINT4_ADAPTER_MAGIC),
          sizeof(QINT4_ADAPTER_MAGIC));
  f.write(reinterpret_cast<const char *>(&QINT4_ADAPTER_VERSION),
          sizeof(QINT4_ADAPTER_VERSION));

  size_t total_bytes = 0;
  std::function<void(ml::train::Layer &, nntrainer::RunLayerContext &, void *)>
    fn = [&](ml::train::Layer &l, nntrainer::RunLayerContext &context, void *) {
      if (l.getType() != "fully_connected")
        return;
      const std::string lname = l.getName();

      for (auto &w : context.getWeights()) {
        const std::string &wname = w->getName();
        const bool is_loraA = wname.find(":loraA") != std::string::npos;
        const bool is_loraB = wname.find(":loraB") != std::string::npos;
        if (!is_loraA && !is_loraB)
          continue;

        const nntrainer::Tensor &t = w->getVariableRef();
        NNTR_THROW_IF(t.getDataType() != ml::train::TensorDim::DataType::FP32,
                      std::runtime_error)
          << "save_weight_lora_qint4: '" << wname << "' is not FP32";
        const uint32_t K = static_cast<uint32_t>(t.getDim().height());
        const uint32_t N = static_cast<uint32_t>(t.getDim().width());

        // QAT: force-feed the EMA channel scales calibrated during training.
        // Non-QAT: natural (PTQ) scales.
        std::vector<float> channel_d;
        if (LORA_QAT) {
          auto [a_cd, b_cd] =
            nntrainer::FullyConnectedLayer::getRegisteredChannelScales(lname);
          channel_d = is_loraA ? a_cd : b_cd;
        }
        const std::vector<uint8_t> rec = buildQInt4Record(
          t.getData<float>(), K, N, channel_d.empty() ? nullptr : &channel_d);

        f.write(reinterpret_cast<const char *>(&K), sizeof(K));
        f.write(reinterpret_cast<const char *>(&N), sizeof(N));
        f.write(reinterpret_cast<const char *>(rec.data()), rec.size());
        total_bytes += rec.size();
      }
    };
  model->forEachLayer(fn, nullptr);

  std::cout << "[save_weight_lora_qint4] Saved qint4 LoRA adapters to " << path
            << " (" << (total_bytes / 1024) << " KB)" << std::endl;
}

/**
 * @brief Load a pretrained (non-LoRA) base checkpoint, then overlay a qint4
 *        LoRA adapter, dequantized into the graph's FP32 loraA/loraB.
 */
void Transformer::load_weight_lora_qint4(const std::string &base_path,
                                         const std::string &lora_qint4_path) {
  // the base checkpoint goes in exactly as for an FP32 adapter
  load_weight_lora(base_path, "");

  std::ifstream f(lora_qint4_path, std::ios::binary);
  NNTR_THROW_IF(!f.is_open(), std::runtime_error)
    << "Failed to open qint4 LoRA file: " << lora_qint4_path;
  uint32_t magic = 0, version = 0;
  f.read(reinterpret_cast<char *>(&magic), sizeof(magic));
  f.read(reinterpret_cast<char *>(&version), sizeof(version));
  NNTR_THROW_IF(!f || magic != QINT4_ADAPTER_MAGIC, std::runtime_error)
    << "load_weight_lora_qint4: " << lora_qint4_path
    << " is not a qint4 LoRA adapter (no NQI4 header). Q4_0 adapter files "
       "saved before the qint4 unification are no longer supported; re-train "
       "and re-save the adapter.";
  NNTR_THROW_IF(version != QINT4_ADAPTER_VERSION, std::runtime_error)
    << "load_weight_lora_qint4: unsupported adapter version " << version;

  std::vector<uint8_t> rec;
  std::function<void(ml::train::Layer &, nntrainer::RunLayerContext &, void *)>
    fn = [&](ml::train::Layer &l, nntrainer::RunLayerContext &context, void *) {
      if (l.getType() != "fully_connected")
        return;

      for (auto &w : context.getWeights()) {
        const std::string &wname = w->getName();
        if (wname.find(":loraA") == std::string::npos &&
            wname.find(":loraB") == std::string::npos)
          continue;

        uint32_t k = 0, n = 0;
        f.read(reinterpret_cast<char *>(&k), sizeof(k));
        f.read(reinterpret_cast<char *>(&n), sizeof(n));
        NNTR_THROW_IF(!f, std::runtime_error)
          << "load_weight_lora_qint4: failed reading shape at '" << wname
          << "'";

        nntrainer::Tensor &dst = w->getVariableRef();
        NNTR_THROW_IF(dst.getDataType() != ml::train::TensorDim::DataType::FP32,
                      std::runtime_error)
          << "load_weight_lora_qint4: '" << wname
          << "' is not FP32 (do not build the graph with lora_weight_q4)";
        NNTR_THROW_IF(k != dst.getDim().height() || n != dst.getDim().width(),
                      std::runtime_error)
          << "load_weight_lora_qint4: shape mismatch for '" << wname
          << "': file=" << k << "x" << n << " model=" << dst.getDim().height()
          << "x" << dst.getDim().width();

        rec.resize((size_t)n * ((k + 1) / 2) + (size_t)n * sizeof(float));
        f.read(reinterpret_cast<char *>(rec.data()), rec.size());
        NNTR_THROW_IF(!f, std::runtime_error)
          << "load_weight_lora_qint4: failed reading qint4 data for '" << wname
          << "'";
        dequantizeQInt4Record(rec.data(), k, n, dst.getData<float>());
      }
    };
  model->forEachLayer(fn, nullptr);

  NNTR_THROW_IF(f.peek() != std::char_traits<char>::eof(), std::runtime_error)
    << "load_weight_lora_qint4: " << lora_qint4_path
    << " holds more adapters than the graph has LoRA weights";
  std::cout << "[load_weight_lora_qint4] Loaded qint4 LoRA adapters from "
            << lora_qint4_path << std::endl;
}

/**
 * @brief Construct the default decoder-only transformer graph.
 */
std::pair<Tensor, Tensor> Transformer::constructModel() {

  // CPU head of a GPU-trunk / CPU-head split: input0 is the trunk's last
  // hidden state, so there is no embedding; blocks start at HEAD_FROM_LAYER.
  if (HEAD_FROM_LAYER > 0) {
    Tensor x = Tensor({1, 1, static_cast<unsigned int>(INIT_SEQ_LEN),
                       static_cast<unsigned int>(DIM)},
                      "input0");
    Tensor h = x;
    for (int i = HEAD_FROM_LAYER; i < NUM_LAYERS; ++i)
      h = createTransformerDecoderBlock(i, h);
    std::vector<std::string> head_norm_props = {
      withKey("name", "output_norm"),
      withKey("epsilon", std::to_string(NORM_EPS)), withKey("packed", "false"),
      withKey("engine", layerEngine())};
    if (LORA_RANK > 0 && !TRAIN_NORMS)
      head_norm_props.push_back(withKey("trainable", "false"));
    LayerHandle head_norm(createLayer("rms_norm", head_norm_props));
    return {x, head_norm(h)};
  }

  // input
  Tensor x =
    Tensor({1, 1, 1, static_cast<unsigned int>(INIT_SEQ_LEN)}, "input0");

  // embedding
  const std::string embedding_type =
    TIE_WORD_EMBEDDINGS ? "tie_word_embeddings" : "embedding_layer";

  NNTR_THROW_IF(TIE_WORD_EMBEDDINGS && !EMBEDDING_FILE_NAME.empty(),
                std::invalid_argument)
    << "embedding_file_name requires untied embedding_layer";
  auto embedding_props =
    buildEmbeddingLayerProperties("embedding0", NUM_VOCAB, DIM, EMBEDDING_DTYPE,
                                  EMBEDDING_SCALE, EMBEDDING_FILE_NAME);
  // embedding0 is never a LoRA target: freeze it whenever LoRA is active.
  if (LORA_RANK > 0)
    embedding_props.push_back(withKey("trainable", "false"));
  LayerHandle embedding(createLayer(embedding_type, embedding_props));
  Tensor h = embedding(x);

  // transformer decoder blocks
  const int last_block = TRUNK_LAYERS > 0 ? TRUNK_LAYERS : NUM_LAYERS;
  for (int i = 0; i < last_block; ++i) {
    h = createTransformerDecoderBlock(i, h);
  }
  // GPU trunk of a GPU-trunk / CPU-head split: stop at the last frozen block.
  if (TRUNK_LAYERS > 0)
    return {x, h};

  // final rms_norm. NOTE: stays on CausalLM's "rms_norm" type (the custom
  // RMSNormLayer on cpu, RMSNormLayerGPU on gpu, CudaRMSNormLayer on cuda --
  // see registerCustomLayers) so the fused-rmsq + v8c FC consumer chain works.
  // The nntrainer GPU RMSNormLayerCl uses type "rmsnorm" (different) and has a
  // separate reduction-order drift issue.
  std::vector<std::string> out_norm_props = {
    withKey("name", "output_norm"),
    withKey("epsilon", std::to_string(NORM_EPS)), withKey("packed", "false"),
    withKey("engine", layerEngine())};
  // never a LoRA target: freeze it whenever LoRA is active
  if (LORA_RANK > 0 && !TRAIN_NORMS)
    out_norm_props.push_back(withKey("trainable", "false"));
  LayerHandle out_norm(createLayer("rms_norm", out_norm_props));
  h = out_norm(h);

  return {x, h};
};

std::vector<std::string> Transformer::buildEmbeddingLayerProperties(
  const std::string &name, unsigned int in_dim, unsigned int out_dim,
  const std::string &weight_dtype, float scale,
  const std::string &quantized_lut_path) const {
  std::vector<std::string> props = {
    withKey("name", name),
    withKey("in_dim", std::to_string(in_dim)),
    withKey("weight_dtype", weight_dtype),
    withKey("out_dim", std::to_string(out_dim)),
    withKey("scale", std::to_string(scale)),
  };

  if (!quantized_lut_path.empty())
    props.emplace_back(withKey("quantized_lut_path", quantized_lut_path));

  return props;
}

/**
 * @brief Load model weights from a binary nntrainer model file.
 */
void Transformer::load_weight(const std::string &weight_path) {
  if (!is_initialized) {
    throw std::runtime_error(
      "Transformer model is not initialized. Please call "
      "initialize() before load_weight().");
  }

  try {
    model->load(weight_path, formatFromExtension(weight_path));
  } catch (const std::exception &e) {
    throw std::runtime_error("Failed to load model weights: " +
                             std::string(e.what()));
  }
  /** Recorded so growPrefillPlane() can replay this exact load after rebuilding
   *  the graph at a taller plane. Stored verbatim: some callers hand a file,
   *  some a model directory, and the override that resolves it is the same one
   *  either way. */
  weight_path_ = weight_path;

  /** Pack the QS4CX weights for the CPU GEMM here, as part of the load,
   *  rather than leaving it to whichever caller remembers repack_weight(). A
   *  load path that did not call it (the library API) used to pay for the
   *  pack inside the first request instead -- every FC of an fp16-activation
   *  graph packed lazily on its first forward, an untied head on the first
   *  decode token -- and an fp32-activation graph re-packed every FC on every
   *  call. repack_weight() is idempotent, so a caller that still calls it
   *  afterwards (the CLI) does no extra work, and it is a no-op on the GPU
   *  engines. A swap (FSU) model keeps packing on demand: it streams weights
   *  precisely so that they are not all resident at once. */
  if (!MEMORY_SWAP)
    repack_weight();
};

unsigned int Transformer::prefillPlaneFor(unsigned int prompt_tokens) const {
  return causallm::prefillPlaneFor(
    prompt_tokens, static_cast<unsigned int>(INIT_SEQ_LEN), PACK_INIT_SEQ_LEN,
    static_cast<unsigned int>(MAX_SEQ_LEN));
}

bool Transformer::growPrefillPlane(unsigned int prompt_tokens) {
  if (plane_pinned_ || !is_initialized || weight_path_.empty())
    return false;
  /** The arm has to be one a bigger block pays off on -- see
   *  causallm::prefillBlockPays() for the measurement on each. */
  if (!causallm::prefillBlockPays())
    return false;
  /** A model that does not chunk its prefill by design is bounded by its plane,
   *  so moving the plane would move which prompts it can answer at all. Only a
   *  chunk-by-design model (the ring models) is grown, and for it the plane is
   *  purely a speed/memory dial -- the prompt it accepts is the same either
   * way.
   */
  if (!chunkLongPrompts())
    return false;
  if (MEMORY_SWAP) // FSU re-plans the swap around the graph; out of scope
    return false;

  const unsigned int from = static_cast<unsigned int>(INIT_SEQ_LEN);
  const unsigned int to = prefillPlaneFor(prompt_tokens);
  if (to <= from)
    return false;

  std::fprintf(stderr,
               "[prefill-plane] prompt %u tokens: plane %u -> %u (pack %u, cap "
               "%u); rebuilding the graph and replaying the weight load\n",
               prompt_tokens, from, to, PACK_INIT_SEQ_LEN,
               std::min<unsigned int>(causallm::kPrefillBlockCap,
                                      static_cast<unsigned int>(MAX_SEQ_LEN)));
  std::fflush(stderr);

  /** Release the OLD graph first: its weights and its planned tensors are the
   *  large part, and holding both graphs at once would double the weight
   *  residency for the length of the reload. is_initialized follows, so a throw
   *  below leaves the object visibly uninitialized rather than half-built. */
  const std::string wpath = weight_path_; // load_weight() rewrites the member
  const bool repack = weight_repacked_;
  INIT_SEQ_LEN = to;
  model.reset();
  is_initialized = false;

  initialize();
  load_weight(wpath);
  if (repack)
    repack_weight();
  return true;
}

/**
 * @brief Save model weights to a binary nntrainer model file.
 */
void Transformer::save_weight(const std::string &weight_path) {

  if (!is_initialized) {
    throw std::runtime_error(
      "Transformer model is not initialized. Please call "
      "initialize() before save_weight().");
  }

  try {
    model->save(weight_path, formatFromExtension(weight_path));
  } catch (const std::exception &e) {
    throw std::runtime_error("Failed to save model weights: " +
                             std::string(e.what()));
  }
};

/**
 * @brief Save model weights with optional dtype conversion.
 */
void Transformer::save_weight(
  const std::string &weight_path, ml::train::TensorDim::DataType dtype,
  const std::map<std::string, ml::train::TensorDim::DataType> &layer_dtype_map,
  ml::train::ISA target_isa) {

  if (!is_initialized) {
    throw std::runtime_error(
      "Transformer model is not initialized. Please call "
      "initialize() before save_weight().");
  }

  try {
    model->save(weight_path, formatFromExtension(weight_path), dtype,
                layer_dtype_map, target_isa);

  } catch (const std::exception &e) {
    throw std::runtime_error("Failed to save model weights with dtype: " +
                             std::string(e.what()));
  }
};

/**
 * @brief Repack all QS4CX weights after loading.
 */
void Transformer::repack_weight() {
  if (!is_initialized) {
    throw std::runtime_error(
      "Transformer model is not initialized. Please call "
      "initialize() before repack_weight().");
  }
  /** Recorded (before the engine gate below, which is a per-engine no-op rather
   *  than "the caller did not ask") so growPrefillPlane() replays the caller's
   *  load sequence in the same order it happened. */
  weight_repacked_ = true;

  // [perf/thermal] The KAI rhs-pack below is consumed ONLY by the ARM CPU
  // KleidiAI GEMM. The GPU (v8c) and x86 paths read the plain on-disk QS4CX
  // blob directly (see the comment at the pack() call). On a GPU run the whole
  // loop is therefore redundant CPU work — when it was still single-threaded,
  // a large model pinned one core to a thermal shutdown (Adreno: GPU idle at
  // 0%, one CPU core -> ~104C -> device reboot; 96%+ of CPU was in
  // kai_run_rhs_pack). Skip it on every non-CPU engine — "gpu" (OpenCL) AND
  // "cuda" both consume the plain blob; only the ARM CPU (KAI) run packs.
  // (On ARM64 CUDA the old =="gpu" check let packF16Activation allocate a
  // full unconsumed host copy of every FC weight.)
  if (causallm_engine() != "cpu") {
    ml_logd("repack_weight: skipped on %s engine (consumes plain QS4CX blob)",
            causallm_engine().c_str());
    return;
  }
  // fp16-act graphs dispatch QS4CX FCs through HalfTensor::dot, whose KAI rhs
  // is the fp16-scale layout (packF16Activation) — pack()'s fp32-facade rhs
  // would be dead weight there (an unconsumed full extra copy of every FC in
  // RAM, and ~11% of a 1K-run's CPU in kai_run_rhs_pack). fp32-act graphs
  // dispatch through FloatTensor::dotQs4cx and need pack() as before.
  const bool f16_act =
    MODEL_TENSOR_TYPE.size() >= 5 &&
    MODEL_TENSOR_TYPE.compare(MODEL_TENSOR_TYPE.size() - 5, 5, "-FP16") == 0;
  std::function<void(ml::train::Layer &, nntrainer::RunLayerContext &, void *)>
    fn = [f16_act, this](ml::train::Layer &l,
                         nntrainer::RunLayerContext &context, void *) {
      auto &node = static_cast<nntrainer::LayerNode &>(l);
      try {
        if (!f16_act) {
          // Upstream repack delegation/restriction: each layer decides whether
          // it repacks its own weights. Only the FC layers override Layer::pack
          // today — importantly the QS4CX *embedding* table must NOT be packed
          // (finalize() stores it transposed as (out_dim, in_dim) and the row
          // decoder reads the plain nibbles + per-channel scales).
          node.pack(context);
          // The CPU GEMM reads only the pack from here on (dotQs4cx prefers
          // it for every M), so the plain payload is a dead second copy of
          // every QS4CX FC weight. Release it now rather than on the FC's
          // first forward, or the load ends holding both. Only a self-owned
          // payload (QS4CX heap bypass); a later host read of it fails loudly
          // through the dropped-payload registry. Off for callers that still
          // read the plain payload after the load (nntr_quantize's save).
          if (release_plain_on_repack_ && nntrainer::qs4cxHeapBypassOn() &&
              node.getType() == "fully_connected") {
            for (auto &w : context.getWeights()) {
              nntrainer::Tensor &var = w->getVariableRef();
              if (var.getDataType() != ml::train::TensorDim::DataType::QS4CX ||
                  !var.isPacked())
                continue;
              const void *base = var.getData<uint8_t>();
              const size_t bytes = var.getMemoryBytes();
              if (nntrainer::releaseHostPages(base, bytes))
                nntrainer::markQs4cxPayloadDropped(base, bytes);
            }
          }
          return;
        }
        // fp16-act graphs dispatch QS4CX FCs through HalfTensor::dot, whose KAI
        // rhs is the fp16-scale layout (packF16Activation) — pack()'s
        // fp32-facade rhs would be dead weight there (an unconsumed full extra
        // copy of every FC in RAM, and ~11% of a 1K-run's CPU in
        // kai_run_rhs_pack). There is no Layer::pack hook for the fp16 layout
        // yet, so mirror the restriction by hand: only the layer types that
        // override pack() get repacked.
        const std::string ty = node.getType();
        if (ty != "fully_connected" && ty != "shared_fully_connected")
          return;
        for (auto &w : context.getWeights()) {
          if (w->getVariableRef().getDataType() ==
              ml::train::TensorDim::DataType::QS4CX) {
            // KAI rhs-pack is the ARM (i8mm) CPU GEMM's in-memory
            // derivation; on x86 it is NYI and the x86 CPU GEMM + the GPU v8c
            // path consume the plain on-disk QS4CX (nibbles + fp32 scales)
            // directly. Skip-on-NYI so a single QS4CX weight set loads on every
            // backend (ARM packs, x86/GPU use the plain blob).
            w->getVariableRef().packF16Activation();
          }
        }
      } catch (const std::exception &e) {
        ml_logd("QS4CX pack skipped (engine consumes plain blob): %s",
                e.what());
      }
    };
  try {
    model->forEachLayer(fn, nullptr);
    ml_logd("weights repacked successfully");
  } catch (const std::exception &e) {
    throw std::runtime_error("Failed to repack weights: " +
                             std::string(e.what()));
  }
};

/**
 * @brief Run a transformer model for a prompt.
 */
void Transformer::run(const WSTR prompt, bool do_sample,
                      const WSTR system_prompt, const WSTR tail_prompt,
                      bool log_output) {
  if (!is_initialized) {
    throw std::runtime_error(
      "Transformer model is not initialized. Please call "
      "initialize() before run().");
  }
  ///@note This part should be filled in.
  /// The run action can be defined by the precedent classes.
}

/**
 * @brief Create one decoder block with attention and feed-forward layers.
 */
Tensor Transformer::createTransformerDecoderBlock(const int layer_id,
                                                  Tensor input) {

  // norms are never a LoRA target: freeze them whenever LoRA is active.
  std::vector<std::string> attn_norm_props = {
    withKey("name", "layer" + std::to_string(layer_id) + "_attention_norm"),
    withKey("epsilon", std::to_string(NORM_EPS)), withKey("packed", "false"),
    withKey("engine", layerEngine())};
  if (LORA_RANK > 0 && !TRAIN_NORMS)
    attn_norm_props.push_back(withKey("trainable", "false"));
  LayerHandle attn_norm(createLayer("rms_norm", attn_norm_props));
  Tensor normed = attn_norm(input);

  Tensor att_out = createAttention(layer_id, INIT_SEQ_LEN, NUM_HEADS, HEAD_DIM,
                                   normed, normed, normed);

  LayerHandle decoder_add(createLayer(
    "addition",
    {withKey("name", "layer" + std::to_string(layer_id) + "_decoder_add"),
     withKey("engine", layerEngine())}));
  Tensor residual = decoder_add({input, att_out});

  std::vector<std::string> ffn_norm_props = {
    withKey("name", "layer" + std::to_string(layer_id) + "_ffn_norm"),
    withKey("epsilon", std::to_string(NORM_EPS)), withKey("packed", "false"),
    withKey("engine", layerEngine())};
  if (LORA_RANK > 0 && !TRAIN_NORMS)
    ffn_norm_props.push_back(withKey("trainable", "false"));
  LayerHandle ffn_norm(createLayer("rms_norm", ffn_norm_props));
  Tensor ffn_normed = ffn_norm(residual);

  Tensor ffn_out = createMlp(layer_id, DIM, INTERMEDIATE_SIZE, ffn_normed);

  LayerHandle decoder_output(createLayer(
    "addition",
    {withKey("name", "layer" + std::to_string(layer_id) + "_decoder_output"),
     withKey("engine", layerEngine())}));
  return decoder_output({residual, ffn_out});
}

/**
 * @brief Create external KV-cache placeholder tensors for one layer.
 */
std::pair<Tensor, Tensor>
Transformer::createKVCachePlaceholders(const int layer_id, int n_heads) {
  const unsigned int cache_rows = getKVCacheRows(layer_id);
  const unsigned int kv_width =
    static_cast<unsigned int>(HEAD_DIM * n_heads / GQA_SIZE);
  const std::string cache_shape = std::to_string(BATCH_SIZE) +
                                  ":1:" + std::to_string(cache_rows) + ":" +
                                  std::to_string(kv_width);

  // KV caches MUST be created as "input" layers (not plain Tensors). Plain
  // Tensors shrink the graph's input-layer set, which changes the tensor-pool
  // in-place/flatten behavior so that the first transformer layer's input is no
  // longer a synced dependent of the model input placeholder. On ARM that broke
  // USE_EMBEDDING prefill: the embedding reached input0's output but never
  // layer0_conv_norm (all-zero activations → <pad>). The x86 (#else) path
  // always used input layers and worked; this keeps both paths symmetric,
  // differing only in the external dtype (FP16 on ARM, UINT16 elsewhere).
#ifdef ENABLE_FP16
  const char *cache_dtype = "FP16";
#else
  const char *cache_dtype = "UINT16";
#endif

  LayerHandle cache_k_input(createLayer(
    "input", {withKey("name", "cache_k_l" + std::to_string(layer_id)),
              withKey("input_shape", cache_shape),
              withKey("input_dtype", cache_dtype)}));
  LayerHandle cache_v_input(createLayer(
    "input", {withKey("name", "cache_v_l" + std::to_string(layer_id)),
              withKey("input_shape", cache_shape),
              withKey("input_dtype", cache_dtype)}));

  return {cache_k_input(Tensor()), cache_v_input(Tensor())};
}

/**
 * @brief Wire the attention core's KV cache: 3-input internal-int8 mode when
 * use_int8, else 5-input external-fp16 mode with per-layer placeholders. The
 * placeholder factory is virtual (createKVCachePlaceholders) so a model with a
 * non-uniform cache width overrides it. Consolidates the wiring previously
 * duplicated across the per-model createAttention overrides.
 */
Tensor Transformer::wireAttentionKVCache(const int layer_id, int n_heads,
                                         LayerHandle mha, Tensor q, Tensor k,
                                         Tensor v, bool use_int8) {
  if (use_int8)
    return mha({q, k, v});
  // External KV cache placeholders (per-layer). Their actual storage is owned
  // by the host (KVCacheManager) and bound at runtime by
  // CausalLM::allocateAndBindKVCache.
  auto [cache_k, cache_v] = createKVCachePlaceholders(layer_id, n_heads);
  return mha({q, k, v, cache_k, cache_v});
}

/**
 * @brief Create the default attention subgraph.
 */
Tensor Transformer::createAttention(const int layer_id, int seq_len,
                                    int n_heads, int head_dim, Tensor query,
                                    Tensor key, Tensor value) {

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
      // (GEMM) arm: the host one on the cpu engine, the device attention on a
      // GPU engine whose arm can run; false keeps the per-row host path. See
      // causallm_gemm_attention().
      withKey("use_gemm_attention",
              causallm_gemm_attention(USE_FLASH_ATTENTION) ? "true" : "false"),
    }));

  Tensor a;
  if (FOR_TRAINING) {
    // No KV-cache placeholders for training: mha_core's internal-cache
    // (3-input) construction is what its training-mode forward/backward
    // (trainForwarding()/calcDerivative() in mha_core.cpp) requires.
    a = mha({q, k, v});
  } else {
    a = wireAttentionKVCache(layer_id, n_heads, mha, q, k, v,
                             /*use_int8=*/false);
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

/**
 * @brief Create the default feed-forward subgraph.
 */
Tensor Transformer::createMlp(const int layer_id, int dim, int hidden_dim,
                              Tensor input) {

  // A positional .bin is read in graph order, and the graph is collected by
  // walking back from the output through each layer's inputs in order, so
  // swiglu({gate, up}) loads ffn_gate before ffn_up whatever order the two
  // are created in. The converters therefore write gate_proj -> up_proj ->
  // down_proj (the HF order, as for Gemma's createMlp). A file written up ->
  // gate loads each projection into the other and computes silu(up) * gate:
  // plausible-looking text that is wrong.
  std::vector<std::string> ffn_gate_props = {
    withKey("name", "layer" + std::to_string(layer_id) + "_ffn_gate"),
    withKey("unit", hidden_dim), withKey("disable_bias", "true"),
    withKey("weight_initializer", "ones"), withKey("engine", layerEngine())};
  if (hasLoRA("ffn_gate", layer_id))
    appendLoRAProps(ffn_gate_props);
  else if (LORA_RANK > 0)
    ffn_gate_props.push_back(withKey("trainable", "false"));
  LayerHandle ffn_gate(createLayer("fully_connected", ffn_gate_props));
  Tensor gate = ffn_gate(input);

  std::vector<std::string> ffn_up_props = {
    withKey("name", "layer" + std::to_string(layer_id) + "_ffn_up"),
    withKey("unit", hidden_dim), withKey("disable_bias", "true"),
    withKey("weight_initializer", "ones"), withKey("engine", layerEngine())};
  if (hasLoRA("ffn_up", layer_id))
    appendLoRAProps(ffn_up_props);
  else if (LORA_RANK > 0)
    ffn_up_props.push_back(withKey("trainable", "false"));
  LayerHandle ffn_up(createLayer("fully_connected", ffn_up_props));
  Tensor up = ffn_up(input);

  LayerHandle swiglu(createLayer(
    "swiglu",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_swiglu"),
     withKey("engine", layerEngine())}));
  Tensor act = swiglu({gate, up});

  std::vector<std::string> ffn_down_props = {
    withKey("name", "layer" + std::to_string(layer_id) + "_ffn_down"),
    withKey("unit", dim), withKey("disable_bias", "true"),
    withKey("weight_initializer", "ones"), withKey("engine", layerEngine())};
  if (hasLoRA("ffn_down", layer_id))
    appendLoRAProps(ffn_down_props);
  else if (LORA_RANK > 0)
    ffn_down_props.push_back(withKey("trainable", "false"));
  LayerHandle ffn_down(createLayer("fully_connected", ffn_down_props));
  return ffn_down(act);
}

/**
 * @brief Register custom CausalLM layers in the nntrainer app context.
 */
void Transformer::registerCustomLayers() {
  ///
  const auto &ct_engine = nntrainer::Engine::Global();

  // Registration is best-effort and repeatable: every model construction runs
  // this, and two conditions are normal rather than errors — the build has no
  // such Context ("[Engine] gpu Context is not registered"), or an earlier
  // model already took the key (the cpu AppContext returns the existing key,
  // the cl/cuda contexts throw). Register each factory on its own so one taken
  // key cannot skip the registrations that follow it in the same block; a
  // factory that really fails to register surfaces loudly later as "Key is not
  // found for the object" at graph build.
  auto tryRegister = [&ct_engine](const char *engine, auto factory,
                                  const char *key = "") {
    try {
      ct_engine.registerLayerFactory(engine, factory, key);
    } catch (const std::exception &) {
    }
  };

  // CPU layer classes on the cpu (app) context — through Engine's registration
  // facade, no static_cast to AppContext.
  // swiglu promoted to core app_context.cpp (merged app fork into the
  // backend-neutral nntrainer::SwiGLULayer).
  tryRegister("cpu", nntrainer::createLayer<causallm::RMSNormLayer>);
  tryRegister("cpu", nntrainer::createLayer<causallm::MHACoreLayer>);
  // tie_word_embedding promoted to core — self-registered on cpu/gpu/cuda.
  tryRegister("cpu", nntrainer::createLayer<causallm::EmbeddingLayer>);
  tryRegister("cpu", nntrainer::createLayer<causallm::LmHeadLossLayer>);
  tryRegister("cpu", nntrainer::createLayer<causallm::DecisionHeadLossLayer>);

  // GPU variants: same type strings as the CPU classes but registered on the
  // gpu context so engine=gpu createLayer routes there. The GPU classes use raw
  // getData() pointers + GPU dispatches; they avoid any CPU-only Tensor ops
  // (Tensor::multiply / add_i / dot) that crash on gpu-context tensors. Inert
  // when there is no "gpu" context (CPU-only / NNTR_ENGINE=cpu builds).
  tryRegister("gpu", nntrainer::createLayer<causallm::RMSNormLayerGPU>);
  // GPU-resident per_layer_slice: the same type string as the CPU class,
  // registered here so engine=gpu routes to the GPU kernel instead of taking a
  // host round-trip that would break the activation's residency.
  tryRegister("gpu", nntrainer::createLayer<causallm::PerLayerSliceLayerGPU>);
  // MHACoreLayer on the gpu context enables engine=gpu attention. The same
  // class runs on both backends: forwarding() dispatches the GPU kernels when
  // NNTR_MHA_GPU is on (value-checked; =0 disables) and Q/K/V/cache are
  // SVM-resident, else the CPU NEON
  // path. Additive — a node with no engine= property keeps routing to CPU.
  tryRegister("gpu", nntrainer::createLayer<causallm::MHACoreLayer>);

#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
  // Additive CUDA backend: register the host CausalLM layer classes on the cuda
  // context too. engine=cuda tensors are Unified Memory (host-coherent), so the
  // CPU implementations run correctly on them (no cl_mem); GPU kernels are
  // layered on per-layer later. Runtime-gated to match the engine.cpp context
  // bring-up gate: on a non-cuda run there is no "cuda" context and the
  // registration would just throw per layer.
  const char *eager_env = std::getenv("NNTR_CUDA_EAGER_CTX");
  if (causallm_engine() == "cuda" ||
      (eager_env != nullptr && eager_env[0] != '0')) {
    // CUDA RMSNorm (device kernel, FP32 sum-of-squares) under the "rms_norm"
    // key this application's graph asks for. The cuda context self-registers
    // the same class under the backend-neutral "rmsnorm" type string, which is
    // a DIFFERENT key, so without this line every rms_norm node on an
    // engine=cuda graph resolved to the host causallm::RMSNormLayer instead —
    // a host loop reading and writing UVM activations inside the device chain,
    // which is both wrong (host reads racing the un-synced device stream) and
    // catastrophically slow (page migration per norm, per token).
    tryRegister("cuda", nntrainer::createLayer<nntrainer::CudaRMSNormLayer>,
                "rms_norm");
    tryRegister("cuda", nntrainer::createLayer<causallm::MHACoreLayer>);
    // tie_word_embedding promoted to core cuda_context.cpp.
    tryRegister("cuda", nntrainer::createLayer<causallm::EmbeddingLayer>);
    // gemma4 PLE host impl (the GPU variant is OpenCL-only and would mis-run on
    // UVM). Without it on the cuda context the type falls back to the wrong
    // factory. scalar_multiply is now self-registered in core cuda_context.cpp.
    tryRegister("cuda", nntrainer::createLayer<causallm::PerLayerSliceLayer>);
  }
#endif
}

} // namespace causallm
