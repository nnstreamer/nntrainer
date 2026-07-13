// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Eunju Yang <ej.yang@samsung.com>
 *
 * @file   transformer.cpp
 * @date   10 July 2025
 * @see    https://github.com/nntrainer/nntrainer
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This file defines Transformer's basic actions
 */

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <mutex>

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

#include <embedding_layer.h>
#include <mha_core.h>
#include <neuralnet.h>
#include <per_layer_slice_gpu.h>
#include <qs4cx_tensor.h>
#include <residency_policy.h>
#include <rms_norm.h>
#include <rms_norm_gpu.h>

namespace causallm {

/**
 * @brief Get model format from weight file extension.
 */


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

  // "causallm" is accepted as a legacy alias so model configs written before
  // the causallm rebrand keep loading unchanged.
  static const std::unordered_map<std::string, ModelType> model_type_map = {
    {"model", ModelType::MODEL},
    {"causallm", ModelType::CAUSALLM},
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

  return;
};

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

/**
 * @brief Construct the default decoder-only transformer graph.
 */
std::pair<Tensor, Tensor> Transformer::constructModel() {

  // input
  Tensor x =
    Tensor({1, 1, 1, static_cast<unsigned int>(INIT_SEQ_LEN)}, "input0");

  // embedding
  const std::string embedding_type =
    TIE_WORD_EMBEDDINGS ? "tie_word_embeddings" : "embedding_layer";

  NNTR_THROW_IF(TIE_WORD_EMBEDDINGS && !EMBEDDING_FILE_NAME.empty(),
                std::invalid_argument)
    << "embedding_file_name requires untied embedding_layer";
  LayerHandle embedding(createLayer(
    embedding_type,
    buildEmbeddingLayerProperties("embedding0", NUM_VOCAB, DIM, EMBEDDING_DTYPE,
                                  EMBEDDING_SCALE, EMBEDDING_FILE_NAME)));
  Tensor h = embedding(x);

  // transformer decoder blocks
  for (int i = 0; i < NUM_LAYERS; ++i) {
    h = createTransformerDecoderBlock(i, h);
  }

  // final rms_norm. NOTE: stays on CausalLM's "rms_norm" type (the custom
  // RMSNormLayer on cpu, RMSNormLayerGPU on gpu, CudaRMSNormLayer on cuda --
  // see registerCustomLayers) so the fused-rmsq + v8c FC consumer chain works.
  // The nntrainer GPU RMSNormLayerCl uses type "rmsnorm" (different) and has a
  // separate reduction-order drift issue.
  LayerHandle out_norm(
    createLayer("rms_norm", {withKey("name", "output_norm"),
                             withKey("epsilon", std::to_string(NORM_EPS)),
                             withKey("packed", "false"),
                             withKey("engine", causallm_engine())}));
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

  LayerHandle attn_norm(createLayer(
    "rms_norm",
    {withKey("name", "layer" + std::to_string(layer_id) + "_attention_norm"),
     withKey("epsilon", std::to_string(NORM_EPS)), withKey("packed", "false"),
     withKey("engine", causallm_engine())}));
  Tensor normed = attn_norm(input);

  Tensor att_out = createAttention(layer_id, INIT_SEQ_LEN, NUM_HEADS, HEAD_DIM,
                                   normed, normed, normed);

  LayerHandle decoder_add(createLayer(
    "addition",
    {withKey("name", "layer" + std::to_string(layer_id) + "_decoder_add"),
     withKey("engine", causallm_engine())}));
  Tensor residual = decoder_add({input, att_out});

  LayerHandle ffn_norm(createLayer(
    "rms_norm",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_norm"),
     withKey("epsilon", std::to_string(NORM_EPS)), withKey("packed", "false"),
     withKey("engine", causallm_engine())}));
  Tensor ffn_normed = ffn_norm(residual);

  Tensor ffn_out = createMlp(layer_id, DIM, INTERMEDIATE_SIZE, ffn_normed);

  LayerHandle decoder_output(createLayer(
    "addition",
    {withKey("name", "layer" + std::to_string(layer_id) + "_decoder_output"),
     withKey("engine", causallm_engine())}));
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
  LayerHandle wq(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_wq"),
     withKey("unit", head_dim * n_heads), withKey("disable_bias", "true"),
     withKey("weight_initializer", "ones"),
     withKey("engine", causallm_engine())}));
  Tensor q = wq(query);

  // K layer
  LayerHandle wk(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_wk"),
     withKey("unit", head_dim * n_heads / GQA_SIZE),
     withKey("disable_bias", "true"), withKey("weight_initializer", "ones"),
     withKey("engine", causallm_engine())}));
  Tensor k = wk(key);

  // V layer
  LayerHandle wv(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_wv"),
     withKey("unit", head_dim * n_heads / GQA_SIZE),
     withKey("disable_bias", "true"), withKey("weight_initializer", "ones"),
     withKey("engine", causallm_engine())}));
  Tensor v = wv(value);

  // Attention core layer
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
  Tensor a = wireAttentionKVCache(layer_id, n_heads, mha, q, k, v,
                                  /*use_int8=*/false);

  // O layer
  LayerHandle wo(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_attention_out"),
     withKey("unit", DIM), withKey("disable_bias", "true"),
     withKey("weight_initializer", "ones"),
     withKey("engine", causallm_engine())}));
  return wo(a);
}

/**
 * @brief Create the default feed-forward subgraph.
 */
Tensor Transformer::createMlp(const int layer_id, int dim, int hidden_dim,
                              Tensor input) {

  // WARNING: legacy .bin compatibility. Existing files store the FFN weights
  // in up, gate, down order, so up is created first and the swiglu slot map
  // {1, 0} makes input[0] = gate, input[1] = up. A model whose file uses
  // gate, up, down must override createMlp() instead of changing this order.
  LayerHandle ffn_up(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_up"),
     withKey("unit", hidden_dim), withKey("disable_bias", "true"),
     withKey("weight_initializer", "ones"),
     withKey("engine", causallm_engine())}));
  Tensor up = ffn_up(input);

  LayerHandle ffn_gate(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_gate"),
     withKey("unit", hidden_dim), withKey("disable_bias", "true"),
     withKey("weight_initializer", "ones"),
     withKey("engine", causallm_engine())}));
  Tensor gate = ffn_gate(input);

  LayerHandle swiglu(createLayer(
    "swiglu",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_swiglu"),
     withKey("engine", causallm_engine())}));
  Tensor act = swiglu({up, gate}, {1, 0});

  LayerHandle ffn_down(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_down"),
     withKey("unit", dim), withKey("disable_bias", "true"),
     withKey("weight_initializer", "ones"),
     withKey("engine", causallm_engine())}));
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
