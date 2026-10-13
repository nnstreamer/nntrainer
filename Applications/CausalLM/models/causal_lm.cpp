// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Jijoong Moon <jijoong.moon@samsung.com>
 * Copyright (C) 2025 Seungback Hong <sb92.hong@samsung.com>
 * Copyright (C) 2025 Hyeonseok Lee <hs89.lee@samsung.com>
 * Copyright (C) 2025 Eunju Yang <ej.yang@samsung.com>
 *
 * @file   causal_lm.cpp
 * @date   10 July 2025
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This file defines CausalLM's basic actions
 * @note   This causal_lm.h constructs a class for Transformer-based Causal
 * Language Model (CausalLM). It aims to support AutoModelForCausalLM with
 * nntrainer. It supports the following models:
 *          - Llama
 */

#include <algorithm>
#include <app_context.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <engine.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <common.h>
#include <compute_ops.h>
#include <context.h>
#include <context_data.h>
#include <htp_decode_hook.h>
#include <layer_context.h>
#include <lm_head.h>
#include <mha_core.h>
#include <neuralnet.h>
#include <nntrainer_error.h>
#include <tensor.h>
#include <tie_word_embedding.h>

#include <causal_lm.h>
#include <llm_util.hpp>
#include <utf8_stream_util.h>

#include "api/streamer.h"

namespace causallm {

CausalLM::CausalLM(json &cfg, json &generation_cfg, json &nntr_cfg) :
  Transformer(cfg, generation_cfg, nntr_cfg, ModelType::CAUSALLM) {
  setupParameters(cfg, generation_cfg, nntr_cfg);
}

void CausalLM::prepareForRun() {
  stop_requested_.store(false, std::memory_order_release);
  stop_prepared_for_run_.store(true, std::memory_order_release);
}

void CausalLM::prepareStopRequestForRun() {
  if (!stop_prepared_for_run_.exchange(false, std::memory_order_acq_rel)) {
    stop_requested_.store(false, std::memory_order_release);
  }
}

void CausalLM::setLogitsProcessor(LogitsProcessor *processor) {
  logits_processor = processor;
}

void CausalLM::resetLogitsProcessor() {
  if (logits_processor != nullptr)
    logits_processor->reset();
}

void CausalLM::setupParameters(json &cfg, json &generation_cfg,
                               json &nntr_cfg) {
  // Initialize output list
  for (unsigned int i = 0; i < BATCH_SIZE; ++i)
    output_list.push_back("");

  // allocate memory for the internal buffer
  ids_history = (unsigned int *)malloc(static_cast<size_t>(BATCH_SIZE) *
                                       MAX_SEQ_LEN * sizeof(unsigned int));

  BAD_WORD_IDS = nntr_cfg["bad_word_ids"].get<std::vector<unsigned int>>();
  NUM_BADWORDS = BAD_WORD_IDS.size();

  LMHEAD_DTYPE = nntr_cfg.contains("lmhead_dtype")
                   ? nntr_cfg["lmhead_dtype"]
                   : nntr_cfg["embedding_dtype"];

  SKIP_PREFILL = nntr_cfg.contains("skip_prefill")
                   ? nntr_cfg["skip_prefill"].get<bool>()
                   : false;

  USE_KVCACHE = false;
  PRE_COMPUTED_CACHE_PATH = "";
  SYS_PROMP_LEN = 0;

  if (nntr_cfg.contains("system_prompt") &&
      nntr_cfg["system_prompt"].contains("kvcache")) {
    USE_KVCACHE = true;
    PRE_COMPUTED_CACHE_PATH =
      nntr_cfg["system_prompt"]["kvcache"]["pre_computed_cache_path"];
    if (nntr_cfg["system_prompt"]["kvcache"].contains("sys_prompt_token_size"))
      SYS_PROMP_LEN =
        nntr_cfg["system_prompt"]["kvcache"]["sys_prompt_token_size"]
          .get<unsigned int>();
  }

  if (generation_cfg["eos_token_id"].is_array()) {
    EOS_TOKEN_ID =
      generation_cfg["eos_token_id"].empty()
        ? cfg["eos_token_id"].get<std::vector<unsigned int>>()
        : generation_cfg["eos_token_id"].get<std::vector<unsigned int>>();
  } else {
    EOS_TOKEN_ID.clear();
    EOS_TOKEN_ID.push_back(generation_cfg["eos_token_id"].get<unsigned int>());
  }
  BOS_TOKEN_ID = generation_cfg["bos_token_id"].empty()
                   ? cfg["bos_token_id"].get<unsigned int>()
                   : generation_cfg["bos_token_id"].get<unsigned int>();
  TOP_K = generation_cfg.contains("top_k")
            ? generation_cfg["top_k"].get<unsigned int>()
            : 20;
  TOP_P = generation_cfg.contains("top_p")
            ? generation_cfg["top_p"].get<float>()
            : 0.95;
  TEMPERATURE = generation_cfg.contains("temperature")
                  ? generation_cfg["temperature"].get<float>()
                  : 0.7;
  global_token_len = 0;
}

// With the attention layers on an accelerator, put the cache in memory
// that accelerator reads in place (ComputeOps::alloc_shared; rpcmem on
// the HTP), so no call copies the used cache range. A backend without
// shared memory returns nullptr from alloc_shared and the manager
// keeps the heap; a missing or unusable engine is reported and the
// model runs as before. Every allocateAndBindKVCache override calls
// this before kv_cache.allocate(): Gemma4's did not, and FastRPC copied
// its K/V every attention call (doc 57 section 9.10).
void CausalLM::installKVCacheSharedAllocator() {
  if (ATTENTION_ENGINE.empty())
    return;
  try {
    auto *ctx =
      nntrainer::Engine::Global().getRegisteredContext(ATTENTION_ENGINE);
    auto data = ctx ? ctx->getContextData() : nullptr;
    nntrainer::ComputeOps *ops = data ? data->getComputeOps() : nullptr;
    if (ops) {
      kv_cache.setSharedAllocator(
        [ops](size_t bytes) { return ops->alloc_shared(bytes); },
        [ops](void *block) { ops->free_shared(block); });
    }
  } catch (const std::exception &e) {
    std::cerr << "KV cache stays in host memory: attention_engine '"
              << ATTENTION_ENGINE << "' unavailable (" << e.what() << ")"
              << std::endl;
  }
}

void CausalLM::allocateAndBindKVCache() {
  if (!kv_cache.isAllocated()) {
    // dtype matches mha_core's cache placeholders so external cache storage
    // is interpreted consistently across platforms.
#ifdef ENABLE_FP16
    const auto cache_dtype = ml::train::TensorDim::DataType::FP16;
#else
    const auto cache_dtype = ml::train::TensorDim::DataType::UINT16;
#endif

    const unsigned int max_timestep = static_cast<unsigned int>(MAX_SEQ_LEN);

    installKVCacheSharedAllocator();
    kv_cache.allocate(static_cast<unsigned int>(NUM_LAYERS), BATCH_SIZE,
                      max_timestep,
                      static_cast<unsigned int>(NUM_KEY_VALUE_HEADS),
                      static_cast<unsigned int>(HEAD_DIM), cache_dtype);
    kv_cache_bound = false;
  }

  if (kv_cache_bound)
    return;

  // Bind each (layer, K|V) buffer into the corresponding input layer
  // declared by Transformer::createKVCachePlaceholders(). The names here
  // must match what createKVCachePlaceholders() registers with the model.
  // We look up each placeholder by name and point it at our cache slab;
  // this is the same wiring Model::setExternalTensors used to do, just
  // without going through that API.
  for (int i = 0; i < NUM_LAYERS; ++i) {
    auto &kc = kv_cache.getKeyCache(i);
    auto &vc = kv_cache.getValueCache(i);

    auto find_cache_placeholder = [this](const std::string &base_name) {
      for (const auto &suffix : {":0", ":input0", ":out0", ""}) {
        auto *tensor = model->getTensor(base_name + suffix);
        if (tensor != nullptr)
          return tensor;
      }
      return static_cast<nntrainer::Tensor *>(nullptr);
    };

    auto *kp =
      model->getTensor("layer" + std::to_string(i) + "_attention:input3");
    auto *vp =
      model->getTensor("layer" + std::to_string(i) + "_attention:input4");
    if (kp == nullptr)
      kp = find_cache_placeholder("cache_k_l" + std::to_string(i));
    if (vp == nullptr)
      vp = find_cache_placeholder("cache_v_l" + std::to_string(i));
    if (kp == nullptr && vp == nullptr) {
      /// This layer has no attention sub-graph (e.g., a conv-only block in a
      /// hybrid architecture like LFM2). Skip KV-cache binding for it.
      continue;
    }
    NNTR_THROW_IF(kp == nullptr || vp == nullptr, std::runtime_error)
      << "allocateAndBindKVCache: cache_k_l" << i << " / cache_v_l" << i
      << " partially found in compiled graph (one placeholder exists but "
         "the other does not)";
    NNTR_THROW_IF(kp->getDataType() != kc.getDataType() ||
                    vp->getDataType() != vc.getDataType(),
                  std::runtime_error)
      << "allocateAndBindKVCache: cache placeholder dtype mismatch for layer "
      << i;

    kp->setData(kc.getMemoryData(), kc.getOffset(), false);
    vp->setData(vc.getMemoryData(), vc.getOffset(), false);
  }

  kv_cache_bound = true;
}

std::vector<float *> CausalLM::buildInferenceInputs(float *input_sample) {
  struct CacheInput {
    std::string name;
    float *data;
    unsigned int width;
  };
  std::vector<CacheInput> cache_inputs;
  cache_inputs.reserve(static_cast<size_t>(NUM_LAYERS) * 2);
  for (int i = 0; i < NUM_LAYERS; ++i) {
    auto &kc = kv_cache.getKeyCache(i);
    auto &vc = kv_cache.getValueCache(i);
    cache_inputs.push_back({"cache_k_l" + std::to_string(i),
                            reinterpret_cast<float *>(kc.getData()),
                            static_cast<unsigned int>(kc.width())});
    cache_inputs.push_back({"cache_v_l" + std::to_string(i),
                            reinterpret_cast<float *>(vc.getData()),
                            static_cast<unsigned int>(vc.width())});
  }
  std::sort(
    cache_inputs.begin(), cache_inputs.end(),
    [](const auto &lhs, const auto &rhs) { return lhs.name < rhs.name; });

  const auto dims = model->getInputDimension();
  std::vector<bool> used(cache_inputs.size(), false);
  std::vector<float *> inputs;
  inputs.reserve(dims.size());
  inputs.push_back(input_sample);
  for (size_t idx = 1; idx < dims.size(); ++idx) {
    size_t j = 0;
    while (j < cache_inputs.size() &&
           (used[j] || cache_inputs[j].width != dims[idx].width()))
      ++j;
    NNTR_THROW_IF(j == cache_inputs.size(), std::runtime_error)
      << "CausalLM: no KV cache buffer of width " << dims[idx].width()
      << " for model input " << idx;
    used[j] = true;
    inputs.push_back(cache_inputs[j].data);
  }
  return inputs;
}

void CausalLM::setKVCachePosition(unsigned int pos) {
  kv_cache.setPosition(pos);
  std::function<void(ml::train::Layer &, nntrainer::RunLayerContext &, void *)>
    fn = [pos](ml::train::Layer &l, nntrainer::RunLayerContext &, void *) {
      if (l.getType() == causallm::MHACoreLayer::type)
        l.setProperty({"cache_index=" + std::to_string(pos)});
    };
  model->forEachLayer(fn, nullptr);
}

void CausalLM::compactKVCache(unsigned int pos, bool force) {
  if (!kv_cache.compact(pos) && !force)
    return;
  // ponytail: the n-th mha_core layer in graph order is attention layer n
  // (every model here builds them in that order); a model that shares or
  // reorders them needs the layer's own index instead.
  unsigned int n = 0;
  std::function<void(ml::train::Layer &, nntrainer::RunLayerContext &, void *)>
    fn = [&](ml::train::Layer &l, nntrainer::RunLayerContext &, void *) {
      if (l.getType() == causallm::MHACoreLayer::type &&
          n < kv_cache.getNumLayers())
        l.setProperty({"cache_base=" + std::to_string(kv_cache.getBase(n++))});
    };
  model->forEachLayer(fn, nullptr);
}

void CausalLM::advanceKVCachePosition(unsigned int step_size) {
  // mha_core advances its own cache_index inside forwarding(), so the host
  // only has to keep KVCacheManager's tracked position in sync.
  kv_cache.advance(step_size);
}

std::pair<Tensor, Tensor> CausalLM::constructModel() {

  // base transformer (input, output_norm)
  auto [x, h] = Transformer::constructModel();

  const std::string lmhead_type =
    TIE_WORD_EMBEDDINGS ? "tie_word_embeddings" : "lm_head";

  std::vector<std::string> lmhead_prop = {
    withKey("name", "output_of_causallm"),
    withKey("unit", NUM_VOCAB),
    withKey("disable_bias", "true"),
    withKey("weight_dtype", LMHEAD_DTYPE),
  };

  if (TIE_WORD_EMBEDDINGS)
    lmhead_prop.emplace_back(withKey("shared_from", "embedding0"));

  LayerHandle lmhead(createLayer(lmhead_type, lmhead_prop));
  Tensor y = lmhead(h);

  return {x, y};
}

void CausalLM::registerOutputs(
  std::unique_ptr<tokenizers::Tokenizer> &tokenizer,
  std::vector<unsigned int> ids, unsigned int pos,
  const std::vector<bool> &eos_list, bool log_output) {

  static const std::vector<char> puncts{',', '!', ':', ';', '?'};
  for (size_t b = 0; b < ids.size(); ++b) {
    if (!eos_list[b]) {
      pending_ids_.push_back(static_cast<int>(ids[b]));
      // prompt + num_to_generate == max_seq_len puts the last token at
      // pos == MAX_SEQ_LEN, one past the row; it is never fed back, so it
      // stays out of the history instead of writing into the next block
      if (pos < MAX_SEQ_LEN)
        ids_history[b * MAX_SEQ_LEN + pos] = ids[b];
      std::string decoded_str = tokenizer->Decode(pending_ids_);

      if (decoded_str.empty()) {
        continue;
      }

      if (std::find(puncts.begin(), puncts.end(), decoded_str.back()) !=
          puncts.end()) {
        // last symbol is a punctuation, hold on
      } else if (utf8stream::shouldHold(decoded_str, pending_ids_.size())) {
      } else {
        if (log_output && streamer_ == nullptr) {
          std::cout << decoded_str;
          std::cout.flush();
        }
        output_list[b].append(decoded_str);
        if (streamer_ != nullptr &&
            streamer_put(streamer_, decoded_str.c_str()) != 0) {
          requestStop();
        }
        pending_ids_.clear();
      }
    }
  }
}

void CausalLM::save_kvcache(std::string path, int to_) {
  if (!kv_cache.isAllocated()) {
    throw std::runtime_error(
      "save_kvcache called before allocateAndBindKVCache()");
  }
  kv_cache.save(path, static_cast<unsigned int>(to_));
  // the int8 attention's scales of this prefill, for the run that loads it
  std::ofstream q2(path + ".q2", std::ios::binary);
  causallm::MHACoreLayer::saveQ2Calibration(q2);
  if (!q2)
    throw std::runtime_error("save_kvcache: cannot write " + path + ".q2");
}

void CausalLM::load_kvcache(std::string path, int to_) {
  if (!kv_cache.isAllocated()) {
    allocateAndBindKVCache();
  }
  kv_cache.load(path, static_cast<unsigned int>(to_));
  // absent (a dump of an older build, a CPU run): the layers calibrate on
  // their first step as before
  if (std::ifstream q2(path + ".q2", std::ios::binary); q2)
    causallm::MHACoreLayer::loadQ2Calibration(q2);
  // a windowed layer's rows start at its base now
  compactKVCache(static_cast<unsigned int>(to_), true);
  // mha_core layers each track their own cache_index; sync them all to the
  // newly-loaded position so the next forwarding() writes at the right slot.
  setKVCachePosition(static_cast<unsigned int>(to_));
}

std::vector<unsigned int> CausalLM::generate(float *logits, bool do_sample,
                                             float repetition_penalty,
                                             unsigned int *input_ids,
                                             unsigned int NUM_INPUT_IDS) {

  std::vector<unsigned int> outputs;
  for (unsigned int iteration = 0; iteration < BATCH_SIZE; ++iteration) {

    // apply repetition penalty
    if (repetition_penalty != 1 && input_ids != nullptr && NUM_INPUT_IDS != 0) {
      applyRepetitionPenalty(logits, input_ids, NUM_INPUT_IDS,
                             repetition_penalty);
    }

    // apply bad words penalty
    if (BAD_WORD_IDS.size() != 0 && NUM_BADWORDS != 0) {
      applyBadWordsPenalty(logits, BAD_WORD_IDS.data(), NUM_BADWORDS);
    }

    if (logits_processor != nullptr)
      logits_processor->process(logits, NUM_VOCAB, iteration);

    unsigned int output_id;

    // return argmax if do_sample is false; [#132 Part B E3] the NPU's own
    // argmax when only the id came back (run() asked for that)
    if (do_sample == false) {
      if (!htpDecodeTokenId(&output_id))
        output_id =
          std::distance(logits, std::max_element(logits, logits + NUM_VOCAB));
    } else {
      // apply temperature & top-k & top-p and sample with original logits
      // unchanged
      output_id = applyTKP(logits, NUM_VOCAB, TEMPERATURE, TOP_K, TOP_P, rng);
    }

    outputs.push_back(output_id);

    if (logits_processor != nullptr)
      logits_processor->acceptToken(output_id, iteration);

    // set batch offset
    logits = logits + NUM_VOCAB;
    if (input_ids != nullptr)
      input_ids = input_ids + MAX_SEQ_LEN;
  }

  return outputs;
};

void CausalLM::registerCustomLayers() {
  Transformer::registerCustomLayers();
  const auto &ct_engine = nntrainer::Engine::Global();
  const auto app_context =
    static_cast<nntrainer::AppContext *>(ct_engine.getRegisteredContext("cpu"));
  try {
    app_context->registerFactory(nntrainer::createLayer<causallm::LmHeadLayer>);
  } catch (std::invalid_argument &e) {
    std::cerr << "failed to register factory, reason: " << e.what()
              << std::endl;
  }
}

void CausalLM::run(const WSTR prompt, bool do_sample, const WSTR system_prompt,
                   const WSTR tail_prompt, bool log_output) {

  auto start_total = std::chrono::high_resolution_clock::now();
  if (!is_initialized) {
    throw std::runtime_error("CausalLM model is not initialized. Please call "
                             "initialize() before run().");
  }
  // Transformer leaves tokenizer null when nntr_config.json has no
  // "tokenizer_file" (or skip_tokenizer is set); run() cannot work without it.
  if (!tokenizer) {
    throw std::runtime_error(
      "CausalLM::run: no tokenizer loaded. nntr_config.json needs a valid "
      "\"tokenizer_file\" and skip_tokenizer must not be set.");
  }

  struct StreamerEndGuard {
    BaseStreamer *streamer;
    ~StreamerEndGuard() { streamer_end(streamer); }
  } streamer_end_guard{streamer_};

  // Allocate the host-owned KV cache and bind it to mha_core's external cache
  // input slots. Idempotent: only the first call does work; subsequent runs
  // reuse the same buffers and continue from the computed absolute token
  // position below.
  allocateAndBindKVCache();

  has_run_ = false;
  prepareStopRequestForRun();

  output_list.clear();
  for (unsigned int b = 0; b < BATCH_SIZE; ++b) {
    output_list.push_back("");
  }

  if (MAX_SEQ_LEN < INIT_SEQ_LEN) {
    throw std::invalid_argument(
      "MAX_SEQ_LEN must be greater than or equal to INIT_SEQ_LEN");
  }

  /**
   * Variables for Log
   */
  unsigned int generation_cnt = 0;
  int64_t total_generation_duration = 0;

  // Record generated IDs without requesting logits or forcing a continuation.
  // This lets performance runs verify the normal DSP token-id path as well.
  const char *generated_ids_path = std::getenv("NNTR_GENERATED_IDS");
  std::vector<unsigned int> generated_ids;

  // NNTR_PPL_DECODE=<ids file> (#134): decode-side teacher-forced NLL of
  // the logits each greedy pick is taken from, after the prompt's prefill.
  // A missing file makes a normal greedy run that scores its own path and
  // writes it there (prefill token + one id per step, whitespace-separated);
  // an existing file forces that continuation. Off by default, never set
  // in a tok/s run.
  // ponytail: ids file only (no text), batch 1, greedy, no SKIP_PREFILL,
  // one run() per process (a second run() finds the first one's file and
  // is forced on it); upgrade = tokenize a .txt through the loaded
  // tokenizer, and decide self/forced once per process.
  const char *ppl_dec_env = std::getenv("NNTR_PPL_DECODE");
  bool ppl_dec = false, ppl_forced = false;
  std::vector<unsigned int> ppl_cont; // forced: the targets; self: the path
  double ppl_dec_nll = 0.0;
  unsigned int ppl_dec_n = 0, ppl_dec_top1 = 0;
  if (ppl_dec_env != nullptr) {
    if (BATCH_SIZE != 1 || SKIP_PREFILL || logits_processor != nullptr ||
        do_sample) {
      std::cerr << "[PPL] decode unsupported (batch > 1, skip_prefill, a "
                   "logits processor or do_sample); running normally"
                << std::endl;
    } else {
      ppl_dec = true;
      std::ifstream f(ppl_dec_env);
      if (f) {
        ppl_forced = true;
        long long id;
        while (f >> id) {
          if (id < 0 || id >= static_cast<long long>(NUM_VOCAB))
            throw std::invalid_argument(
              std::string("NNTR_PPL_DECODE: id out of vocab range in ") +
              ppl_dec_env);
          ppl_cont.push_back(static_cast<unsigned int>(id));
        }
        if (!f.eof() || ppl_cont.empty())
          throw std::invalid_argument(
            std::string("NNTR_PPL_DECODE: not a non-empty list of ids: ") +
            ppl_dec_env);
      }
    }
  }
  // [#194] NNTR_PPL_DECODE_ALTS=<id,...> (at most 8, with NNTR_PPL_DECODE):
  // each step line also prints those ids' logits (the 40-question multiple
  // choice of plan 194 P3 reads the four answer letters); off by default,
  // never set in a tok/s run.
  std::vector<unsigned int> ppl_alts;
  if (const char *a = std::getenv("NNTR_PPL_DECODE_ALTS")) {
    std::stringstream ss(a);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
      char *end = nullptr;
      const unsigned long id = std::strtoul(tok.c_str(), &end, 10);
      if (tok.empty() || *end != '\0' || id >= NUM_VOCAB ||
          ppl_alts.size() == 8)
        throw std::invalid_argument(
          std::string("NNTR_PPL_DECODE_ALTS: want at most 8 comma-separated "
                      "ids below the vocab size: ") +
          a);
      ppl_alts.push_back(static_cast<unsigned int>(id));
    }
  }

  /**
   * INPUT PREPARATION
   */
  std::vector<float *> input;
  std::vector<float *> label;

  /**
   * SAVE_KVCACHE ?
   *  if USE_KVCACHE && system_prompt is given && but the
   * PRE_COMPUTED_CACHE_PATH does not exist
   */
  SAVE_KVCACHE = (USE_KVCACHE && system_prompt != "" &&
                  !std::filesystem::exists(PRE_COMPUTED_CACHE_PATH));

  // print input text
  if (log_output)
    std::cout << system_prompt << prompt << tail_prompt << std::endl;

  // actual prompt to be used in computation
  std::string prompt_;

  if (USE_KVCACHE) {
    prompt_ = SAVE_KVCACHE ? system_prompt : (prompt + tail_prompt);
  } else {
    prompt_ = system_prompt + prompt + tail_prompt;
  }

  if (USE_KVCACHE && !SAVE_KVCACHE && SYS_PROMP_LEN == 0)
    SYS_PROMP_LEN = tokenizer->Encode(system_prompt).size();

  auto _input = tokenizer->Encode(prompt_);
  ///@note insert bos token at the beginning of the input
  // _input.insert(_input.begin(), BOS_TOKEN_ID);

  // | <------------------- MAX_SEQ_LEN -------------------> |
  //                       ||             ||
  // |<-- System prompt -->||<-- input -->||<-- generate -->|

  std::vector<int64_t> init_input;
  unsigned int _len = _input.size();
  unsigned int num_allow_str = MAX_SEQ_LEN - NUM_TO_GENERATE;
  unsigned int text_len = _len;

  if (_len > num_allow_str) {
    text_len = num_allow_str;
    // Truncation drops tokens from the tail of the prompt, which is where
    // instructions in "summarize this document"-style prompts live: a
    // silently truncated prompt can make the model continue the body
    // instead of following a dropped trailing instruction. Always warn
    // with the exact counts.
    std::cerr << "[CausalLM] WARNING: prompt (" << _len
              << " tokens) exceeds the max allowed prefill length ("
              << num_allow_str
              << " = max_seq_len - num_to_generate); "
                 "truncating "
              << (_len - num_allow_str) << " tail tokens." << std::endl;
  }

  // feed only available length
  // if _input is allowed, it feeds all of the _input
  // otherwise, feeds only a part of _input
  for (unsigned int i = 0; i < text_len; ++i)
    init_input.push_back(_input[i]);

  ///@todo currently, the whole sequence may not be fed into the model
  /// This should be handled later.
  _input.clear();

  unsigned int init_len = init_input.size();
  float *input_sample =
    (float *)malloc(sizeof(float) * BATCH_SIZE * MAX_SEQ_LEN);
  std::vector<bool> eos_list(BATCH_SIZE, false);

  unsigned int input_len = init_len;

  for (unsigned int b = 0; b < BATCH_SIZE; ++b) {
    for (unsigned int i = 0; i < input_len; ++i) {
      input_sample[static_cast<size_t>(b) * MAX_SEQ_LEN + i] =
        static_cast<float>(init_input[i]);
      ids_history[static_cast<size_t>(b) * MAX_SEQ_LEN + i] = init_input[i];
    }
  }

  /**
   * PREFILL
   */
  std::vector<int64_t> token_ids;
  input.push_back(input_sample);
  input = buildInferenceInputs(input_sample);

  ///@note contains possible bug
  // std::vector<ml::train::TensorDim> input_dims;
  // ml::train::TensorDim input_dim(1, 1, input_len, DIM);
  // input_dims.push_back(input_dim);
  // model->resetInputDimension(input_dims);

  auto start_prefill = std::chrono::high_resolution_clock::now();

  // NNTR_PPL: the prompt's teacher-forced NLL at prefill (doc 51 section
  // 2.14), the accuracy gate. Tied lm_head only.
  const bool ppl_on = std::getenv("NNTR_PPL") != nullptr;
  if (ppl_on) {
    std::vector<unsigned int> ids(init_input.begin(), init_input.end());
    TieWordEmbedding::setPplTargets(ids);
  }

  std::vector<float *> output;

  // The graph (and the HTP prefill workspace) is sized by INIT_SEQ_LEN rows,
  // so a prompt longer than that runs as INIT_SEQ_LEN-row chunks: each chunk's
  // tokens go to the front of the input buffer and attend to the rows the
  // earlier chunks left in the KV cache. Only the last chunk's logits are kept.
  auto prefill = [&](unsigned int from, unsigned int n) {
    std::vector<float *> out;
    for (unsigned int done = 0; done < n;) {
      const unsigned int step = std::min(n - done, INIT_SEQ_LEN);
      if (done > 0)
        for (unsigned int b = 0; b < BATCH_SIZE; ++b)
          for (unsigned int i = 0; i < step; ++i)
            input_sample[static_cast<size_t>(b) * MAX_SEQ_LEN + i] =
              static_cast<float>(init_input[done + i]);
      for (auto &o : out)
        delete[] o;
      // a windowed (sliding) layer keeps its window in front of the chunk
      compactKVCache(from + done);
      out = model->incremental_inference(
        BATCH_SIZE, input, label, step, from + done, from + done + step, false);
      done += step;
    }
    return out;
  };

  if (SAVE_KVCACHE) {
    //@note This is for the save the kv cache. precomputed kv cache should be
    // always located at the begining of the prompt.
    // Therefore, it start from 0. and system prompt should be saved in the
    // init_input, so that we can compute system prompt size properly
    //
    // The structure of this precomputed K,V Cache is :
    //
    //  //<-- System Prompt -->/<-- Input Tokens -->/<-- Tail prompt --> //
    //  //< Precomputed cache >/<--given as input-->/<--- from json ---->//
    //

    if (log_output)
      std::cout << "\n==============[KV CACHE SAVE MODE]================\n";
    allocateAndBindKVCache();
    setKVCachePosition(0);
    // chunked like any prefill: a prompt longer than INIT_SEQ_LEN rows
    for (auto &o : prefill(0, input_len))
      delete[] o;

    SYS_PROMP_LEN = input_len;
    save_kvcache(PRE_COMPUTED_CACHE_PATH, SYS_PROMP_LEN);

    if (log_output) {

      std::cout << "kv caches are saved in " << PRE_COMPUTED_CACHE_PATH
                << std::endl
                << "and the size of prompt is " << SYS_PROMP_LEN << ".\n"
                << "You may need this prompt length to set the "
                   "\"sys_prompt_token_size\""
                << "\n==================================================\n"
                << std::endl;
    }
    return;
  }

  if (USE_KVCACHE) {
    const auto t0 = std::chrono::steady_clock::now();
    load_kvcache(PRE_COMPUTED_CACHE_PATH, SYS_PROMP_LEN);
    std::cerr << "[TIMING] kv_load_ms="
              << std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now() - t0)
                   .count()
              << " rows=" << SYS_PROMP_LEN << std::endl;
  } else {
    SYS_PROMP_LEN = 0;
  }
  allocateAndBindKVCache();
  const unsigned int prefill_from = SYS_PROMP_LEN + global_token_len;
  std::vector<unsigned int> id_list;

  if (SKIP_PREFILL && init_len > 1) {
    // Prefill only N-1 tokens; the last input token will be used as the first
    // token in the generation phase (assigned directly, not sampled).
    unsigned int skipped_token =
      static_cast<unsigned int>(init_input[init_len - 1]);

    setKVCachePosition(prefill_from);
    output = prefill(prefill_from, init_len - 1);

    for (unsigned int b = 0; b < BATCH_SIZE; ++b)
      id_list.push_back(skipped_token);

    // Adjust lengths so the generation loop processes the skipped token
    // at the correct KV cache position.
    input_len -= 1;
    init_len -= 1;
  } else {
    setKVCachePosition(prefill_from);
    output = prefill(prefill_from, init_len);

    // post process of model output
    id_list = generate(output[0], do_sample, 1, ids_history, init_len);
    if (ppl_forced) {
      if (id_list[0] != ppl_cont[0])
        std::cerr << "[PPL] decode warning: prefill top1 != continuation[0]"
                  << std::endl;
      id_list[0] = ppl_cont[0];
    } else if (ppl_dec) {
      ppl_cont.push_back(id_list[0]);
    }

    // a chunked prompt (longer than INIT_SEQ_LEN) registers its first token too
    if (init_len != INIT_SEQ_LEN)
      registerOutputs(tokenizer, id_list, init_len, eos_list, log_output);
  }
  // output should be deallocated after use
  for (auto &out : output) {
    delete[] out;
  }
  if (generated_ids_path)
    generated_ids.insert(generated_ids.end(), id_list.begin(), id_list.end());

#ifdef ENABLE_HEXKL
  // The resident decode graph only needs a row of the CPU prefill scratch.
  // Outputs have been consumed above; KV placeholders and persistent tensors
  // retain their pages. Keep this opt-in while measuring the memory profile.
  if (const char *release = std::getenv("NNTR_HTP_RELEASE_PREFILL_SCRATCH")) {
    if (std::strcmp(release, "1") == 0) {
      if (auto *network =
            dynamic_cast<nntrainer::NeuralNetwork *>(model.get())) {
        const size_t bytes = network->releaseInferenceScratch();
        std::cerr << "[HTP] CPU prefill scratch discarded: " << (bytes >> 20)
                  << " MiB of virtual pages" << std::endl;
      }
    }
  }
#endif

  auto finish_prefill = std::chrono::high_resolution_clock::now();
  if (ppl_on) {
    double nll = 0.0;
    unsigned int n = 0;
    if (TieWordEmbedding::takePpl(nll, n) && n > 0) {
      std::cerr << "[PPL] prompt tokens=" << n << " nll/token=" << (nll / n)
                << " ppl=" << std::exp(nll / n) << std::endl;
    } else {
      std::cerr << "[PPL] no positions scored (untied lm_head, or batch > 1)"
                << std::endl;
    }
  }
  auto prefill_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
    finish_prefill - start_prefill);

  /**
   * TOKEN GENERATION
   */

  input_len += SYS_PROMP_LEN;

  // The windowed layers' caches from here on: the window, then one row a
  // generated token. The accelerator's decode caches of those layers are
  // sized and based the same (before its first decode row).
  // ponytail: one (rows, base) for all windowed layers -- they share one
  // window and one DSP cache; per-layer values if a model ever mixes them.
  compactKVCache(input_len + global_token_len);
  for (unsigned int l = 0; l < kv_cache.getNumLayers(); ++l)
    if (kv_cache.getRows(l) < kv_cache.getMaxSeqLen()) {
      const unsigned int base = kv_cache.getBase(l);
      htpDecodeKvWindow(input_len + global_token_len - base + NUM_TO_GENERATE,
                        base);
      break;
    }

  // Update generated token by prefill as an input
  for (unsigned int b = 0; b < BATCH_SIZE; ++b)
    input_sample[static_cast<size_t>(b) * MAX_SEQ_LEN] =
      static_cast<float>(id_list[b]);

  auto start_generation = std::chrono::high_resolution_clock::now();
  // One timestamp per completed token, seeded with start_generation, so the
  // report can time the true last 64 tokens even when the loop leaves early
  // (EOS or stop request). NUM_TO_GENERATE + 1 slots, allocated once.
  std::vector<std::chrono::high_resolution_clock::time_point> token_done_ts;
  token_done_ts.reserve(NUM_TO_GENERATE + 1);
  token_done_ts.push_back(start_generation);
  // [#150] NNTR_OP_TIME=1: the decode loop's split beside the per-node
  // [OP-TIME] lines (neuralnet.cpp). Unset, no clock is read.
  static const bool op_time = [] {
    const char *e = std::getenv("NNTR_OP_TIME");
    return e != nullptr && std::strcmp(e, "1") == 0;
  }();
  std::chrono::steady_clock::duration op_infer{}, op_sample{}, op_register{},
    op_token{};
  auto op_now = [] {
    return op_time ? std::chrono::steady_clock::now()
                   : std::chrono::steady_clock::time_point{};
  };
  // token_us runs from one token's end to the next, like token_done_ts, so
  // whatever no stage timer covers lands in the report's unattributed row
  auto op_prev = op_now();

  // [#132 Part B E3] the two-session decode returns only the token id
  // unless something reads the logits (NNTR_PPL_DECODE, sampling, a logits
  // processor, a batch); the NPU's pick skips the bad words as
  // applyBadWordsPenalty does
  const bool want_logits =
    ppl_dec || do_sample || logits_processor != nullptr || BATCH_SIZE != 1;
  htpDecodeWantLogits(want_logits);
  // The public pointer-return inference API always materializes its outputs.
  // Use the internal tensor result only when generation can consume an NPU
  // token id instead. A backend fallback still copies logits before generate.
  nntrainer::NeuralNetwork *decode_model = nullptr;
#ifdef ENABLE_HEXKL
  // Use the prefill result count: getOutputDimension() reports label shapes
  // and can be unavailable on an inference-only graph without label nodes.
  if (!want_logits && label.empty() && output.size() == 1u)
    decode_model = dynamic_cast<nntrainer::NeuralNetwork *>(model.get());
#endif
  htpDecodeBan(BAD_WORD_IDS.data(),
               BAD_WORD_IDS.size() != 0 ? NUM_BADWORDS : 0u);
  for (unsigned int token_generation_idx = input_len + 1;
       token_generation_idx < input_len + 1 + NUM_TO_GENERATE &&
       !stop_requested_.load(std::memory_order_acquire);
       ++token_generation_idx) {

    if (ppl_forced && generation_cnt + 1 >= ppl_cont.size())
      break; // the continuation is used up

    allocateAndBindKVCache();
    const auto op_t0 = op_now();
    const unsigned int from = token_generation_idx - 1 + global_token_len;
    const unsigned int to = token_generation_idx + global_token_len;
    std::vector<float *> output_interval;
    std::optional<nntrainer::Tensor> tensor_logits;
    unsigned int decoded_id = 0;
    bool have_decoded_id = false;
    nntrainer::sharedConstTensors decode_inputs;
    if (decode_model) {
      auto dims = decode_model->getInputDimension();
      bool supported = dims.size() == input.size();
      for (const auto &dim : dims) {
        const auto type = dim.getDataType();
        supported &= type == nntrainer::TensorDim::DataType::FP32 ||
                     type == nntrainer::TensorDim::DataType::FP16 ||
                     type == nntrainer::TensorDim::DataType::UINT16;
      }
      if (supported) {
        decode_inputs.reserve(dims.size());
        for (size_t i = 0; i < dims.size(); ++i) {
          dims[i].batch(BATCH_SIZE);
          const auto bytes = static_cast<unsigned int>(
            dims[i].getDataLen() * dims[i].getDataTypeSize());
          auto mapped =
            dims[i].getDataType() == nntrainer::TensorDim::DataType::FP32
              ? nntrainer::Tensor::Map(input[i], bytes, dims[i])
              : nntrainer::Tensor::Map(reinterpret_cast<uint16_t *>(input[i]),
                                       bytes, dims[i]);
          decode_inputs.push_back(
            std::make_shared<nntrainer::Tensor>(std::move(mapped)));
        }
      }
    }
    if (!decode_inputs.empty()) {
      auto tensors =
        decode_model->incremental_inference(decode_inputs, input_len, from, to);
      have_decoded_id = htpDecodeTokenId(&decoded_id);
      if (!have_decoded_id)
        tensor_logits.emplace(
          tensors.at(0)->clone(nntrainer::TensorDim::DataType::FP32));
    } else {
      output_interval = model->incremental_inference(BATCH_SIZE, input, label,
                                                     input_len, from, to);
    }
    const auto op_t1 = op_now();
    std::vector<unsigned int> ids_list =
      have_decoded_id
        ? std::vector<unsigned int>{decoded_id}
        : generate(decode_inputs.empty() ? output_interval[0]
                                         : tensor_logits->getData<float>(),
                   do_sample);
    const auto op_t2 = op_now();
    if (ppl_dec) {
      // Scored after generate(), on the logits the pick is taken from (the
      // bad-word ids are -inf there), so the self and the forced run score
      // the same array.
      const unsigned int top1 = ids_list[0];
      const unsigned int target =
        ppl_forced ? ppl_cont[generation_cnt + 1] : top1;
      const double nll =
        TieWordEmbedding::nllOf(output_interval[0], NUM_VOCAB, target);
      ppl_dec_nll += nll;
      ++ppl_dec_n;
      ppl_dec_top1 += top1 == target;
      std::string alts;
      for (size_t i = 0; i < ppl_alts.size(); ++i) {
        char buf[48];
        std::snprintf(buf, sizeof(buf), "%s%u:%.9g",
                      i ? "," : " alts=", ppl_alts[i],
                      output_interval[0][ppl_alts[i]]);
        alts += buf;
      }
      std::fprintf(stderr,
                   "[PPL] decode step=%u target=%u nll=%.17g top1=%u%s\n",
                   ppl_dec_n, target, nll, top1, alts.c_str());
      if (ppl_forced)
        ids_list[0] = target;
      else
        ppl_cont.push_back(target);
    }

    if (generated_ids_path)
      generated_ids.insert(generated_ids.end(), ids_list.begin(),
                           ids_list.end());

    // Feed the newly generated token back as the next input token.
    // token_generation_idx always starts at input_len + 1, so we are
    // always in the auto-regressive generation phase here.
    for (unsigned int b = 0; b < BATCH_SIZE; ++b) {
      input_sample[static_cast<size_t>(b) * MAX_SEQ_LEN] =
        static_cast<float>(ids_list[b]);
    }
    const auto op_t3 = op_now();
    registerOutputs(tokenizer, ids_list, token_generation_idx, eos_list,
                    log_output);
    ++generation_cnt;
    token_done_ts.push_back(std::chrono::high_resolution_clock::now());
    if (op_time) {
      const auto op_t4 = std::chrono::steady_clock::now();
      op_infer += op_t1 - op_t0;
      op_sample += op_t2 - op_t1;
      op_register += op_t4 - op_t3;
      op_token += op_t4 - op_prev;
      op_prev = op_t4;
    }

    // output should be deallocated after use
    for (auto out : output_interval) {
      delete[] out;
    }

    // check FINISH (a forced continuation runs to its end)
    for (unsigned int j = 0; j < BATCH_SIZE && !ppl_forced; ++j) {
      if (!eos_list[j] && (std::find(EOS_TOKEN_ID.begin(), EOS_TOKEN_ID.end(),
                                     ids_list[j]) != EOS_TOKEN_ID.end())) {
        eos_list[j] = true;
      }
    }

    bool is_finish = true;
    for (unsigned int j = 0; j < BATCH_SIZE; ++j) {
      if (!eos_list[j]) {
        is_finish = false;
        break;
      }
    }

    if (is_finish) {
      break;
    }

    if (stop_requested_.load(std::memory_order_acquire)) {
      break;
    }
  }

  // Always release the input buffer after the generation loop, whether
  // the loop exited early (EOS found) or ran to the maximum token limit.
  free(input_sample);

  if (ppl_dec && ppl_dec_n == 0) {
    std::cerr << "[PPL] decode no steps scored (num_to_generate 0, a "
                 "one-id file, or a stop request)"
              << std::endl;
  } else if (ppl_dec) {
    const double x = ppl_dec_nll / ppl_dec_n;
    std::fprintf(stderr,
                 "[PPL] decode tokens=%u nll/token=%.6g ppl=%.6g top1=%u/%u "
                 "source=%s nll_sum=%.17g\n",
                 ppl_dec_n, x, std::exp(x), ppl_dec_top1, ppl_dec_n,
                 ppl_forced ? "file" : "self", ppl_dec_nll);
    if (!ppl_forced) {
      std::ofstream f(ppl_dec_env);
      for (unsigned int id : ppl_cont)
        f << id << '\n';
      if (!f)
        std::cerr << "[PPL] decode cannot write " << ppl_dec_env << std::endl;
    }
  }

  global_token_len += (generation_cnt + init_len);

  htpDecodeWantLogits(true); // [#132 Part B E3] back to the default
  htpDecodeBan(nullptr, 0u);
  auto finish_generation = std::chrono::high_resolution_clock::now();
  if (generated_ids_path) {
    std::ofstream ids_file(generated_ids_path);
    for (unsigned int id : generated_ids)
      ids_file << id << '\n';
    if (!ids_file)
      throw std::runtime_error(std::string("Cannot write generated IDs: ") +
                               generated_ids_path);
  }
  auto generation_duration =
    std::chrono::duration_cast<std::chrono::milliseconds>(finish_generation -
                                                          start_generation);

  auto finish_total = std::chrono::high_resolution_clock::now();
  auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
    finish_total - start_total);
  size_t peak_memory = getPeakMemoryKb();

  if (log_output) {

    std::cout << "\n\n";
    std::cout << "=================[ LLM with NNTrainer ]===================\n";
    std::cout << "prefill: " << init_len << " tokens, "
              << prefill_duration.count() << " ms, "
              << ((double)init_len / prefill_duration.count() * 1000)
              << " TPS\n";
    std::cout << "generation: " << generation_cnt << " tokens, "
              << generation_duration.count() << " ms, "
              << ((double)generation_cnt / generation_duration.count() * 1000)
              << " TPS\n";
    // Decode tok/s over the last 64 completed tokens (contract §1.1). With
    // fewer than 64 tokens the window is the whole generation and says so.
    {
      const size_t n_done = token_done_ts.size() - 1;
      const size_t window = n_done >= 64 ? 64 : n_done;
      auto last64_duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(
          token_done_ts[n_done] - token_done_ts[n_done - window]);
      { // [#282 F] the first token against the rest
        using msd = std::chrono::duration<double, std::milli>;
        const size_t nt = token_done_ts.size() - 1;
        const double first = msd(token_done_ts[1] - token_done_ts[0]).count();
        const double rest =
          nt > 1 ? msd(token_done_ts[nt] - token_done_ts[1]).count() / (nt - 1)
                 : 0.0;
        std::cerr << "[TIMING] first_token_ms=" << first
                  << " rest_avg_ms=" << rest << std::endl;
      }
      std::cout << "generation(last 64): " << window << " tokens, "
                << last64_duration.count() << " ms, "
                << ((double)window / last64_duration.count() * 1000) << " TPS";
      if (window < 64)
        std::cout << " (whole generation, fewer than 64)";
      std::cout << "\n";
    }
    std::cout << "total: " << total_duration.count() << " ms\n";
    std::cout << "peak memory: " << peak_memory << " KB\n";
    std::cout << "==========================================================\n";
  }

  if (op_time) {
    using us = std::chrono::duration<double, std::micro>;
    std::fprintf(stderr,
                 "[OP-TIME] step tokens=%u infer_us=%.3f sample_us=%.3f "
                 "register_us=%.3f token_us=%.3f\n",
                 generation_cnt, us(op_infer).count(), us(op_sample).count(),
                 us(op_register).count(), us(op_token).count());
  }

  performance_metrics.prefill_tokens = init_len;
  performance_metrics.prefill_duration_ms = prefill_duration.count();
  performance_metrics.generation_tokens = generation_cnt;
  performance_metrics.generation_duration_ms = generation_duration.count();
  performance_metrics.total_duration_ms = total_duration.count();
  performance_metrics.peak_memory_kb = peak_memory;

  has_run_ = true;
}

std::string CausalLM::getOutput(int batch_idx) const {
  if (batch_idx < 0 || batch_idx >= static_cast<int>(output_list.size())) {
    return "";
  }
  return output_list[batch_idx];
}

} // namespace causallm
