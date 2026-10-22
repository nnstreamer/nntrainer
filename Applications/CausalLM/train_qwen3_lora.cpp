// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   train_qwen3_lora.cpp
 * @date   31 July 2026
 * @brief  CLI driver for LoRA fine-tuning a Qwen3 CausalLM model.
 * @bug    No known bugs except for NYI items
 */

#include <causal_lm.h>
#include <decision.h>
#include <factory.h>
#include <lora_train.h>
#include <model_registry.h>
#include <tokenizer_cache.h>
#include <transformer.h>

#include <dataset.h>
#include <engine.h> // does this build have a gpu context at all
#include <model.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

namespace {

void printUsage(const char *prog) {
  std::cout
    << "Usage: " << prog << " <model_dir> <train_data.txt> [options]\n"
    << "\nOptions:\n"
    << "  --lr <float>          learning rate (default 1e-4)\n"
    << "  --lr_decay_rate <f>   exponential lr decay rate; lr is multiplied\n"
    << "                        by this every --lr_decay_steps iterations\n"
    << "                        (0 = constant lr, the default). With the\n"
    << "                        default 90 samples/epoch, decay_steps 90\n"
    << "                        decays once per epoch.\n"
    << "  --lr_decay_steps <f>  decay interval in iterations (see above)\n"
    << "  --epochs <int>        number of epochs (default 1)\n"
    << "  --batch_size <int>    samples per optimizer step (default 1). The\n"
    << "                        graph is recompiled at this batch. An epoch\n"
    << "                        is floor(samples / batch) whole batches; the\n"
    << "                        reshuffled remainder sits that epoch out.\n"
    << "                        lr_decay_steps counts these steps.\n"
    << "  --output <path>       LoRA adapter output path\n"
    << "                        (default <model_dir>/lora_adapter.bin)\n"
    << "  --lora_path <path>    resume from an existing LoRA adapter\n"
    << "  --lora_rank <int>     LoRA rank; overrides nntr_config.json\n"
    << "  --lora_alpha <int>    LoRA alpha; overrides nntr_config.json\n"
    << "  --lora_last_n <int>   put LoRA only on the last N decoder layers\n"
    << "                        (default: all). Earlier layers have nothing\n"
    << "                        trainable, so their backward pass is skipped.\n"
    << "  --max_samples <int>   cap the number of training samples\n"
    << "  --hybrid_trunk <dir>  run the frozen layers (before\n"
    << "                        --lora_last_n) of the training graph on the\n"
    << "                        GPU instead of the host. The directory is not\n"
    << "                        read: the frozen region runs from the\n"
    << "                        training model's own weights, in the same\n"
    << "                        graph (it used to be a second model, which\n"
    << "                        cost 0.8 GB more than either half needs).\n"
    << "  --seq_len <int>       training sequence length; overrides\n"
    << "                        nntr_config.json's init_seq_len. Attention\n"
    << "                        memory is quadratic in this, so prefer the\n"
    << "                        smallest length that fits your samples.\n"
    << "  --clip_grad <float>   clip LoRA gradients to this global norm\n"
    << "                        (0 = off, the default)\n"
    << "  --train_norms         also train the RMSNorm gammas alongside the\n"
    << "                        LoRA adapters (default: norms frozen)\n"
    << "  --lora_qat            fake-quantize loraA/loraB to the int4 grid\n"
    << "                        during training (per-channel EMA scales,\n"
    << "                        straight-through backward). Implies\n"
    << "                        --lora_weight_q4.\n"
    << "  --qat_ema_momentum <f> EMA momentum for the per-channel QAT scale\n"
    << "                        update (0-1, default 0.1). Lower values "
       "smooth\n"
    << "                        the channel scales over more steps — try "
       "0.01\n"
    << "                        if long QAT runs destabilize.\n"
    << "  --frozen_cache        keep the frozen blocks' hidden state per\n"
    << "                        sample: epoch 1 runs them once, later epochs\n"
    << "                        skip them (they depend on nothing training\n"
    << "                        changes). With or without --hybrid_trunk.\n"
    << "  --frozen_cache_mb <n> budget for --frozen_cache in MB (default:\n"
    << "                        every sample fitting half the memory at\n"
    << "                        start; past the budget, samples compute)\n"
    << "  --gpu_base_gemm       with --hybrid_trunk: run the frozen blocks'\n"
    << "                        base-weight GEMMs on the GPU (verified "
       "against\n"
    << "                        the host dot). Slower than the host today --\n"
    << "                        for measurement, not for speed.\n"
    << "  --gpu_prefix_layers <n> with --hybrid_trunk: run only the first n\n"
    << "                        frozen blocks [0, n) on the GPU; blocks\n"
    << "                        [n, S) stay on the host as frozen FP32\n"
    << "                        forward-only blocks (S = the frozen count,\n"
    << "                        --lora_last_n back). Default: n = S, the\n"
    << "                        whole frozen prefix on the GPU.\n"
    << "  --activation_ckpt     activation checkpointing: recompute each\n"
    << "                        trained decoder block's activations before\n"
    << "                        its backward instead of keeping them (lower\n"
    << "                        peak memory, slower step; default: off)\n"
    << "  --validate            run a validation pass after every epoch and\n"
    << "                        keep the adapter with the best validation\n"
    << "                        loss (default: off -- the adapter with the\n"
    << "                        best epoch-mean training loss is kept)\n"
    << "  --valid_data <file>   validate on this JSONL file instead of the\n"
    << "                        training data (implies --validate)\n"
    << "  --patience <int>      stop early after N epochs in a row without a\n"
    << "                        validation improvement (implies --validate,\n"
    << "                        0 = off)\n"
    << "  --lora_weight_q4      also save a qint4 adapter (output path with\n"
    << "                        a _q4.bin suffix) alongside the FP32 one.\n"
    << "                        With --lora_qat, force-feeds the calibrated\n"
    << "                        EMA scales; without it, quantizes the\n"
    << "                        trained FP32 adapter post hoc (PTQ).\n"
    << "  --seed <int>          RNG seed for epoch shuffling (default 42)\n"
    << "  --task <lm|decision>  lm (default): every line of <train_data> is a\n"
    << "                        text sample, next-token loss on every\n"
    << "                        position. decision: <train_data> is eve-rlcd\n"
    << "                        JSONL questions (decision.h) and the model\n"
    << "                        learns the option letter at the last prompt\n"
    << "                        token.\n"
    << "  --objective <o>       decision: sft (default, cross-entropy on the\n"
    << "                        answer), rlcd (REINFORCE, reward 1[correct] -\n"
    << "                        p(action)) or rlvr (reward 1[correct])\n"
    << "  --group <int>         decision rlcd / rlvr: actions sampled per\n"
    << "                        question (default 4)\n";
}

} // namespace

namespace causallm {

/**
 * @brief Architecture key for the Factory, from a model dir's config.json.
 * The multimodal wrapper's architectures[0] ("Qwen3_5ForConditional...")
 * is registered directly; the generic fallback matches the plain-Qwen3
 * packs this driver shipped with originally.
 */
static std::string resolveTrainingArchitecture(const json &cfg) {
  if (cfg.contains("architectures") && cfg["architectures"].is_array() &&
      !cfg["architectures"].empty())
    return cfg["architectures"].get<std::vector<std::string>>()[0];
  return "Qwen3ForCausalLM";
}

/** @brief Bridges model.train(onEpochComplete, ...) to a TrainProgressCb. */
struct EpochBridge {
  Transformer *model;
  TrainParams params;
  std::string output_path;
  std::string q4_output_path; ///< empty unless lora_weight_q4 was passed
  unsigned int epoch = 0;
  float best_loss = std::numeric_limits<float>::max();
  unsigned int patience = 0; ///< 0 = off
  unsigned int epochs_without_improvement = 0;
  bool stop_requested = false;
  TrainProgressCb user_cb = nullptr;
  void *user_data = nullptr;
  unsigned int num_samples = 0;
  IterationProgressCb iter_cb = nullptr;
  void *iter_data = nullptr;
  /** when the current epoch's first iteration could start: model.train()
   *  for the first epoch, the end of the previous epoch's callback after */
  std::chrono::steady_clock::time_point epoch_start;
  /** when the current epoch's last training iteration ended; validation
   *  runs from here to the epoch callback */
  std::chrono::steady_clock::time_point last_iter_end;
};

static void onEpochComplete(void *user_data) {
  // epoch_complete_cb invoked from model.train()
  auto *st = static_cast<EpochBridge *>(user_data);
  ++st->epoch;

  // Timed before the adapter save below, which is not part of the epoch.
  using secs = std::chrono::duration<double>;
  const auto epoch_end = std::chrono::steady_clock::now();
  const double epoch_s = secs(epoch_end - st->epoch_start).count();
  const double train_s = st->last_iter_end > st->epoch_start
                           ? secs(st->last_iter_end - st->epoch_start).count()
                           : epoch_s;

  // train_stats.loss is the mean over the epoch's iterations
  auto train_stats = st->model->getTrainingStats();
  const bool validate = st->params.validate;
  const float valid_loss = validate ? st->model->getValidStats().loss
                                    : std::numeric_limits<float>::quiet_NaN();
  // the adapter is kept on the validation loss when there is one, else on
  // the epoch-mean training loss
  const float select_loss = validate ? valid_loss : train_stats.loss;
  const bool improved = select_loss < st->best_loss;

  // early stop: patience consecutive epochs without a validation improvement
  if (st->patience && validate) {
    if (improved)
      st->epochs_without_improvement = 0;
    else if (++st->epochs_without_improvement >= st->patience) {
      st->stop_requested = true;
      std::cout << "  no valid_loss improvement for " << st->patience
                << " epochs; stopping early" << std::endl;
    }
  }

  char timing[96];
  if (validate)
    std::snprintf(timing, sizeof(timing), "%.2fs (train %.2fs + valid %.2fs)",
                  epoch_s, train_s, epoch_s - train_s);
  else
    std::snprintf(timing, sizeof(timing), "%.2fs", train_s);
  std::cout << "[epoch " << st->epoch << "] train_loss=" << train_stats.loss;
  if (validate)
    std::cout << " valid_loss=" << valid_loss;
  std::cout << " time=" << timing << std::endl;

  // What the frozen-prefix reuse actually did this epoch, so a run can be read
  // without instrumenting it: hits == the samples this epoch skipped the frozen
  // blocks for, and MB is the memory the switch costs. Epoch 1 is all runs and
  // no hits by construction -- it is the pass that fills the cache -- so the
  // number to look at is epoch 2 onward.
  if (st->model->frozenCacheEnabled())
    std::cout << "  frozen prefix: ran " << st->model->getFrozenPrefixRuns()
              << ", reused " << st->model->getFrozenCacheHits() << " (cache "
              << (st->model->getFrozenCacheUsedBytes() >> 20) << " MB of "
              << (st->model->getFrozenCacheBudgetBytes() >> 20) << " MB)\n";

  bool saved = false;
  if (improved) {
    st->best_loss = select_loss;
    try {
      st->model->save_weight_lora(st->output_path);
      saved = true;
      std::cout << "  saved LoRA adapter -> " << st->output_path << std::endl;
    } catch (const std::exception &e) {
      std::cerr << "  failed to save LoRA adapter: " << e.what() << std::endl;
    }
    if (!st->q4_output_path.empty()) {
      try {
        st->model->save_weight_lora_qint4(st->q4_output_path);
        std::cout << "  saved qint4 LoRA adapter -> " << st->q4_output_path
                  << std::endl;
      } catch (const std::exception &e) {
        std::cerr << "  failed to save qint4 LoRA adapter: " << e.what()
                  << std::endl;
      }
    }
  }

  if (st->user_cb != nullptr) {
    TrainProgress progress;
    progress.epoch = st->epoch;
    progress.train_loss = train_stats.loss;
    progress.valid_loss = valid_loss;
    progress.accuracy = 0.0f;
    progress.saved = saved;
    progress.output_path = st->output_path;
    progress.q4_output_path = st->q4_output_path;
    progress.best_loss = st->best_loss;
    progress.num_samples = st->num_samples;
    progress.epoch_seconds = epoch_s;
    progress.train_seconds = train_s;
    st->user_cb(progress, st->user_data);
  }
  st->epoch_start = std::chrono::steady_clock::now();
}

/** @brief stop_cb for --patience: break before the next epoch once the epoch
 *  callback has counted `patience` epochs without a validation improvement. */
static bool onEpochStop(void *user_data) {
  return static_cast<EpochBridge *>(user_data)->stop_requested;
}

/**
 * @brief Memory the system can hand out now (MemAvailable); 0 if unknown.
 */
static size_t availableMemoryBytes() {
#if defined(__linux__)
  std::ifstream f("/proc/meminfo");
  std::string key;
  size_t kb = 0;
  std::string unit;
  while (f >> key >> kb >> unit)
    if (key == "MemAvailable:")
      return kb << 10;
#endif
  return 0;
}

int runLoraTraining(const TrainParams &params_in, TrainProgressCb callback,
                    void *user_data, IterationProgressCb iter_cb,
                    void *iter_data) {
  // --frozen_cache rewrites the trunk settings below
  TrainParams params = params_in;
  const std::string &model_path = params.model_path;
  const std::string &data_path = params.data_path;

  std::string output_path = params.output_path;
  if (output_path.empty())
    output_path = model_path + "/lora_adapter.bin";

  try {
    causallm::json cfg = causallm::LoadJsonFile(model_path + "/config.json");
    // Multimodal wrappers (Qwen3.5) nest the text fields under text_config;
    // flatten them before the driver's direct cfg reads (vocab_size) and
    // before the model's own sanitizeConfig (which skips existing keys).
    if (cfg.contains("text_config") && cfg["text_config"].is_object()) {
      const auto &text_cfg = cfg["text_config"];
      for (auto it = text_cfg.begin(); it != text_cfg.end(); ++it)
        if (!cfg.contains(it.key()))
          cfg[it.key()] = it.value();
    }
    causallm::json generation_cfg = causallm::json::object();
    const std::string gen_cfg_path = model_path + "/generation_config.json";
    if (std::filesystem::exists(gen_cfg_path))
      generation_cfg = causallm::LoadJsonFile(gen_cfg_path);
    causallm::json nntr_cfg =
      causallm::LoadJsonFile(model_path + "/nntr_config.json");

    // Resolve the tokenizer path against the model directory. These configs
    // are shipped with on-device (e.g. /data/local/tmp/...) absolute paths
    // that do not exist on a build host, so fall back to the same filename
    // inside model_dir whenever the configured path is missing.
    {
      std::filesystem::path tok =
        nntr_cfg.value("tokenizer_file", std::string("tokenizer.json"));
      if (tok.is_relative())
        tok = std::filesystem::path(model_path) / tok;
      if (!std::filesystem::exists(tok))
        tok = std::filesystem::path(model_path) / tok.filename();
      if (!std::filesystem::exists(tok))
        throw std::runtime_error("tokenizer not found; looked for " +
                                 tok.string());
      nntr_cfg["tokenizer_file"] = tok.string();
    }

    // Default the LoRA config if the model's nntr_config.json leaves it off
    // (or has it at 0, the "LoRA disabled" default). Params win.
    if (params.lora_rank)
      nntr_cfg["lora_rank"] = params.lora_rank;
    else if (!nntr_cfg.contains("lora_rank") ||
             nntr_cfg["lora_rank"].get<unsigned int>() == 0)
      nntr_cfg["lora_rank"] = 8;

    if (params.lora_alpha)
      nntr_cfg["lora_alpha"] = params.lora_alpha;
    else if (!nntr_cfg.contains("lora_alpha") ||
             nntr_cfg["lora_alpha"].get<unsigned int>() == 0)
      nntr_cfg["lora_alpha"] = nntr_cfg["lora_rank"].get<unsigned int>() * 2;

    if (!nntr_cfg.contains("lora_target") || nntr_cfg["lora_target"].empty())
      nntr_cfg["lora_target"] =
        std::vector<std::string>{"wq",           "wk",         "wv",
                                 "wo",           "ffn_up",     "ffn_gate",
                                 "ffn_down",     "wgate",      "linattn_in_qkv",
                                 "linattn_in_z", "linattn_out"};

    if (params.lora_last_n)
      nntr_cfg["lora_last_n_layers"] = params.lora_last_n;

    if (params.train_norms)
      nntr_cfg["lora_train_norms"] = true;

    if (params.clip_grad >= 0.0f)
      nntr_cfg["lora_clip_grad_by_norm"] = params.clip_grad;

    // lora_qat implies lora_weight_q4: there is no point calibrating
    // fake-quant EMA scales without also saving the adapter they describe.
    // It only selects the extra adapter save; the graph's LoRA tensors are
    // always FP32.
    bool do_qat = params.lora_qat;
    bool do_weight_q4 = params.lora_weight_q4;
    if (do_qat)
      do_weight_q4 = true;
    if (do_qat)
      nntr_cfg["lora_qat"] = true;
    if (do_qat && params.qat_ema_momentum > 0.0f)
      nntr_cfg["lora_qat_ema_momentum"] = params.qat_ema_momentum;

    if (params.seq_len) {
      nntr_cfg["init_seq_len"] = params.seq_len;
      // max_seq_len must not sit below the training length; mha_core derives
      // its max_timestep from it.
      if (nntr_cfg.value("max_seq_len", 0u) < params.seq_len)
        nntr_cfg["max_seq_len"] = params.seq_len;
    }

    // The batch is a model property: the graph is compiled at this batch and
    // nntrainer's IterationQueue groups that many generator samples per
    // optimizer step. The generator itself stays sample-at-a-time.
    if (params.batch_size)
      nntr_cfg["batch_size"] = params.batch_size;

    const unsigned int vocab_size = cfg["vocab_size"].get<unsigned int>();

    const bool decision = params.task == "decision";
    // "lm35" is the Qwen3.5 demo of the same language-model task (the
    // Android TASK dropdown uses the value to switch the demo paths);
    // anything starting with "lm" trains the next-token loss.
    if (!decision && params.task.rfind("lm", 0) != 0)
      throw std::invalid_argument("--task must be lm* or decision, got " +
                                  params.task);

    // The data is read before the graph is built, so the graph can be built
    // only as wide as the data needs. A sample is right-padded to the
    // sequence width and every padded row is computed through the trainable
    // blocks (forward and backward) without ever being read: the causal
    // mask hides it from the real rows and the loss scores none of it. The
    // configured seq_len is therefore a cap; the graph takes the longest
    // sample rounded up to 16 rows (measured on the 123-139 token decision
    // demo at seq_len 256: epoch 2.05 s -> 1.35 s, bit-identical losses).
    const unsigned int seq_cap = nntr_cfg["init_seq_len"].get<unsigned int>();
    std::unique_ptr<TrainingDataGenerator> data_gen;
    std::unique_ptr<TrainingDataGenerator> valid_gen;
    // A separate validation file implies --validate. It gets its own
    // generator (and GPU-trunk writer in hybrid mode) so its rows never mix
    // into the training epoch's shuffle; the adapter is then kept on the
    // validation loss and --patience can stop the run early.
    const bool do_validate =
      params.validate || !params.valid_data_path.empty() || params.patience > 0;
    {
      // the model's own tokenizer is not built yet; this copy only encodes
      // the samples (the snapshot cache makes the second load cheap)
      auto data_tok = causallm::LoadTokenizerCached(
        nntr_cfg["tokenizer_file"].get<std::string>());
      NNTR_THROW_IF(!data_tok, std::runtime_error)
        << "failed to load the tokenizer for the training data";
      data_gen = std::make_unique<TrainingDataGenerator>(
        data_path, data_tok.get(), seq_cap, vocab_size, params.max_samples,
        params.seed, decision);
      if (do_validate) {
        const std::string &vpath =
          params.valid_data_path.empty() ? data_path : params.valid_data_path;
        valid_gen = std::make_unique<TrainingDataGenerator>(
          vpath, data_tok.get(), seq_cap, vocab_size, /*max_samples=*/0,
          params.seed, decision);
      }
    }
    {
      unsigned int longest = data_gen->longestInput();
      if (valid_gen)
        longest = std::max(longest, valid_gen->longestInput());
      const unsigned int fitted = std::min(seq_cap, (longest + 15) / 16 * 16);
      if (fitted < seq_cap) {
        nntr_cfg["init_seq_len"] = fitted;
        data_gen->setSeqLen(fitted);
        if (valid_gen)
          valid_gen->setSeqLen(fitted);
        std::cout << "seq_fit:      " << seq_cap << " -> " << fitted
                  << " (longest sample " << longest << " rows)" << std::endl;
      }
    }
    const unsigned int seq_len = nntr_cfg["init_seq_len"].get<unsigned int>();

    std::string q4_output_path;
    if (do_weight_q4) {
      std::filesystem::path p(output_path);
      q4_output_path =
        (p.parent_path() / (p.stem().string() + "_q4.bin")).string();
    }

    std::cout
      << "model:        " << model_path << "\n"
      << "data:         " << data_path << "\n"
      << "lora_rank:    " << nntr_cfg["lora_rank"].get<unsigned int>() << "\n"
      << "lora_alpha:   " << nntr_cfg["lora_alpha"].get<unsigned int>() << "\n"
      << "seq_len:      " << nntr_cfg["init_seq_len"].get<unsigned int>()
      << "\n"
      << "clip_grad:    " << nntr_cfg.value("lora_clip_grad_by_norm", 0.0f)
      << "\n"
      << "lora_qat:     " << (do_qat ? "true" : "false") << "\n"
      << "lr:           " << params.lr << "\n"
      << "lr_decay:     "
      << (params.lr_decay_rate > 0.0f && params.lr_decay_steps > 0.0f
            ? "rate " + std::to_string(params.lr_decay_rate) + " every " +
                std::to_string(params.lr_decay_steps) + " steps"
            : "off")
      << "\n"
      << "epochs:       " << params.epochs << "\n"
      << "batch_size:   " << nntr_cfg.value("batch_size", 1u) << "\n"
      << "output:       " << output_path << "\n";
    if (!q4_output_path.empty())
      std::cout << "output_q4:    " << q4_output_path << std::endl;
    else
      std::cout << std::flush;

    // Factory-based model construction: the driver is architecture-agnostic
    // (config.json's architectures[0] names the model; model_type=CausalLM
    // is the generic fallback). registerAllModels() shares the api's single
    // factory instance.
    const std::string architecture = resolveTrainingArchitecture(cfg);
    registerAllModels();
    auto model_uptr = causallm::Factory::Instance().create(
      architecture, cfg, generation_cfg, nntr_cfg);
    NNTR_THROW_IF(!model_uptr, std::invalid_argument)
      << "unsupported architecture for LoRA training: " << architecture
      << " (register it in model_registry.cpp / quantize.cpp)";
    // The driver's plumbing (lora save/load, training stats, hybrid trunk)
    // lives on Transformer; CausalLM adds the lm head. Both are covered.
    Transformer &model = *model_uptr;

    // One graph, two regions: the frozen blocks [0, S) -- everything before
    // --lora_last_n, which has nothing trainable in it -- and the trainable
    // blocks [S, L), inside the SAME graph built from the SAME weights.
    //
    // This replaces a second, separate model that used to be loaded for the
    // frozen region (setHeadFromLayer + a trunk CausalLM + a producer-thread
    // hidden-state writer). Two graphs meant two weight pools, two copies of
    // the embedding and the lm head, two device contexts and a per-sample
    // hand-off through a placeholder tensor: measured on the S26U at 1.43 GB
    // against 0.6 GB host-only, for a split that on its own saves about a third
    // of the step. The duplication, not the split, was the cost.
    const bool hybrid = !params.hybrid_trunk_path.empty();
    // Both switches name the range; only --hybrid_trunk also asks for the
    // device. Naming it is what lets epochs 2.. skip it (--frozen_cache) and
    // lets the offload below target it.
    const bool split = hybrid || params.frozen_cache;
    const int frozen_blocks = split ? model.getLoraLayerStart() : 0;
    if (split)
      NNTR_THROW_IF(frozen_blocks <= 0, std::invalid_argument)
        << "--hybrid_trunk / --frozen_cache need --lora_last_n < "
           "num_hidden_layers (otherwise no block is frozen and there is "
           "nothing to split or reuse)";

    // Does this build have a device at all? A CPU-only build (or
    // NNTR_ENGINE=cpu) keeps the frozen range entirely on the host, which is
    // exactly what
    // --frozen_cache alone asks for.
    bool gpu_present = false;
#if defined(ENABLE_OPENCL)
    try {
      gpu_present =
        nntrainer::Engine::Global().getRegisteredContext("gpu") != nullptr;
    } catch (std::exception &) {
      gpu_present = false;
    }
#endif
    if (hybrid && !gpu_present)
      std::cout << "note:         no gpu context in this build; the frozen "
                   "blocks stay on the host"
                << std::endl;

    const bool prefix_on_gpu = hybrid && gpu_present;
    // When prefix runs on GPU, run GPU activations in FP16 for 2x memory
    // reduction and Adreno FP16 kernel acceleration (unless NNTR_GPU_FP32=1).
    const bool gpu_force_fp32 = std::getenv("NNTR_GPU_FP32") != nullptr &&
                                std::atoi(std::getenv("NNTR_GPU_FP32")) != 0;
    const bool frozen_fp16 = prefix_on_gpu && !gpu_force_fp32;

    if (prefix_on_gpu) {
      // --gpu_prefix_layers G: end the device sub-range at G, leaving [G, S)
      // on the host as frozen FP32 forward-only blocks. 0 (unset) = G = S.
      unsigned int gpu_end = frozen_blocks;
      if (params.gpu_prefix_layers > 0) {
        NNTR_THROW_IF(params.gpu_prefix_layers > (unsigned int)frozen_blocks,
                      std::invalid_argument)
          << "--gpu_prefix_layers " << params.gpu_prefix_layers
          << " exceeds the frozen block count " << frozen_blocks
          << " (--lora_last_n " << params.lora_last_n << ")";
        gpu_end = params.gpu_prefix_layers;
      }
      model.setFrozenPrefixEngine(frozen_blocks, "gpu", frozen_fp16,
                                  (int)gpu_end);
      std::cout << "gpu prefix:   blocks [0, " << gpu_end << ") of "
                << frozen_blocks << " frozen on engine=gpu"
                << (gpu_end < (unsigned int)frozen_blocks
                      ? ", [" + std::to_string(gpu_end) + ", " +
                          std::to_string(frozen_blocks) + ") on cpu"
                      : "")
                << std::endl;
    } else if (split) {
      model.setFrozenPrefix(frozen_blocks, false);
    }
    if (hybrid && gpu_present && params.gpu_base_gemm) {
      nntrainer::nntr_setenv("NNTR_TRAIN_BASE_GPU", "1", 0);
    }
    // The frozen region's output depends on nothing training changes -- its
    // weights are frozen and a sample's token ids never change -- so from epoch
    // 2 a repeated sample can skip it. Budgeted in bytes; samples past the
    // budget are computed normally, so a small budget only costs time.
    if (params.frozen_cache) {
      // No estimate of the data set here on purpose: the cache keeps samples
      // until the budget is full and computes the rest, so giving it the
      // smaller of "what the user asked for" and "half of what the system can
      // hand out now" bounds the memory without knowing how many samples there
      // are or how long they are.
      size_t budget = (size_t)params.frozen_cache_mb << 20;
      const size_t avail = availableMemoryBytes() / 2;
      if (budget == 0)
        budget = avail;
      else if (avail)
        budget = std::min(budget, avail);
      model.setFrozenCache(true, budget);
    }

    if (decision) {
      model.setDecisionTask(decisionLetterIds(model_path, model.getTokenizer()),
                            params.decision_objective, params.decision_group,
                            params.seed);
      std::cout << "task:         decision (" << params.decision_objective
                << (params.decision_objective == "sft"
                      ? ""
                      : ", group " + std::to_string(params.decision_group))
                << ")" << std::endl;
    }

    if (params.activation_ckpt)
      model.setActivationCheckpointing(true);
    std::cout << "act. ckpt:    "
              << (model.activationCheckpointing() ? "on" : "off") << std::endl;
    model.initializeForTraining(params.lr, params.epochs, params.lr_decay_rate,
                                params.lr_decay_steps);

    const std::string weight_file =
      model_path + "/" + nntr_cfg["model_file_name"].get<std::string>();
    // The frozen region is part of this graph and comes from this file: there
    // is no second model left to load.
    model.load_weight_lora(weight_file, params.resume_lora_path);

    auto *tokenizer = model.getTokenizer();
    if (tokenizer == nullptr)
      throw std::runtime_error(
        "model has no tokenizer; a tokenizer_file is required for training");

    // Throws when the batch is larger than the sample count, which would
    // otherwise train zero iterations and end with the data buffer's own
    // "No data came while buffer ran".
    data_gen->setBatchSize(nntr_cfg.value("batch_size", 1u));
    std::cout << "samples:      " << data_gen->size() << " ("
              << data_gen->samplesPerEpoch() << " per epoch)" << std::endl;
    if (valid_gen) {
      valid_gen->setBatchSize(nntr_cfg.value("batch_size", 1u));
      std::cout << "valid:        " << valid_gen->size() << " samples from "
                << (params.valid_data_path.empty() ? data_path
                                                   : params.valid_data_path)
                << std::endl;
    }

    // Nothing to wire between a trunk and a head any more: the frozen region
    // hands its hidden state to the trainable region through the graph's own
    // tensors, and the generator produces token ids for every configuration --
    // including the accelerator one, which used to receive precomputed hidden
    // states through a producer-thread writer. Dropping that writer also drops
    // the 2-slot iteration queue it needed: cross-sample overlap between the
    // two regions is deliberately not re-enabled here, because inside one graph
    // the planner reuses the frozen region's memory for the trainable region's
    // activations, so sample N+1's prefix would overwrite sample N's tail while
    // it is still being trained. Overlap is a separate change (it needs a
    // lifespan partition of the pool, not a second pool), and until it lands
    // the single graph is serial -- which is also the only configuration whose
    // memory is bounded.
    std::shared_ptr<ml::train::Dataset> dataset = ml::train::createDataset(
      ml::train::DatasetType::GENERATOR, trainingDataGenCb, data_gen.get());
    model.setDataset(ml::train::DatasetModeType::MODE_TRAIN, dataset);

    // Validation keeps its own generator so its rows never mix into the
    // training shuffle.
    if (valid_gen) {
      std::shared_ptr<ml::train::Dataset> valid_dataset =
        ml::train::createDataset(ml::train::DatasetType::GENERATOR,
                                 trainingDataGenCb, valid_gen.get());
      model.setDataset(ml::train::DatasetModeType::MODE_VALID, valid_dataset);
    }

    if (hybrid || params.frozen_cache)
      std::cout << "frozen cache: "
                << (model.frozenCacheEnabled() ? "on" : "off") << std::endl;

    EpochBridge bridge;
    bridge.model = &model;
    bridge.params = params;
    // a separate validation file or --patience implies the validation pass
    bridge.params.validate = do_validate;
    bridge.patience = params.patience;
    bridge.output_path = output_path;
    bridge.q4_output_path = q4_output_path;
    bridge.user_cb = callback;
    bridge.user_data = user_data;
    bridge.num_samples = static_cast<unsigned int>(data_gen->size());
    bridge.iter_cb = iter_cb;
    bridge.iter_data = iter_data;

    auto iteration_wrapper = [&bridge](unsigned int iteration, float loss,
                                       void *) {
      bridge.last_iter_end = std::chrono::steady_clock::now();
      if (loraTrainCancelledFlag().load())
        throw std::runtime_error("training cancelled by user");
      if (bridge.iter_cb)
        bridge.iter_cb(iteration, bridge.epoch + 1, loss, bridge.iter_data);
    };

    bridge.epoch_start = std::chrono::steady_clock::now();
    model.train(onEpochComplete, &bridge, onEpochStop, &bridge,
                iteration_wrapper, nullptr);

    std::cout << "training complete; best "
              << (bridge.params.validate ? "valid_loss=" : "train_loss=")
              << bridge.best_loss << ", adapter at " << output_path
              << std::endl;
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

} // namespace causallm

int main(int argc, char *argv[]) {
  if (argc < 3) {
    printUsage(argv[0]);
    return EXIT_FAILURE;
  }

  causallm::TrainParams params;
  params.model_path = argv[1];
  params.data_path = argv[2];

  for (int i = 3; i < argc; ++i) {
    const std::string arg = argv[i];
    auto next = [&](const char *name) -> std::string {
      if (i + 1 >= argc) {
        throw std::invalid_argument(std::string("missing value for ") + name);
      }
      return argv[++i];
    };
    try {
      if (arg == "--lr")
        params.lr = std::stof(next("--lr"));
      else if (arg == "--lr_decay_rate")
        params.lr_decay_rate = std::stof(next("--lr_decay_rate"));
      else if (arg == "--lr_decay_steps")
        params.lr_decay_steps = std::stof(next("--lr_decay_steps"));
      else if (arg == "--epochs")
        params.epochs = static_cast<unsigned int>(std::stoul(next("--epochs")));
      else if (arg == "--batch_size")
        params.batch_size =
          static_cast<unsigned int>(std::stoul(next("--batch_size")));
      else if (arg == "--output")
        params.output_path = next("--output");
      else if (arg == "--lora_path")
        params.resume_lora_path = next("--lora_path");
      else if (arg == "--lora_rank")
        params.lora_rank =
          static_cast<unsigned int>(std::stoul(next("--lora_rank")));
      else if (arg == "--lora_alpha")
        params.lora_alpha =
          static_cast<unsigned int>(std::stoul(next("--lora_alpha")));
      else if (arg == "--lora_last_n")
        params.lora_last_n =
          static_cast<unsigned int>(std::stoul(next("--lora_last_n")));
      else if (arg == "--max_samples")
        params.max_samples =
          static_cast<unsigned int>(std::stoul(next("--max_samples")));
      else if (arg == "--seq_len")
        params.seq_len =
          static_cast<unsigned int>(std::stoul(next("--seq_len")));
      else if (arg == "--clip_grad")
        params.clip_grad = std::stof(next("--clip_grad"));
      else if (arg == "--train_norms")
        params.train_norms = true;
      else if (arg == "--lora_qat")
        params.lora_qat = true;
      else if (arg == "--lora_weight_q4")
        params.lora_weight_q4 = true;
      else if (arg == "--qat_ema_momentum")
        params.qat_ema_momentum = std::stof(next("--qat_ema_momentum"));
      else if (arg == "--validate")
        params.validate = true;
      else if (arg == "--activation_ckpt")
        params.activation_ckpt = true;
      else if (arg == "--hybrid_sync")
        params.hybrid_sync = true;
      else if (arg == "--frozen_cache")
        params.frozen_cache = true;
      else if (arg == "--gpu_base_gemm")
        params.gpu_base_gemm = true;
      else if (arg == "--gpu_prefix_layers")
        params.gpu_prefix_layers =
          static_cast<unsigned int>(std::stoul(next("--gpu_prefix_layers")));
      else if (arg == "--frozen_cache_mb")
        params.frozen_cache_mb =
          static_cast<unsigned int>(std::stoul(next("--frozen_cache_mb")));
      else if (arg == "--valid_data")
        params.valid_data_path = next("--valid_data");
      else if (arg == "--patience")
        params.patience =
          static_cast<unsigned int>(std::stoul(next("--patience")));
      else if (arg == "--seed")
        params.seed = static_cast<unsigned int>(std::stoul(next("--seed")));
      else if (arg == "--task")
        params.task = next("--task");
      else if (arg == "--objective")
        params.decision_objective = next("--objective");
      else if (arg == "--group")
        params.decision_group =
          static_cast<unsigned int>(std::stoul(next("--group")));
      else if (arg == "--hybrid_trunk")
        params.hybrid_trunk_path = next("--hybrid_trunk");
      else {
        std::cerr << "Unknown option: " << arg << std::endl;
        printUsage(argv[0]);
        return EXIT_FAILURE;
      }
    } catch (const std::exception &e) {
      std::cerr << "Bad arguments: " << e.what() << std::endl;
      return EXIT_FAILURE;
    }
  }

  return causallm::runLoraTraining(params);
}
