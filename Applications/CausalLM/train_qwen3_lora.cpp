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
#include <residency_policy.h>
#include <transformer.h>

#include <dataset.h>
#include <model.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
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
    << "  --hybrid_trunk <dir>  run the frozen layers (before --lora_last_n)\n"
    << "                        on the GPU from this QS4CX-FP16 model dir,\n"
    << "                        overlapped with CPU training of the rest\n"
    << "  --seq_len <int>       training sequence length; overrides\n"
    << "                        nntr_config.json's init_seq_len. Attention\n"
    << "                        memory is quadratic in this, so prefer the\n"
    << "                        smallest length that fits your samples.\n"
    << "  --clip_grad <float>   clip LoRA gradients to this global norm\n"
    << "                        (0 = off, the default)\n"
    << "  --train_norms         also train the RMSNorm gammas alongside the\n"
    << "                        LoRA adapters (default: norms frozen)\n"
    << "  --lora_qat            fake-quantize loraA/loraB to the Q4_0 grid\n"
    << "                        during training (per-block EMA scales,\n"
    << "                        straight-through backward). Implies\n"
    << "                        --lora_weight_q4.\n"
    << "  --qat_ema_momentum <f> EMA momentum for the per-block QAT scale\n"
    << "                        update (0-1, default 0.1). Lower values "
       "smooth\n"
    << "                        the block scales over more steps — try 0.01\n"
    << "                        if long QAT runs destabilize.\n"
    << "  --validate            run a validation pass after every epoch and\n"
    << "                        keep the adapter with the best validation\n"
    << "                        loss (default: off -- the adapter with the\n"
    << "                        best epoch-mean training loss is kept)\n"
    << "  --lora_weight_q4      also save a Q4_0 adapter (output path with\n"
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

  bool saved = false;
  if (select_loss < st->best_loss) {
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
        st->model->save_weight_lora_q4(st->q4_output_path);
        std::cout << "  saved Q4_0 LoRA adapter -> " << st->q4_output_path
                  << std::endl;
      } catch (const std::exception &e) {
        std::cerr << "  failed to save Q4_0 LoRA adapter: " << e.what()
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

int runLoraTraining(const TrainParams &params, TrainProgressCb callback,
                    void *user_data, IterationProgressCb iter_cb,
                    void *iter_data) {
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
      nntr_cfg["lora_target"] = std::vector<std::string>{
        "wq",    "wk",         "wv",         "wo",      "ffn_up",
        "ffn_gate", "ffn_down", "wgate", "linattn_in_qkv", "linattn_in_z",
        "linattn_out"};

    if (params.lora_last_n)
      nntr_cfg["lora_last_n_layers"] = params.lora_last_n;

    if (params.train_norms)
      nntr_cfg["lora_train_norms"] = true;

    if (params.clip_grad >= 0.0f)
      nntr_cfg["lora_clip_grad_by_norm"] = params.clip_grad;

    // lora_qat implies lora_weight_q4: there is no point calibrating
    // fake-quant EMA scales without also saving the adapter they describe.
    // It only selects the extra Q4_0 save; the graph's LoRA tensors stay
    // FP32 (the lora_weight_q4 layer property builds inference-only Q4_0
    // tensors, which cannot train).
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

    const unsigned int seq_len = nntr_cfg["init_seq_len"].get<unsigned int>();
    const unsigned int vocab_size = cfg["vocab_size"].get<unsigned int>();

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
    auto model_uptr =
      causallm::Factory::Instance().create(architecture, cfg, generation_cfg,
                                           nntr_cfg);
    NNTR_THROW_IF(!model_uptr, std::invalid_argument)
      << "unsupported architecture for LoRA training: " << architecture
      << " (register it in model_registry.cpp / quantize.cpp)";
    // The driver's plumbing (lora save/load, training stats, hybrid trunk)
    // lives on Transformer; CausalLM adds the lm head. Both are covered.
    Transformer &model = *model_uptr;

    // GPU-trunk / CPU-head split: the frozen blocks [0, S) run as an
    // inference graph on the GPU engine, the trainable blocks [S, L) as the
    // CPU training graph whose input is the trunk's hidden state.
    const bool hybrid = !params.hybrid_trunk_path.empty();
    std::unique_ptr<Transformer> trunk;
    int trunk_layers = 0;
    if (hybrid) {
      trunk_layers = model.getLoraLayerStart();
      if (trunk_layers <= 0)
        throw std::invalid_argument(
          "--hybrid_trunk needs --lora_last_n < num_hidden_layers");
      model.setHeadFromLayer(trunk_layers);

      const std::string &td = params.hybrid_trunk_path;
      causallm::json tcfg = causallm::LoadJsonFile(td + "/config.json");
      causallm::json tgen = causallm::json::object();
      if (std::filesystem::exists(td + "/generation_config.json"))
        tgen = causallm::LoadJsonFile(td + "/generation_config.json");
      causallm::json tnntr = causallm::LoadJsonFile(td + "/nntr_config.json");
      tnntr["lora_rank"] = 0;
      tnntr["lora_alpha"] = 0;
      tnntr["tokenizer_file"] = nntr_cfg["tokenizer_file"];
      // the GPU attention arms (see README_GPU.md); an explicit setting wins
      setenv("NNTR_MHA_GPU", "1", 0);
      setenv("NNTR_KV_IMG_ATTN", "1", 0);
      // Keep the trunk's QS4CX host payload: with it released after the
      // device repack, the first trunk prefill trips the v8c weight-cache
      // rebuild guard (refuseIfQs4cxPayloadDropped). Costs one host copy of
      // the trunk FC weights.
      setenv("NNTR_V8C_DROP_PLAIN", "0", 0);
      trunk = causallm::Factory::Instance().create(
        resolveTrainingArchitecture(tcfg), tcfg, tgen, tnntr);
      NNTR_THROW_IF(!trunk, std::invalid_argument)
        << "unsupported trunk architecture";
      trunk->setTrunkLayers(trunk_layers);
      // the trunk's output is read back on the host by the CPU head
      auto &rp = nntrainer::ResidencyPolicy::global();
      rp.lower_patterns += std::string(rp.lower_patterns.empty() ? "" : ",") +
                           "layer" + std::to_string(trunk_layers - 1) +
                           "_decoder_output:out0";
      trunk->initialize();
      trunk->load_weight(td + "/" +
                         tnntr["model_file_name"].get<std::string>());
      std::cout << "hybrid:       GPU trunk blocks [0, " << trunk_layers
                << ") from " << td << ", CPU head blocks [" << trunk_layers
                << ", ...)" << std::endl;
    }

    const bool decision = params.task == "decision";
    // "lm35" is the Qwen3.5 demo of the same language-model task (the
    // Android TASK dropdown uses the value to switch the demo paths);
    // anything starting with "lm" trains the next-token loss.
    if (!decision && params.task.rfind("lm", 0) != 0)
      throw std::invalid_argument("--task must be lm* or decision, got " +
                                  params.task);
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

    model.initializeForTraining(params.lr, params.epochs, params.lr_decay_rate,
                                params.lr_decay_steps);

    const std::string weight_file =
      model_path + "/" + nntr_cfg["model_file_name"].get<std::string>();
    model.load_weight_lora(weight_file, params.resume_lora_path);

    auto *tokenizer = model.getTokenizer();
    if (tokenizer == nullptr)
      throw std::runtime_error(
        "model has no tokenizer; a tokenizer_file is required for training");

    TrainingDataGenerator data_gen(data_path, tokenizer, seq_len, vocab_size,
                                   params.max_samples, params.seed, decision);
    // Throws when the batch is larger than the sample count, which would
    // otherwise train zero iterations and end with the data buffer's own
    // "No data came while buffer ran".
    data_gen.setBatchSize(nntr_cfg.value("batch_size", 1u));
    std::cout << "samples:      " << data_gen.size() << " ("
              << data_gen.samplesPerEpoch() << " per epoch)" << std::endl;

    // Hybrid: the producer thread runs the GPU trunk for each sample, so the
    // trunk of the next sample overlaps the current training step. Which
    // positions are scored travels with the sample, in its label.
    if (hybrid) {
      const unsigned int L = seq_len;
      // the trunk feeds the head its hidden state through CausalLM::
      // trunkHidden(); every trainable architecture is a CausalLM. The
      // Transformer base is virtual (diamond), so the downcast must be
      // dynamic.
      auto *t = dynamic_cast<CausalLM *>(trunk.get());
      NNTR_THROW_IF(!t, std::invalid_argument)
        << "hybrid trunk is not a CausalLM";
      data_gen.input_writer = [t, L](const int32_t *ids, unsigned int used,
                                     float *input) {
        const auto t0 = std::chrono::steady_clock::now();
        t->trunkHidden(ids, used, L, input);
        /// [DIAG] NNTR_TRAIN_GRAD_DUMP=<dir>: the trunk hidden of the first
        /// produced samples, in production order (matches samples.txt).
        if (const char *dump = std::getenv("NNTR_TRAIN_GRAD_DUMP")) {
          static int produced = 0;
          if (produced < 16) {
            const std::string f = std::string(dump) + "/hidden_" +
                                  std::to_string(produced) + ".bin";
            if (FILE *fp = std::fopen(f.c_str(), "wb")) {
              std::fwrite(input, sizeof(float),
                          (size_t)used * t->getHiddenDim(), fp);
              std::fclose(fp);
            }
          }
          ++produced;
        }
        const double ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - t0)
                            .count();
        static double acc = 0;
        static unsigned int n = 0;
        static const bool prof = std::getenv("NNTR_TRAIN_PROF") != nullptr;
        acc += ms;
        if (++n % 10 == 0 && prof) {
          std::cout << "[PERF-TRUNK] avg " << acc / 10 << " ms (gpu trunk, "
                    << used << " tokens)" << std::endl;
          acc = 0;
        }
      };
    }

    std::shared_ptr<ml::train::Dataset> dataset = ml::train::createDataset(
      ml::train::DatasetType::GENERATOR, trainingDataGenCb, &data_gen);
    // Hybrid: a 1-slot iteration queue frees its slot only when the step that
    // consumed it ends, so the GPU trunk of the next sample would start after
    // the CPU step instead of under it. Two slots let it run one sample ahead.
    if (hybrid)
      dataset->setProperty({"buffer_size=2"});
    model.setDataset(ml::train::DatasetModeType::MODE_TRAIN, dataset);
    if (params.validate)
      model.setDataset(ml::train::DatasetModeType::MODE_VALID, dataset);

    EpochBridge bridge;
    bridge.model = &model;
    bridge.params = params;
    bridge.output_path = output_path;
    bridge.q4_output_path = q4_output_path;
    bridge.user_cb = callback;
    bridge.user_data = user_data;
    bridge.num_samples = static_cast<unsigned int>(data_gen.size());
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
    model.train(onEpochComplete, &bridge, nullptr, nullptr, iteration_wrapper,
                nullptr);

    std::cout << "training complete; best "
              << (params.validate ? "valid_loss=" : "train_loss=")
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
