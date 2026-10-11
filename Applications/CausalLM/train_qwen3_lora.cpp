// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   train_qwen3_lora.cpp
 * @date   31 July 2026
 * @brief  CLI driver for LoRA fine-tuning a Qwen3 CausalLM model.
 * @bug    No known bugs except for NYI items
 */

#include <lora_train.h>
#include <qwen3_causallm.h>
#include <transformer.h>

#include <dataset.h>
#include <model.h>

#include <cstdlib>
#include <filesystem>
#include <iostream>
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
    << "  --output <path>       LoRA adapter output path\n"
    << "                        (default <model_dir>/lora_adapter.bin)\n"
    << "  --lora_path <path>    resume from an existing LoRA adapter\n"
    << "  --lora_rank <int>     LoRA rank; overrides nntr_config.json\n"
    << "  --lora_alpha <int>    LoRA alpha; overrides nntr_config.json\n"
    << "  --lora_last_n <int>   put LoRA only on the last N decoder layers\n"
    << "                        (default: all). Earlier layers have nothing\n"
    << "                        trainable, so their backward pass is skipped.\n"
    << "  --max_samples <int>   cap the number of training samples\n"
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
    << "  --lora_weight_q4      also save a Q4_0 adapter (output path with\n"
    << "                        a _q4.bin suffix) alongside the FP32 one.\n"
    << "                        With --lora_qat, force-feeds the calibrated\n"
    << "                        EMA scales; without it, quantizes the\n"
    << "                        trained FP32 adapter post hoc (PTQ).\n"
    << "  --seed <int>          RNG seed for epoch shuffling (default 42)\n";
}

} // namespace

namespace causallm {

/** @brief Bridges model.train(onEpochComplete, ...) to a TrainProgressCb. */
struct EpochBridge {
  Qwen3CausalLM *model;
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
};

static void onEpochComplete(void *user_data) {
  // epoch_complete_cb invoked from model.train()
  auto *st = static_cast<EpochBridge *>(user_data);
  ++st->epoch;

  auto train_stats = st->model->getTrainingStats();
  auto valid_stats = st->model->getValidStats();

  std::cout << "[epoch " << st->epoch << "] train_loss=" << train_stats.loss
            << " valid_loss=" << valid_stats.loss << std::endl;

  bool saved = false;
  if (valid_stats.loss < st->best_loss) {
    st->best_loss = valid_stats.loss;
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
    progress.valid_loss = valid_stats.loss;
    progress.accuracy = 0.0f;
    progress.saved = saved;
    progress.output_path = st->output_path;
    progress.q4_output_path = st->q4_output_path;
    progress.best_loss = st->best_loss;
    progress.num_samples = st->num_samples;
    st->user_cb(progress, st->user_data);
  }
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
        "wq", "wk", "wv", "wo", "ffn_up", "ffn_gate", "ffn_down"};

    if (params.lora_last_n)
      nntr_cfg["lora_last_n_layers"] = params.lora_last_n;

    if (params.train_norms)
      nntr_cfg["lora_train_norms"] = true;

    if (params.clip_grad >= 0.0f)
      nntr_cfg["lora_clip_grad_by_norm"] = params.clip_grad;

    // lora_qat implies lora_weight_q4: there is no point calibrating
    // fake-quant EMA scales without also saving the adapter they describe.
    bool do_qat = params.lora_qat;
    bool do_weight_q4 = params.lora_weight_q4;
    if (do_qat)
      do_weight_q4 = true;
    if (do_qat)
      nntr_cfg["lora_qat"] = true;
    if (do_weight_q4)
      nntr_cfg["lora_weight_q4"] = true;
    if (do_qat && params.qat_ema_momentum > 0.0f)
      nntr_cfg["lora_qat_ema_momentum"] = params.qat_ema_momentum;

    if (params.seq_len) {
      nntr_cfg["init_seq_len"] = params.seq_len;
      // max_seq_len must not sit below the training length; mha_core derives
      // its max_timestep from it.
      if (nntr_cfg.value("max_seq_len", 0u) < params.seq_len)
        nntr_cfg["max_seq_len"] = params.seq_len;
    }

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
      << "output:       " << output_path << "\n";
    if (!q4_output_path.empty())
      std::cout << "output_q4:    " << q4_output_path << std::endl;
    else
      std::cout << std::flush;

    Qwen3CausalLM model(cfg, generation_cfg, nntr_cfg);
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
                                   params.max_samples, params.seed);
    std::cout << "samples:      " << data_gen.size() << std::endl;

    std::shared_ptr<ml::train::Dataset> dataset = ml::train::createDataset(
      ml::train::DatasetType::GENERATOR, trainingDataGenCb, &data_gen);
    model.setDataset(ml::train::DatasetModeType::MODE_TRAIN, dataset);
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
      if (loraTrainCancelledFlag().load())
        throw std::runtime_error("training cancelled by user");
      if (bridge.iter_cb)
        bridge.iter_cb(iteration, bridge.epoch + 1, loss, bridge.iter_data);
    };

    model.train(onEpochComplete, &bridge, nullptr, nullptr, iteration_wrapper,
                nullptr);

    std::cout << "training complete; best valid_loss=" << bridge.best_loss
              << ", adapter at " << output_path << std::endl;
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
      else if (arg == "--seed")
        params.seed = static_cast<unsigned int>(std::stoul(next("--seed")));
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
