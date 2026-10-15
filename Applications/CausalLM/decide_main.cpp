// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   decide_main.cpp
 * @date   12 October 2026
 * @brief  nntr_decide: answer decision questions with a decision model
 * @bug    No known bugs except for NYI items
 *
 * @details nntr_decide <model_dir> <questions.jsonl> [--max_len N]
 *                     [--lora <adapter> [--lora_rank N] [--lora_alpha N]
 *                      [--lora_last_n N]]
 *          A trained adapter is applied when nntr_config.json names it
 *          (lora_file_name / lora_q4_file_name, as for nntr_causallm), or
 *          when --lora names one on the command line (FP32 and qint4
 *          adapter files are told apart by their file magic). A command
 *          line adapter needs the graph to have LoRA capacity: --lora_rank
 *          (and usually --lora_last_n) must repeat the adapter's training
 *          config, because a model pack ships with LoRA disabled
 *          (lora_rank = 0) and the FP32 adapter file carries no header to
 *          recover it from.
 *          For every question (an eve-rlcd JSONL row, see decision.h) runs
 *          one prefill of the decision prompt and prints one JSON line:
 *          {"tokens", "value", "probs", "confidence", "entropy_confidence"
 *          [, "score" | "p_true"], "ms"}.
 */
#include <decision.h>
#include <qwen3_causallm.h>
#include <transformer.h>

#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cerr << "usage: " << argv[0]
              << " <model_dir> <questions.jsonl> [--max_len N] "
                 "[--lora <adapter>]\n"
                 "  --max_len N  prompt token budget, cut from the left "
                 "(default: init_seq_len)\n"
                 "  --lora F     apply this LoRA adapter (FP32 or qint4,\n"
                 "               picked by the file magic) instead of the\n"
                 "               one nntr_config.json names\n"
                 "  --lora_rank N     adapter's training rank (needed for\n"
                 "                    --lora; builds LoRA into the graph)\n"
                 "  --lora_alpha N    adapter's training alpha\n"
                 "                    (default: rank * 2)\n"
                 "  --lora_last_n N   adapter's trained block count\n";
    return EXIT_FAILURE;
  }
  const std::string model_path = argv[1];
  const std::string questions_path = argv[2];
  unsigned int max_len = 0;
  std::string lora_path;
  unsigned int lora_rank = 0, lora_alpha = 0, lora_last_n = 0;
  for (int i = 3; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--max_len" && i + 1 < argc)
      max_len = static_cast<unsigned int>(std::stoul(argv[++i]));
    else if (arg == "--lora" && i + 1 < argc)
      lora_path = argv[++i];
    else if (arg == "--lora_rank" && i + 1 < argc)
      lora_rank = static_cast<unsigned int>(std::stoul(argv[++i]));
    else if (arg == "--lora_alpha" && i + 1 < argc)
      lora_alpha = static_cast<unsigned int>(std::stoul(argv[++i]));
    else if (arg == "--lora_last_n" && i + 1 < argc)
      lora_last_n = static_cast<unsigned int>(std::stoul(argv[++i]));
    else {
      std::cerr << "unknown argument " << arg << "\n";
      return EXIT_FAILURE;
    }
  }

  try {
    causallm::json cfg = causallm::LoadJsonFile(model_path + "/config.json");
    causallm::json gen = causallm::json::object();
    if (std::filesystem::exists(model_path + "/generation_config.json"))
      gen = causallm::LoadJsonFile(model_path + "/generation_config.json");
    causallm::json nntr_cfg =
      causallm::LoadJsonFile(model_path + "/nntr_config.json");
    std::filesystem::path tok =
      nntr_cfg.value("tokenizer_file", std::string("tokenizer.json"));
    if (tok.is_relative() || !std::filesystem::exists(tok))
      tok = std::filesystem::path(model_path) / tok.filename();
    nntr_cfg["tokenizer_file"] = tok.string();
    if (max_len == 0)
      max_len = nntr_cfg["init_seq_len"].get<unsigned int>();

    const auto questions = causallm::readDecisionQuestions(questions_path);

    // LoRA capacity must exist in the graph BEFORE the model is created: an
    // adapter overlays onto loraA/loraB slots that are only built when
    // lora_rank > 0. A model pack ships with LoRA disabled (lora_rank = 0),
    // and the FP32 adapter file has no header to recover the training
    // config from, so --lora without --lora_rank would silently load
    // nothing (every question answered as if by the base model).
    if (!lora_path.empty()) {
      if (lora_rank == 0) {
        std::cerr << "--lora requires --lora_rank (and usually "
                     "--lora_last_n) repeating the adapter's training "
                     "config; the FP32 adapter file has no header and the "
                     "pack ships with lora_rank = 0.\n";
        return EXIT_FAILURE;
      }
      nntr_cfg["lora_rank"] = lora_rank;
      nntr_cfg["lora_alpha"] = lora_alpha ? lora_alpha : lora_rank * 2;
      if (lora_last_n)
        nntr_cfg["lora_last_n_layers"] = lora_last_n;
      if (!nntr_cfg.contains("lora_target") || nntr_cfg["lora_target"].empty())
        nntr_cfg["lora_target"] = std::vector<std::string>{
          "wq",           "wk",         "wv",
          "wo",           "ffn_up",     "ffn_gate",
          "ffn_down",     "wgate",      "linattn_in_qkv",
          "linattn_in_z", "linattn_out"};
      // The adapter graph runs on the cpu engine (setForceCpuEngine()
      // below), but mha_core's internal GPU arms gate on the
      // use_gemm_attention LAYER property: with the base model's
      // use_flash_attention=true they still open on this cpu graph and the
      // OpenCL kernels fail on host-produced operands (rc=3). Close them
      // for adapter loads so attention takes the host per-row path, the
      // same one training uses.
      nntr_cfg["use_flash_attention"] = false;
    }

    causallm::Qwen3CausalLM model(cfg, gen, nntr_cfg);
    // LoRA adapter inference runs the CPU W4A8 GEMM path (the OpenCL FC
    // layer has no LoRA support): build the graph on the cpu engine,
    // otherwise the gpu FC rejects the lora_* properties and the graph
    // fails to initialize.
    if (!lora_path.empty())
      model.setForceCpuEngine(true);
    model.initialize();
    // a trained adapter loads the way nntr_causallm (main.cpp) loads it;
    // --lora wins over the nntr_config.json entry, and the loader is picked
    // by the adapter file's magic ("NQI4" = qint4, anything else = FP32)
    const std::string weight_file =
      model_path + "/" + nntr_cfg["model_file_name"].get<std::string>();
    if (!lora_path.empty()) {
      std::ifstream f(lora_path, std::ios::binary);
      char magic[4] = {};
      f.read(magic, 4);
      if (f && std::memcmp(magic, "NQI4", 4) == 0)
        model.load_weight_lora_qint4(weight_file, lora_path);
      else
        model.load_weight_lora(weight_file, lora_path);
      std::cout << "adapter:      " << lora_path << std::endl;
    } else if (!nntr_cfg.value("lora_q4_file_name", std::string()).empty())
      model.load_weight_lora_qint4(
        weight_file,
        model_path + "/" + nntr_cfg["lora_q4_file_name"].get<std::string>());
    else if (!nntr_cfg.value("lora_file_name", std::string()).empty())
      model.load_weight_lora(weight_file,
                             model_path + "/" +
                               nntr_cfg["lora_file_name"].get<std::string>());
    else
      model.load_weight(weight_file);
    model.repack_weight();
    auto *tokenizer = model.getTokenizer();
    const auto letters = causallm::decisionLetterIds(model_path, tokenizer);

    for (const auto &q : questions) {
      const auto t0 = std::chrono::steady_clock::now();
      const auto ids = causallm::encodeDecisionPrompt(tokenizer, q, max_len);
      const std::vector<float> logits = model.promptLogits(ids);

      // softmax over the letters of the declared options only
      const size_t k = q.choices.size();
      std::vector<float> p(k);
      float mx = -INFINITY;
      for (size_t i = 0; i < k; ++i)
        mx = std::max(mx, logits[letters[i]]);
      double sum = 0.0;
      for (size_t i = 0; i < k; ++i)
        sum += std::exp((double)logits[letters[i]] - mx);
      for (size_t i = 0; i < k; ++i)
        p[i] =
          static_cast<float>(std::exp((double)logits[letters[i]] - mx) / sum);

      auto a = causallm::decisionAnswer(q, p);
      a["tokens"] = ids.size();
      a["ms"] = std::chrono::duration<double, std::milli>(
                  std::chrono::steady_clock::now() - t0)
                  .count();
      if (q.answer >= 0)
        a["correct"] = q.choices[q.answer];
      std::cout << a.dump() << std::endl;
    }
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
