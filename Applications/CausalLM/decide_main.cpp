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
 *          A trained adapter is applied when nntr_config.json names it
 *          (lora_file_name / lora_q4_file_name, as for nntr_causallm).
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
#include <filesystem>
#include <iostream>
#include <string>

int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cerr << "usage: " << argv[0]
              << " <model_dir> <questions.jsonl> [--max_len N]\n"
                 "  --max_len N  prompt token budget, cut from the left "
                 "(default: init_seq_len)\n";
    return EXIT_FAILURE;
  }
  const std::string model_path = argv[1];
  const std::string questions_path = argv[2];
  unsigned int max_len = 0;
  for (int i = 3; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--max_len" && i + 1 < argc)
      max_len = static_cast<unsigned int>(std::stoul(argv[++i]));
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

    causallm::Qwen3CausalLM model(cfg, gen, nntr_cfg);
    model.initialize();
    // a trained adapter loads the way nntr_causallm (main.cpp) loads it
    const std::string weight_file =
      model_path + "/" + nntr_cfg["model_file_name"].get<std::string>();
    if (!nntr_cfg.value("lora_q4_file_name", std::string()).empty())
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
