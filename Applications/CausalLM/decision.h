// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   decision.h
 * @date   12 October 2026
 * @brief  Decision-model (rlcd-decision-only-v1) questions, prompt and readout
 * @see    https://huggingface.co/anthonym21/qwen3-0.6b-rlcd-decision
 * @bug    No known bugs except for NYI items
 *
 * @details A decision model reads one context and a question with 2..26
 *          options, and answers with a probability per option: the softmax
 *          over the logits of the letter tokens " A".." Z" at the last prompt
 *          token, cut to the first K letters (K = option count). Nothing is
 *          generated. The prompt is the eve-rlcd one:
 *
 *            User: Context:
 *            <context>
 *
 *            Question: <question>
 *            Options:
 *            A) <option>
 *            B) <option>
 *            Answer with the letter only.
 *            Assistant: The answer is
 *
 *          tokenized whole and cut from the left to a token budget, as the
 *          model's training did.
 */
#ifndef __CAUSALLM_DECISION_H__
#define __CAUSALLM_DECISION_H__

#include <json.hpp>
#include <tokenizers_cpp.h>

#include <cstdint>
#include <string>
#include <vector>

namespace causallm {

/** @brief The most options a question can declare (one letter each). */
constexpr unsigned int DECISION_MAX_CHOICES = 26;

/**
 * @brief One decision question: an eve-rlcd JSONL row
 *        {"primitive", "context", "question", "choices", "answer"}.
 */
struct DecisionQuestion {
  std::string primitive = "choice"; ///< choice | score | noul
  std::string context;
  std::string question;
  std::vector<std::string> choices;
  int answer = -1; ///< index of the correct option, -1 if unknown

  /** @brief Parse and validate one JSONL row (throws on a bad row). */
  static DecisionQuestion fromJson(const nlohmann::json &j);
};

/** @brief Read a JSONL file of questions, one per non-empty line. */
std::vector<DecisionQuestion> readDecisionQuestions(const std::string &path);

/**
 * @brief The letter token ids " A".." Z" of a decision model: from its
 *        decision.json when @a model_dir has one, else from @a tokenizer
 *        (each " X" must be a single token).
 */
std::vector<int32_t> decisionLetterIds(const std::string &model_dir,
                                       tokenizers::Tokenizer *tokenizer);

/** @brief The decision prompt of @a q (see the file comment). */
std::string renderDecisionPrompt(const DecisionQuestion &q);

/**
 * @brief Token ids of the decision prompt, cut from the left to at most
 *        @a max_tokens so the options and the footer always survive.
 */
std::vector<int32_t> encodeDecisionPrompt(tokenizers::Tokenizer *tokenizer,
                                          const DecisionQuestion &q,
                                          unsigned int max_tokens);

/**
 * @brief The answer of @a q given the probabilities of its options:
 *        value, probs, confidence, entropy_confidence, and score (score
 *        questions) or p_true (noul questions).
 */
nlohmann::json decisionAnswer(const DecisionQuestion &q,
                              const std::vector<float> &probs);

} // namespace causallm

#endif // __CAUSALLM_DECISION_H__
