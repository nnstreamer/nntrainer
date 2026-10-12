// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   decision.cpp
 * @date   12 October 2026
 * @brief  Decision-model questions, prompt and readout
 * @bug    No known bugs except for NYI items
 */
#include <decision.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <set>
#include <stdexcept>

namespace causallm {

namespace {

std::string strip(const std::string &s) {
  const char *ws = " \t\r\n";
  const size_t b = s.find_first_not_of(ws);
  if (b == std::string::npos)
    return "";
  return s.substr(b, s.find_last_not_of(ws) - b + 1);
}

} // namespace

DecisionQuestion DecisionQuestion::fromJson(const nlohmann::json &j) {
  DecisionQuestion q;
  q.primitive = j.value("primitive", std::string("choice"));
  q.context = j.at("context").get<std::string>();
  q.question = j.at("question").get<std::string>();
  q.choices = j.at("choices").get<std::vector<std::string>>();
  if (j.contains("answer") && !j["answer"].is_null())
    q.answer = j["answer"].get<int>();

  if (q.primitive != "choice" && q.primitive != "score" &&
      q.primitive != "noul")
    throw std::invalid_argument("unknown primitive '" + q.primitive + "'");
  const size_t k = q.choices.size();
  if (k < 2 || k > DECISION_MAX_CHOICES)
    throw std::invalid_argument("need 2..26 choices, got " + std::to_string(k));
  std::set<std::string> seen;
  for (const auto &c : q.choices) {
    if (strip(c).empty() || c.find('\n') != std::string::npos ||
        c.find('\r') != std::string::npos)
      throw std::invalid_argument("choices must be non-empty single lines");
    if (!seen.insert(strip(c)).second)
      throw std::invalid_argument("duplicate choice '" + c + "'");
  }
  if (q.primitive == "noul" &&
      q.choices != std::vector<std::string>{"true", "false"})
    throw std::invalid_argument("noul choices must be [\"true\", \"false\"]");
  if (q.answer >= static_cast<int>(k))
    throw std::invalid_argument("answer " + std::to_string(q.answer) +
                                " out of range for " + std::to_string(k) +
                                " choices");
  return q;
}

std::vector<DecisionQuestion> readDecisionQuestions(const std::string &path) {
  std::ifstream f(path);
  if (!f.is_open())
    throw std::runtime_error("cannot open questions file " + path);
  std::vector<DecisionQuestion> out;
  std::string line;
  for (size_t n = 1; std::getline(f, line); ++n) {
    if (strip(line).empty())
      continue;
    try {
      out.push_back(DecisionQuestion::fromJson(nlohmann::json::parse(line)));
    } catch (const std::exception &e) {
      throw std::invalid_argument(path + ":" + std::to_string(n) + ": " +
                                  e.what());
    }
  }
  return out;
}

std::vector<int32_t> decisionLetterIds(const std::string &model_dir,
                                       tokenizers::Tokenizer *tokenizer) {
  std::vector<int32_t> ids;
  const auto meta = std::filesystem::path(model_dir) / "decision.json";
  if (std::filesystem::exists(meta)) {
    std::ifstream f(meta);
    ids = nlohmann::json::parse(f).at("letter_ids").get<std::vector<int32_t>>();
  } else {
    for (unsigned int i = 0; i < DECISION_MAX_CHOICES; ++i) {
      const auto t = tokenizer->Encode(std::string(" ") + char('A' + i), false);
      if (t.size() != 1)
        throw std::runtime_error(std::string("' ") + char('A' + i) +
                                 "' is not a single token");
      ids.push_back(t[0]);
    }
  }
  if (ids.size() != DECISION_MAX_CHOICES)
    throw std::runtime_error("a decision model needs 26 letter ids, got " +
                             std::to_string(ids.size()));
  return ids;
}

std::string renderDecisionPrompt(const DecisionQuestion &q) {
  std::string s = "User: Context:\n" + strip(q.context) + "\n\n" +
                  "Question: " + strip(q.question) + "\nOptions:\n";
  for (size_t i = 0; i < q.choices.size(); ++i)
    s += std::string(1, char('A' + i)) + ") " + q.choices[i] + "\n";
  s += "Answer with the letter only.\nAssistant: The answer is";
  return s;
}

std::vector<int32_t> encodeDecisionPrompt(tokenizers::Tokenizer *tokenizer,
                                          const DecisionQuestion &q,
                                          unsigned int max_tokens) {
  std::vector<int32_t> ids = tokenizer->Encode(renderDecisionPrompt(q), false);
  if (ids.size() > max_tokens)
    ids.erase(ids.begin(), ids.end() - max_tokens);
  return ids;
}

nlohmann::json decisionAnswer(const DecisionQuestion &q,
                              const std::vector<float> &probs) {
  const size_t k = q.choices.size();
  size_t best = 0;
  double entropy = 0.0;
  nlohmann::json p = nlohmann::json::object();
  for (size_t i = 0; i < k; ++i) {
    if (probs[i] > probs[best])
      best = i;
    if (probs[i] > 0.0f)
      entropy -= probs[i] * std::log((double)probs[i]);
    p[q.choices[i]] = probs[i];
  }
  nlohmann::json a = {
    {"kind", q.primitive},
    {"value", q.choices[best]},
    {"probs", p},
    {"confidence", probs[best]},
    {"entropy_confidence", 1.0 - entropy / std::log((double)k)}};
  if (q.primitive == "score") {
    double s = 0.0;
    for (size_t i = 0; i < k; ++i)
      s += probs[i] * (double)i;
    a["score"] = s / (double)(k - 1);
  } else if (q.primitive == "noul") {
    a = {{"kind", "noul"},
         {"p_true", probs[0]},
         {"confidence", std::max(probs[0], 1.0f - probs[0])}};
  }
  return a;
}

} // namespace causallm
