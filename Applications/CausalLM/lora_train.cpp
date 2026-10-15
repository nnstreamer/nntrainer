// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   lora_train.cpp
 * @date   31 July 2026
 * @brief  Text-file training data generator for Qwen3 LoRA fine-tuning.
 * @bug    No known bugs except for NYI items
 */
#include <lora_train.h>

#include <decision.h>

#include <nntrainer_error.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <stdexcept>

namespace causallm {

namespace {

/**
 * @brief Insert Qwen3's think-disable marker after the last assistant turn,
 *        unless the sample already contains a <think> block.
 *
 * @details Qwen3 is a hybrid thinking/non-thinking model: its own chat
 * template inserts "<think>\n\n</think>\n\n" right after
 * "<|im_start|>assistant\n" whenever generation is requested with
 * enable_thinking=false (see tokenizer_config.json's chat_template). Without
 * that marker, the model's own strong prior is to begin a real <think>...
 * </think> reasoning block, NOT to answer directly.
 *
 * Plain-text training samples that go straight from
 * "<|im_start|>assistant\n" to a short direct answer (e.g. a LAMP-3-style
 * sentiment label) omit this marker, so the label token is wildly
 * out-of-distribution for what the base model actually wants to emit next —
 * this alone can keep training loss stuck 15+ nats above what the same data
 * gives once the marker is restored (empirically confirmed: ~29 -> ~13 on
 * lamp3_user_train.txt with no other change). Insert it here so plain
 * short-answer training data doesn't have to be hand-edited to match the
 * chat template's own convention. Samples that already contain a <think>
 * block (deliberate reasoning-mode training) are left untouched.
 */
std::string ensureThinkDisableMarker(const std::string &sample) {
  if (sample.find("<think>") != std::string::npos)
    return sample;

  static const std::string assistant_marker = "<|im_start|>assistant\n";
  static const std::string think_disable = "<think>\n\n</think>\n\n";

  size_t pos = sample.rfind(assistant_marker);
  if (pos == std::string::npos)
    return sample;

  std::string out = sample;
  out.insert(pos + assistant_marker.size(), think_disable);
  return out;
}

} // namespace

TrainingDataGenerator::TrainingDataGenerator(const std::string &data_path,
                                             tokenizers::Tokenizer *tokenizer,
                                             unsigned int seq_len,
                                             unsigned int vocab_size,
                                             unsigned int max_samples,
                                             unsigned int seed, bool decision) :
  seq_len_(seq_len), vocab_size_(vocab_size), rng_(seed) {
  if (decision) {
    // Auto-detect the personalization package's pre-tokenized device_io
    // format ({"input_ids": [...], "target_probs": [...]}): consuming its
    // ids verbatim avoids any tokenizer drift between the package and the
    // device model.
    bool pretokenized = false;
    {
      std::ifstream peek(data_path);
      std::string first;
      while (first.empty() && std::getline(peek, first))
        if (first.find_first_not_of(" \t\r\n") == std::string::npos)
          first.clear();
      pretokenized = first.find("\"input_ids\"") != std::string::npos;
    }
    if (pretokenized)
      loadPreTokenizedFile(data_path, max_samples);
    else
      loadDecisionFile(data_path, tokenizer, max_samples);
  } else
    loadTextFile(data_path, tokenizer, max_samples);
  index_order_.resize(samples_.size());
  std::iota(index_order_.begin(), index_order_.end(), 0);
  reshuffle();
}

void TrainingDataGenerator::loadDecisionFile(const std::string &path,
                                             tokenizers::Tokenizer *tokenizer,
                                             unsigned int max_samples) {
  for (const auto &q : readDecisionQuestions(path)) {
    if (max_samples && samples_.size() >= max_samples)
      break;
    if (q.answer < 0)
      throw std::invalid_argument(
        "decision training needs an answer on every question: " + path);
    Sample s;
    s.ids = encodeDecisionPrompt(tokenizer, q, seq_len_);
    s.k = static_cast<unsigned int>(q.choices.size());
    s.answer = static_cast<unsigned int>(q.answer);
    samples_.push_back(std::move(s));
  }
  if (samples_.empty())
    throw std::runtime_error("No usable decision questions found in " + path);
}

void TrainingDataGenerator::loadPreTokenizedFile(const std::string &path,
                                                 unsigned int max_samples) {
  std::ifstream file(path);
  if (!file.is_open())
    throw std::runtime_error("Failed to open pre-tokenized data file: " + path);

  std::string line;
  for (size_t n = 1; std::getline(file, line); ++n) {
    if (line.find_first_not_of(" \t\r\n") == std::string::npos)
      continue;
    if (max_samples && samples_.size() >= max_samples)
      break;
    try {
      const auto j = nlohmann::json::parse(line);
      NNTR_THROW_IF(!j.contains("input_ids") || !j["input_ids"].is_array(),
                    std::invalid_argument)
        << "input_ids (array) is required";
      NNTR_THROW_IF(!j.contains("target_probs") ||
                      !j["target_probs"].is_array(),
                    std::invalid_argument)
        << "target_probs (array) is required; rows without targets (e.g. the "
           "package's *_inputs.jsonl) cannot be trained on -- merge the "
           "targets in first";

      Sample s;
      for (const auto &t : j["input_ids"]) {
        const int64_t id = t.get<int64_t>();
        NNTR_THROW_IF(id < 0 || id >= static_cast<int64_t>(vocab_size_),
                      std::invalid_argument)
          << "token id " << id << " outside the vocabulary";
        s.ids.push_back(static_cast<int32_t>(id));
      }
      // the readout sits at the last prompt token; when the row exceeds the
      // sequence budget keep its tail, as encodeDecisionPrompt does
      if (s.ids.size() > seq_len_)
        s.ids.erase(s.ids.begin(), s.ids.end() - seq_len_);

      for (const auto &p : j["target_probs"])
        s.probs.push_back(p.get<float>());
      s.k = static_cast<unsigned int>(s.probs.size());
      NNTR_THROW_IF(s.k < 2 || s.k > DECISION_MAX_CHOICES,
                    std::invalid_argument)
        << "target_probs has " << s.k << " entries, need 2.."
        << DECISION_MAX_CHOICES;
      double tsum = 0.0;
      for (float p : s.probs)
        tsum += p;
      NNTR_THROW_IF(tsum <= 0.0, std::invalid_argument)
        << "target_probs sums to " << tsum;
      if (j.contains("primary_label_index"))
        s.answer =
          static_cast<unsigned int>(j["primary_label_index"].get<int>());
      NNTR_THROW_IF(s.answer >= s.k, std::invalid_argument)
        << "primary_label_index " << s.answer << " out of range for " << s.k
        << " options";

      samples_.push_back(std::move(s));
    } catch (const std::exception &e) {
      throw std::invalid_argument(path + ":" + std::to_string(n) + ": " +
                                  e.what());
    }
  }
  if (samples_.empty())
    throw std::runtime_error("No usable pre-tokenized samples found in " +
                             path);
}

void TrainingDataGenerator::loadTextFile(const std::string &path,
                                         tokenizers::Tokenizer *tokenizer,
                                         unsigned int max_samples) {
  std::ifstream file(path);
  if (!file.is_open())
    throw std::runtime_error("Failed to open training data file: " + path);

  std::vector<std::string> lines;
  std::string line;
  while (std::getline(file, line)) {
    if (!line.empty())
      lines.push_back(line);
  }

  bool chat_format = false;
  for (const auto &l : lines) {
    if (l.find("<|im_start|>user") != std::string::npos) {
      chat_format = true;
      break;
    }
  }

  std::vector<std::string> raw_samples;
  if (chat_format) {
    std::string current;
    bool started = false;
    for (const auto &l : lines) {
      if (l.find("<|im_start|>user") != std::string::npos) {
        if (started && !current.empty())
          raw_samples.push_back(ensureThinkDisableMarker(current));
        current = l;
        started = true;
      } else if (started) {
        current += "\n" + l;
      }
    }
    if (started && !current.empty())
      raw_samples.push_back(ensureThinkDisableMarker(current));
  } else {
    raw_samples = lines;
  }

  for (const auto &s : raw_samples) {
    if (max_samples && samples_.size() >= max_samples)
      break;

    auto ids = tokenizer->Encode(s);
    // Need at least one input token plus the label token.
    if (ids.size() < 2)
      continue;

    // Input (everything but the last token) must fit within seq_len_; if
    // the tokenized sample is longer, keep the most recent seq_len_+1
    // tokens so the label (the true last token) is preserved.
    if (ids.size() > seq_len_ + 1)
      ids = std::vector<int32_t>(ids.end() - (seq_len_ + 1), ids.end());

    samples_.push_back({std::move(ids)});
  }

  if (samples_.empty())
    throw std::runtime_error("No usable training samples found in " + path);
}

void TrainingDataGenerator::setBatchSize(unsigned int batch_size) {
  batch_size_ = batch_size ? batch_size : 1;
  NNTR_THROW_IF(samplesPerEpoch() == 0, std::invalid_argument)
    << "batch_size " << batch_size_ << " exceeds the " << samples_.size()
    << " available training samples";
}

size_t TrainingDataGenerator::samplesPerEpoch() const {
  return samples_.size() / batch_size_ * batch_size_;
}

unsigned int TrainingDataGenerator::longestInput() const {
  size_t longest = 0;
  for (const auto &s : samples_)
    longest = std::max(longest, s.ids.size() - (s.k > 0 ? 0 : 1));
  return static_cast<unsigned int>(longest);
}

void TrainingDataGenerator::setSeqLen(unsigned int seq_len) {
  NNTR_THROW_IF(seq_len < longestInput(), std::invalid_argument)
    << "seq_len " << seq_len << " is shorter than the longest sample ("
    << longestInput() << " rows)";
  seq_len_ = seq_len;
}

void TrainingDataGenerator::reshuffle() {
  std::shuffle(index_order_.begin(), index_order_.end(), rng_);
  cursor_ = 0;
}

int TrainingDataGenerator::next(float **input, float **label, bool *last) {
  const Sample &s = samples_[index_order_[cursor_]];
  // A text sample's last token is only a target; a decision prompt is read
  // whole, the readout at its last token.
  const bool decision = s.k > 0;
  const unsigned int used =
    static_cast<unsigned int>(s.ids.size() - (decision ? 0 : 1));

  /// [DIAG] NNTR_TRAIN_GRAD_DUMP=<dir>: record each produced sample's ids
  if (const char *dump = std::getenv("NNTR_TRAIN_GRAD_DUMP")) {
    static int produced = 0;
    if (produced++ < 16) {
      if (FILE *f =
            std::fopen((std::string(dump) + "/samples.txt").c_str(), "a")) {
        for (size_t i = 0; i < s.ids.size(); ++i)
          std::fprintf(f, i ? " %d" : "%d", s.ids[i]);
        std::fprintf(f, "\n");
        std::fclose(f);
      }
    }
  }
  if (input_writer) {
    input_writer(s.ids.data(), used, input[0]);
  } else {
    float *in = input[0];
    std::fill(in, in + seq_len_, 0.0f);
    for (unsigned int i = 0; i < used; ++i)
      in[i] = static_cast<float>(s.ids[i]);
  }

  float *lbl = label[0];
  if (decision) {
    if (!s.probs.empty()) {
      // soft targets: the declared options carry the target distribution,
      // -1 (masked) beyond them, then the readout row
      for (unsigned int i = 0; i < DECISION_MAX_CHOICES; ++i)
        lbl[i] = i >= s.k ? -1.0f : s.probs[i];
    } else {
      // decision_head_loss's label: per letter 1 (answer) / 0 (another
      // option) / -1 (beyond the options), then the readout row
      for (unsigned int i = 0; i < DECISION_MAX_CHOICES; ++i)
        lbl[i] = i >= s.k ? -1.0f : (i == s.answer ? 1.0f : 0.0f);
    }
    lbl[DECISION_MAX_CHOICES] = static_cast<float>(used - 1);
  } else {
    // Next-token targets: position i (input token ids[i]) is scored against
    // ids[i + 1]. The padding after the sample is not scored (-1).
    std::fill(lbl, lbl + seq_len_, -1.0f);
    for (unsigned int i = 0; i < used; ++i) {
      const int32_t t = s.ids[i + 1];
      if (t >= 0 && static_cast<unsigned int>(t) < vocab_size_)
        lbl[i] = static_cast<float>(t);
    }
  }

  ++cursor_;
  const bool epoch_done = cursor_ >= samplesPerEpoch();
  if (epoch_done)
    reshuffle();
  *last = epoch_done;

  return ML_ERROR_NONE;
}

int trainingDataGenCb(float **input, float **label, bool *last,
                      void *user_data) {
  return static_cast<TrainingDataGenerator *>(user_data)->next(input, label,
                                                               last);
}

std::atomic<bool> &loraTrainCancelledFlag() {
  static std::atomic<bool> flag{false};
  return flag;
}

} // namespace causallm
