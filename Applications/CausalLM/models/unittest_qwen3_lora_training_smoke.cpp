// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   unittest_qwen3_lora_training_smoke.cpp
 * @date   31 July 2026
 * @brief  End-to-end smoke test for LoRA training wiring: builds a tiny
 *         synthetic Qwen3CausalLM, compiles it for training with LoRA
 *         enabled, runs a few training steps on synthetic data, and checks
 *         that the LoRA adapter save/load round-trips.
 * @bug    No known bugs except for NYI items
 */

#include <lora_train.h>
#include <qwen3_causallm.h>

#include <dataset.h>
#include <layer_context.h>
#include <model.h>

#include <fc_layer.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <functional>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

causallm::json makeTinyQwen3Config() {
  return {
    {"architectures", {"Qwen3ForCausalLM"}},
    {"bos_token_id", 0},
    {"eos_token_id", {31}},
    {"head_dim", 8},
    {"hidden_size", 64},
    {"intermediate_size", 64},
    {"is_causal", true},
    {"max_position_embeddings", 8},
    {"num_attention_heads", 8},
    {"num_hidden_layers", 2},
    {"num_key_value_heads", 4},
    {"rms_norm_eps", 1e-5},
    {"rope_theta", 10000},
    {"tie_word_embeddings", true},
    {"vocab_size", 32},
  };
}

causallm::json makeTinyGenerationConfig() {
  return {
    {"bos_token_id", 0}, {"eos_token_id", 31}, {"do_sample", false},
    {"top_k", 1},        {"top_p", 1.0},       {"temperature", 1.0},
  };
}

causallm::json makeTinyNntrainerConfig(unsigned int lora_rank,
                                       unsigned int batch_size = 1) {
  return {
    {"bad_word_ids", std::vector<unsigned int>{}},
    {"batch_size", batch_size},
    {"embedding_dtype", "FP32"},
    {"fc_layer_dtype", "FP32"},
    {"init_seq_len", 4},
    {"lmhead_dtype", "FP32"},
    {"max_seq_len", 8},
    {"model_tensor_type", "FP32-FP32"},
    {"model_type", "CausalLM"},
    {"num_to_generate", 1},
    {"skip_tokenizer", true},
    {"lora_rank", lora_rank},
    {"lora_alpha", lora_rank * 2},
    {"lora_target", std::vector<std::string>{"wq", "wk", "wv", "wo", "ffn_up",
                                             "ffn_gate", "ffn_down"}},
  };
}

/** @brief Synthetic in-memory next-token-classification generator matching
 *  the training graph's lm_head_loss: a fixed-length input of token ids and
 *  a next-token id for every position. */
struct SyntheticDataGen {
  unsigned int seq_len;
  unsigned int vocab_size;
  std::mt19937 rng{42};

  int operator()(float **input, float **label, bool *last) {
    std::uniform_int_distribution<unsigned int> tok(0, vocab_size - 1);
    for (unsigned int i = 0; i < seq_len; ++i)
      input[0][i] = static_cast<float>(tok(rng));

    for (unsigned int i = 0; i < seq_len; ++i)
      label[0][i] = static_cast<float>(tok(rng));

    *last = true;
    return ML_ERROR_NONE;
  }
};

int datasetCb(float **input, float **label, bool *last, void *user_data) {
  auto *gen = static_cast<SyntheticDataGen *>(user_data);
  return (*gen)(input, label, last);
}

} // namespace

TEST(Qwen3LoraTrainingSmoke, InitializeForTrainingCompilesWithLoraEnabled) {
  auto cfg = makeTinyQwen3Config();
  auto generation_cfg = makeTinyGenerationConfig();
  auto nntr_cfg = makeTinyNntrainerConfig(/*lora_rank=*/4);

  causallm::Qwen3CausalLM model(cfg, generation_cfg, nntr_cfg);
  ASSERT_NO_THROW(model.initializeForTraining(/*lr=*/1e-3f, /*epochs=*/1));
}

TEST(Qwen3LoraTrainingSmoke, TrainableFlagsMatchLoraFreezePolicy) {
  auto cfg = makeTinyQwen3Config();
  auto generation_cfg = makeTinyGenerationConfig();
  auto nntr_cfg = makeTinyNntrainerConfig(/*lora_rank=*/4);

  causallm::Qwen3CausalLM model(cfg, generation_cfg, nntr_cfg);
  ASSERT_NO_THROW(model.initializeForTraining(1e-3f, 1));

  std::vector<std::string> trainable_layers, frozen_layers;
  std::function<void(ml::train::Layer &, nntrainer::RunLayerContext &, void *)>
    fn = [&](ml::train::Layer &l, nntrainer::RunLayerContext &context, void *) {
      // A layer is "trainable" here iff it has at least one weight with a
      // populated gradient buffer, which only happens when the layer-level
      // trainable flag AND the weight's own need_gradient flag are both true.
      // Layers with zero weights (input, residual "addition", the loss
      // layer) are trivially frozen and not part of this check at all.
      if (context.getNumWeights() == 0)
        return;
      bool has_grad = false;
      for (unsigned int i = 0; i < context.getNumWeights(); ++i) {
        if (context.weightHasGradient(i)) {
          has_grad = true;
          break;
        }
      }
      (has_grad ? trainable_layers : frozen_layers).push_back(l.getName());
    };
  model.forEachLayer(fn, nullptr);

  // wq/wk/wv/wo/ffn_up/ffn_gate/ffn_down are LoRA targets -> trainable
  // (their loraA/loraB weights have gradients; the base weight does not).
  bool found_lora_fc = false;
  for (const auto &name : trainable_layers) {
    if (name.find("_wq") != std::string::npos ||
        name.find("_ffn_up") != std::string::npos) {
      found_lora_fc = true;
    }
  }
  EXPECT_TRUE(found_lora_fc)
    << "expected at least one LoRA-targeted FC layer to have a trainable "
       "(loraA/loraB) weight";

  // norms, embedding, mha_core, lm_head must NOT be trainable.
  for (const auto &name : frozen_layers) {
    EXPECT_TRUE(
      name.find("_norm") != std::string::npos || name == "embedding0" ||
      name.find("_attention") != std::string::npos ||
      name == "output_of_causallm" || name.find("_swiglu") != std::string::npos)
      << "unexpected frozen layer: " << name;
  }
}

/** @brief Read every FP32 weight's first element, keyed by weight name. */
std::map<std::string, float> snapshotFirstElements(causallm::Qwen3CausalLM &m) {
  std::map<std::string, float> out;
  std::function<void(ml::train::Layer &, nntrainer::RunLayerContext &, void *)>
    fn =
      [&out](ml::train::Layer &, nntrainer::RunLayerContext &context, void *) {
        for (unsigned int i = 0; i < context.getNumWeights(); ++i) {
          auto &w = context.getWeight(i);
          if (w.getDataType() == ml::train::TensorDim::DataType::FP32)
            out[w.getName()] = w.getData<float>()[0];
        }
      };
  m.forEachLayer(fn, nullptr);
  return out;
}

TEST(Qwen3LoraTrainingSmoke, TrainsAFewStepsAndSaveLoadLoraRoundTrips) {
  auto cfg = makeTinyQwen3Config();
  auto generation_cfg = makeTinyGenerationConfig();

  // Step 1: build a plain (no-LoRA) model, initialize (random init, no
  // pretrained checkpoint available in this synthetic test), and save it as
  // the "base checkpoint" that load_weight_lora() will read back.
  auto base_nntr_cfg = makeTinyNntrainerConfig(/*lora_rank=*/0);
  causallm::Qwen3CausalLM base_model(cfg, generation_cfg, base_nntr_cfg);
  ASSERT_NO_THROW(base_model.initialize());
  const auto base_path =
    (std::filesystem::temp_directory_path() / "nntrainer_qwen3_base_smoke.bin")
      .string();
  ASSERT_NO_THROW(base_model.save_weight(base_path));
  auto base_snapshot = snapshotFirstElements(base_model);

  // Step 2: build the LoRA training model, train a few steps so loraA/loraB
  // move away from their initial (zero/LeCun-normal) values, then save just
  // the LoRA adapter.
  auto lora_nntr_cfg = makeTinyNntrainerConfig(/*lora_rank=*/4);
  causallm::Qwen3CausalLM lora_model(cfg, generation_cfg, lora_nntr_cfg);
  ASSERT_NO_THROW(lora_model.initializeForTraining(/*lr=*/1e-2f, /*epochs=*/1));

  SyntheticDataGen gen{/*seq_len=*/4, /*vocab_size=*/32};
  std::shared_ptr<ml::train::Dataset> dataset = ml::train::createDataset(
    ml::train::DatasetType::GENERATOR, datasetCb, &gen);
  ASSERT_NO_THROW(
    lora_model.setDataset(ml::train::DatasetModeType::MODE_TRAIN, dataset));
  ASSERT_NO_THROW(
    lora_model.setDataset(ml::train::DatasetModeType::MODE_VALID, dataset));
  ASSERT_NO_THROW(lora_model.train());

  auto stats = lora_model.getTrainingStats();
  EXPECT_GT(stats.loss, 0.0f);

  const auto lora_path =
    (std::filesystem::temp_directory_path() / "nntrainer_qwen3_lora_smoke.bin")
      .string();
  ASSERT_NO_THROW(lora_model.save_weight_lora(lora_path));
  auto trained_lora_snapshot = snapshotFirstElements(lora_model);

  // Step 3: fresh LoRA model, load_weight_lora(base, lora), and verify (a)
  // its base weights now match the base checkpoint (not fresh random init),
  // and (b) its loraA/loraB weights now match what was just saved (not
  // fresh zero/LeCun-normal init).
  auto reloaded_nntr_cfg = makeTinyNntrainerConfig(/*lora_rank=*/4);
  causallm::Qwen3CausalLM reloaded_model(cfg, generation_cfg,
                                         reloaded_nntr_cfg);
  ASSERT_NO_THROW(
    reloaded_model.initializeForTraining(/*lr=*/1e-2f, /*epochs=*/1));
  ASSERT_NO_THROW(reloaded_model.load_weight_lora(base_path, lora_path));

  auto reloaded_snapshot = snapshotFirstElements(reloaded_model);

  unsigned int base_matches = 0, lora_matches = 0;
  for (const auto &[name, value] : reloaded_snapshot) {
    bool is_lora = name.find(":loraA") != std::string::npos ||
                   name.find(":loraB") != std::string::npos;
    if (is_lora) {
      auto it = trained_lora_snapshot.find(name);
      if (it != trained_lora_snapshot.end() &&
          std::abs(it->second - value) < 1e-6f)
        ++lora_matches;
    } else {
      auto it = base_snapshot.find(name);
      if (it != base_snapshot.end() && std::abs(it->second - value) < 1e-6f)
        ++base_matches;
    }
  }

  EXPECT_GT(base_matches, 0u)
    << "expected at least one base weight to match the base checkpoint after "
       "load_weight_lora()";
  EXPECT_GT(lora_matches, 0u)
    << "expected at least one loraA/loraB weight to match the saved adapter "
       "after load_weight_lora()";

  std::filesystem::remove(base_path);
  std::filesystem::remove(lora_path);
}

/**
 * @brief A QAT run's qint4 adapter round-trips: save_weight_lora_qint4()
 *        writes every loraA AND loraB (rank 4 -- the per-channel format has
 *        no minimum-shape constraint) with the per-channel scales training
 *        calibrated, and load_weight_lora_qint4() dequantizes it into the
 *        FP32 LoRA tensors of a fresh model. Every reloaded element must
 *        equal the fake-quantized value the saved scales give the trained
 *        weight.
 */
TEST(Qwen3LoraTrainingSmoke, QatQInt4AdapterRoundTripsTheTrainedInt4Values) {
  auto cfg = makeTinyQwen3Config();
  auto generation_cfg = makeTinyGenerationConfig();

  auto base_nntr_cfg = makeTinyNntrainerConfig(/*lora_rank=*/0);
  causallm::Qwen3CausalLM base_model(cfg, generation_cfg, base_nntr_cfg);
  ASSERT_NO_THROW(base_model.initialize());
  // initialize() leaves the weights as allocated; give the base defined
  // values (and so the LoRA a non-zero gradient)
  std::mt19937 rng(7);
  std::uniform_real_distribution<float> dist(-0.05f, 0.05f);
  auto fill = [&](causallm::Qwen3CausalLM &m, const std::string &only) {
    std::function<void(ml::train::Layer &, nntrainer::RunLayerContext &,
                       void *)>
      fn =
        [&](ml::train::Layer &, nntrainer::RunLayerContext &context, void *) {
          for (unsigned int i = 0; i < context.getNumWeights(); ++i) {
            nntrainer::Tensor &t = context.getWeight(i);
            if (t.getDataType() != ml::train::TensorDim::DataType::FP32 ||
                t.getName().find(only) == std::string::npos)
              continue;
            const bool gamma = t.getName().find("gamma") != std::string::npos;
            for (size_t j = 0; j < t.size(); ++j)
              t.getData<float>()[j] = gamma ? 1.0f : dist(rng);
          }
        };
    m.forEachLayer(fn, nullptr);
  };
  fill(base_model, "");
  const auto base_path =
    (std::filesystem::temp_directory_path() / "nntrainer_qwen3_base_q4.bin")
      .string();
  ASSERT_NO_THROW(base_model.save_weight(base_path));

  auto qat_nntr_cfg = makeTinyNntrainerConfig(/*lora_rank=*/4);
  qat_nntr_cfg["lora_qat"] = true;
  causallm::Qwen3CausalLM qat_model(cfg, generation_cfg, qat_nntr_cfg);
  ASSERT_NO_THROW(qat_model.initializeForTraining(/*lr=*/1e-2f, /*epochs=*/1));
  ASSERT_NO_THROW(qat_model.load_weight_lora(base_path, ""));
  // loraB starts at zero; give it values so the training forward
  // calibrates real loraB channel scales
  fill(qat_model, ":loraB");
  SyntheticDataGen gen{/*seq_len=*/4, /*vocab_size=*/32};
  std::shared_ptr<ml::train::Dataset> dataset = ml::train::createDataset(
    ml::train::DatasetType::GENERATOR, datasetCb, &gen);
  ASSERT_NO_THROW(
    qat_model.setDataset(ml::train::DatasetModeType::MODE_TRAIN, dataset));
  ASSERT_NO_THROW(qat_model.train());

  const auto q4_path =
    (std::filesystem::temp_directory_path() / "nntrainer_qwen3_lora_q4.bin")
      .string();
  ASSERT_NO_THROW(qat_model.save_weight_lora_qint4(q4_path));

  // expected: each trained LoRA weight fake-quantized with its saved scale
  std::map<std::string, std::vector<float>> expected;
  std::function<void(ml::train::Layer &, nntrainer::RunLayerContext &, void *)>
    collect = [&](ml::train::Layer &l, nntrainer::RunLayerContext &context,
                  void *) {
      if (l.getType() != "fully_connected")
        return;
      auto [a_cd, b_cd] =
        nntrainer::FullyConnectedLayer::getRegisteredChannelScales(l.getName());
      for (unsigned int i = 0; i < context.getNumWeights(); ++i) {
        const std::string name = context.getWeight(i).getName();
        const bool is_a = name.find(":loraA") != std::string::npos;
        if (!is_a && name.find(":loraB") == std::string::npos)
          continue;
        const std::vector<float> &cd = is_a ? a_cd : b_cd;
        const nntrainer::Tensor &t = context.getWeight(i);
        const size_t K = t.getDim().height(), N = t.getDim().width();
        ASSERT_EQ(cd.size(), N) << name;
        std::vector<float> v(K * N);
        for (size_t k = 0; k < K; ++k)
          for (size_t n = 0; n < N; ++n) {
            // mirror buildQInt4Record: the calibrated EMA scale when the
            // channel has one, else the natural amax/7 (an all-zero
            // channel is saved as all-zero codes with scale 0)
            float d = cd[n];
            if (d <= 1e-10f) {
              float amax = 0.0f;
              for (size_t kk = 0; kk < K; ++kk)
                amax = std::max(amax, std::abs(t.getData<float>()[kk * N + n]));
              d = amax / 7.0f;
            }
            const float w = t.getData<float>()[k * N + n];
            v[k * N + n] =
              d < 1e-10f
                ? 0.0f
                : std::max(-8.0f, std::min(7.0f, std::round(w / d))) * d;
          }
        expected[name] = v;
      }
    };
  qat_model.forEachLayer(collect, nullptr);
  ASSERT_FALSE(expected.empty());

  auto infer_nntr_cfg = makeTinyNntrainerConfig(/*lora_rank=*/4);
  causallm::Qwen3CausalLM reloaded(cfg, generation_cfg, infer_nntr_cfg);
  ASSERT_NO_THROW(reloaded.initialize());
  ASSERT_NO_THROW(reloaded.load_weight_lora_qint4(base_path, q4_path));

  unsigned int checked = 0, nonzero_b = 0;
  std::function<void(ml::train::Layer &, nntrainer::RunLayerContext &, void *)>
    compare =
      [&](ml::train::Layer &, nntrainer::RunLayerContext &context, void *) {
        for (unsigned int i = 0; i < context.getNumWeights(); ++i) {
          const std::string name = context.getWeight(i).getName();
          auto it = expected.find(name);
          if (it == expected.end())
            continue;
          const float *got = context.getWeight(i).getData<float>();
          for (size_t j = 0; j < it->second.size(); ++j) {
            EXPECT_EQ(got[j], it->second[j]) << name << " element " << j;
            if (name.find(":loraB") != std::string::npos && got[j] != 0.0f)
              ++nonzero_b;
          }
          ++checked;
        }
      };
  reloaded.forEachLayer(compare, nullptr);
  EXPECT_EQ(checked, expected.size());
  EXPECT_GT(nonzero_b, 0u) << "every reloaded loraB is zero";

  std::filesystem::remove(base_path);
  std::filesystem::remove(q4_path);
}

namespace {

/** @brief Whitespace tokenizer good enough to drive TrainingDataGenerator:
 *         each word maps to a fixed non-zero id, so a sample's token ids are
 *         predictable without shipping a tokenizer.json. */
class StubTokenizer : public tokenizers::Tokenizer {
public:
  std::vector<int32_t> Encode(const std::string &text) override {
    return Encode(text, false);
  }
  std::vector<int32_t> Encode(const std::string &text,
                              bool /*add_special_tokens*/) override {
    std::vector<int32_t> ids;
    std::istringstream ss(text);
    std::string word;
    while (ss >> word)
      ids.push_back(id_of(word));
    return ids;
  }
  std::string Decode(const std::vector<int32_t> & /*ids*/) override {
    return "";
  }
  size_t GetVocabSize() override { return VOCAB; }
  std::string IdToToken(int32_t /*id*/) override { return ""; }
  int32_t TokenToId(const std::string &token) override { return id_of(token); }

  static constexpr unsigned int VOCAB = 32;

private:
  /// ids stay in [1, VOCAB-1]: 0 is reserved as the right-pad id.
  static int32_t id_of(const std::string &word) {
    return static_cast<int32_t>(std::hash<std::string>{}(word) % (VOCAB - 1) +
                                1);
  }
};

/** @brief One sample's worth of buffers, fenced by a canary.
 *
 *  nntrainer hands the generator exactly one sample slot per call (it slices
 *  the batch itself), so this is the size the generator is allowed to write.
 *  A generator that writes a whole batch runs into `canary`.
 */
template <unsigned int SEQ> struct FencedSample {
  float input[SEQ];
  float label[SEQ];
  static constexpr float CANARY = -12345.0f;
  float canary[8];
  FencedSample() { std::fill(canary, canary + 8, CANARY); }
  bool intact() const {
    return std::all_of(canary, canary + 8, [](float f) { return f == CANARY; });
  }
};

} // namespace

/**
 * @brief The data generator hands out exactly one sample per call.
 *
 * nntrainer's IterationQueue calls the dataset callback once per batch slot
 * (getBatchSlice(b,1)) and assembles the batch itself. The generator must
 * therefore write one sample -- writing batch_size samples into the one slot
 * it was handed overran the buffer and corrupted the heap, which surfaced
 * later as an unrelated-looking crash.
 */
TEST(Qwen3LoraTrainingSmoke, DataGeneratorYieldsOneSamplePerCall) {
  constexpr unsigned int SEQ = 8;
  constexpr unsigned int VOCAB = StubTokenizer::VOCAB;

  const auto data_path =
    (std::filesystem::temp_directory_path() / "nntrainer_qwen3_gen_smoke.txt")
      .string();
  const unsigned int num_lines = 5;
  {
    std::ofstream file(data_path);
    for (unsigned int i = 0; i < num_lines; ++i)
      file << "alpha" << i << " beta gamma delta\n";
  }

  StubTokenizer tokenizer;
  causallm::TrainingDataGenerator gen(data_path, &tokenizer, SEQ, VOCAB);
  ASSERT_EQ(gen.size(), num_lines);

  FencedSample<SEQ> buf;
  unsigned int samples = 0;
  bool last = false;
  while (!last) {
    std::fill(buf.input, buf.input + SEQ, -2.0f);
    std::fill(buf.label, buf.label + SEQ, -2.0f);

    float *in = buf.input;
    float *lbl = buf.label;
    ASSERT_EQ(gen.next(&in, &lbl, &last), ML_ERROR_NONE)
      << "after " << samples << " samples";
    ++samples;

    // exactly one sample's worth of content was written
    unsigned int non_pad = 0;
    for (unsigned int i = 0; i < SEQ; ++i)
      if (in[i] != 0.0f)
        ++non_pad;
    EXPECT_GT(non_pad, 0u) << "sample " << samples << " wrote no tokens";
    EXPECT_EQ(in[SEQ - 1], 0.0f)
      << "sample " << samples << " filled the whole sequence, expected padding";

    // the label is the next token of every real position, -1 on padding
    for (unsigned int i = 0; i + 1 < non_pad; ++i)
      EXPECT_EQ(lbl[i], in[i + 1]) << "sample " << samples << " position " << i;
    EXPECT_GT(lbl[non_pad - 1], 0.0f)
      << "sample " << samples << ": the last real position has no target";
    EXPECT_LT(lbl[non_pad - 1], static_cast<float>(VOCAB));
    for (unsigned int i = non_pad; i < SEQ; ++i)
      EXPECT_EQ(lbl[i], -1.0f) << "sample " << samples << " padding " << i;

    EXPECT_TRUE(buf.intact())
      << "sample " << samples
      << " wrote past the end of the one-sample buffer it was handed";
    ASSERT_LT(samples, num_lines * 4u) << "generator never reported last";
  }

  EXPECT_EQ(samples, num_lines) << "one epoch must yield one call per sample";

  std::filesystem::remove(data_path);
}

/**
 * @brief The generator ends every epoch on a whole batch.
 *
 * nntrainer skips a trailing iteration it cannot fill, so its samples are
 * produced but never trained on -- and in hybrid mode each of them has queued
 * an lm_head read row nobody pops, shifting every later sample onto its
 * neighbour's row. The generator therefore reports `last` after
 * floor(size / batch) * batch samples, and keeps doing so epoch after epoch.
 */
TEST(Qwen3LoraTrainingSmoke, DataGeneratorEndsEpochOnWholeBatch) {
  constexpr unsigned int SEQ = 8;
  constexpr unsigned int VOCAB = StubTokenizer::VOCAB;

  const auto data_path = (std::filesystem::temp_directory_path() /
                          "nntrainer_qwen3_gen_batch_smoke.txt")
                           .string();
  const unsigned int num_lines = 10;
  {
    std::ofstream file(data_path);
    for (unsigned int i = 0; i < num_lines; ++i)
      file << "alpha" << i << " beta gamma delta\n";
  }

  StubTokenizer tokenizer;
  causallm::TrainingDataGenerator gen(data_path, &tokenizer, SEQ, VOCAB);
  ASSERT_EQ(gen.size(), num_lines);
  ASSERT_NO_THROW(gen.setBatchSize(4));
  EXPECT_EQ(gen.samplesPerEpoch(), 8u);

  FencedSample<SEQ> buf;
  for (unsigned int epoch = 0; epoch < 3; ++epoch) {
    unsigned int samples = 0;
    bool last = false;
    while (!last) {
      float *in = buf.input;
      float *lbl = buf.label;
      ASSERT_EQ(gen.next(&in, &lbl, &last), ML_ERROR_NONE);
      ++samples;
      ASSERT_LT(samples, num_lines * 4u) << "generator never reported last";
    }
    EXPECT_EQ(samples, 8u) << "epoch " << epoch;
  }

  // a batch no epoch can fill is refused up front
  EXPECT_THROW(gen.setBatchSize(num_lines + 1), std::invalid_argument);

  std::filesystem::remove(data_path);
}

namespace {

/** @brief Fixed set of synthetic samples, replayed from the start every
 *  epoch, `last` on the final one. */
struct FixedSampleGen {
  unsigned int seq_len;
  unsigned int vocab_size;
  unsigned int num_samples;
  unsigned int cursor = 0;

  int operator()(float **input, float **label, bool *last) {
    std::mt19937 rng(1000 + cursor);
    std::uniform_int_distribution<unsigned int> tok(1, vocab_size - 1);
    for (unsigned int i = 0; i < seq_len; ++i)
      input[0][i] = static_cast<float>(tok(rng));
    for (unsigned int i = 0; i < seq_len; ++i)
      label[0][i] = static_cast<float>(tok(rng));

    *last = ++cursor == num_samples;
    if (*last)
      cursor = 0;
    return ML_ERROR_NONE;
  }
};

int fixedSampleCb(float **input, float **label, bool *last, void *user_data) {
  return (*static_cast<FixedSampleGen *>(user_data))(input, label, last);
}

} // namespace

/**
 * @brief A batch of 2 scores exactly what two batches of 1 score.
 *
 * With lr = 0 the weights never move, so an epoch's mean training loss over
 * the same samples must not depend on how they are grouped into batches. Any
 * layer that computes only the first batch item, or mixes items, breaks this.
 */
TEST(Qwen3LoraTrainingSmoke, BatchOfTwoMatchesTwoBatchesOfOne) {
  auto cfg = makeTinyQwen3Config();
  auto generation_cfg = makeTinyGenerationConfig();

  auto base_nntr_cfg = makeTinyNntrainerConfig(/*lora_rank=*/0);
  causallm::Qwen3CausalLM base_model(cfg, generation_cfg, base_nntr_cfg);
  ASSERT_NO_THROW(base_model.initialize());
  // An inference graph leaves its weights unset; with all-equal logits every
  // grouping of the samples would score ln(vocab) and the check below would
  // pass vacuously. Give the base real, fixed values.
  {
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-0.2f, 0.2f);
    std::function<void(ml::train::Layer &, nntrainer::RunLayerContext &,
                       void *)>
      fill =
        [&](ml::train::Layer &, nntrainer::RunLayerContext &context, void *) {
          for (unsigned int i = 0; i < context.getNumWeights(); ++i) {
            auto &w = context.getWeight(i);
            if (w.getDataType() != ml::train::TensorDim::DataType::FP32)
              continue;
            const bool norm = w.getName().find("norm") != std::string::npos;
            float *d = w.getData<float>();
            for (size_t k = 0; k < w.size(); ++k)
              d[k] = norm ? 1.0f : dist(rng);
          }
        };
    base_model.forEachLayer(fill, nullptr);
  }
  const auto base_path = (std::filesystem::temp_directory_path() /
                          "nntrainer_qwen3_base_batch_smoke.bin")
                           .string();
  ASSERT_NO_THROW(base_model.save_weight(base_path));

  constexpr unsigned int num_samples = 6;
  auto epoch_loss = [&](unsigned int batch_size) {
    auto nntr_cfg = makeTinyNntrainerConfig(/*lora_rank=*/4, batch_size);
    causallm::Qwen3CausalLM model(cfg, generation_cfg, nntr_cfg);
    EXPECT_NO_THROW(model.initializeForTraining(/*lr=*/0.0f, /*epochs=*/1));
    EXPECT_NO_THROW(model.load_weight_lora(base_path, ""));

    FixedSampleGen gen{/*seq_len=*/4, /*vocab_size=*/32, num_samples};
    std::shared_ptr<ml::train::Dataset> dataset = ml::train::createDataset(
      ml::train::DatasetType::GENERATOR, fixedSampleCb, &gen);
    EXPECT_NO_THROW(
      model.setDataset(ml::train::DatasetModeType::MODE_TRAIN, dataset));
    EXPECT_NO_THROW(model.train());

    auto stats = model.getTrainingStats();
    EXPECT_EQ(stats.num_iterations, num_samples / batch_size)
      << "batch_size=" << batch_size;
    return stats.loss;
  };

  const float loss_b1 = epoch_loss(1);
  const float loss_b2 = epoch_loss(2);
  ASSERT_TRUE(std::isfinite(loss_b1)) << "loss_b1=" << loss_b1;
  // not the uniform-logit loss: the samples really are scored
  ASSERT_GT(std::abs(loss_b1 - std::log(32.0f)), 1e-3f);
  EXPECT_NEAR(loss_b2, loss_b1, 1e-5f * loss_b1);

  std::filesystem::remove(base_path);
}
