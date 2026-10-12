// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   lora_train.h
 * @date   31 July 2026
 * @brief  Text-file training data generator for Qwen3 LoRA fine-tuning.
 * @bug    No known bugs except for NYI items
 *
 * @details Each line of the input file is one training sample. A sample of
 *          n tokens is split into (input, label) as: input = its first
 *          n - 1 token ids, right-padded with 0 to seq_len; label = the next
 *          token id of every input position (ids[1..n-1]), -1 on the
 *          padding. Every real position is scored (lm_head_loss_layer.h);
 *          causal attention keeps the padding from reaching them.
 *
 *          A line is treated as "chat format" (joined as a single sample
 *          rather than split into multiple lines) whenever the file
 *          contains a `<|im_start|>user` marker anywhere; otherwise each
 *          line is its own independent sample (e.g. one-sentence
 *          classification data).
 */
#ifndef __LORA_TRAIN_H__
#define __LORA_TRAIN_H__

#include <atomic>
#include <functional>
#include <limits>
#include <random>
#include <string>
#include <tokenizers_cpp.h>
#include <vector>

namespace causallm {

/**
 * @brief Generates (input, label) training samples from a text file for
 *        Qwen3 LoRA fine-tuning: next-token targets for every position.
 */
class TrainingDataGenerator {
public:
  /**
   * @param data_path path to a plain-text or chat-format training file
   * @param tokenizer tokenizer used to encode each sample (not owned)
   * @param seq_len fixed input width every sample is right-padded to
   * @param vocab_size vocabulary size; a target id outside it is not scored
   * @param max_samples optional cap on the number of samples loaded (0 =
   *        no cap)
   * @param seed RNG seed for deterministic epoch shuffling
   */
  TrainingDataGenerator(const std::string &data_path,
                        tokenizers::Tokenizer *tokenizer, unsigned int seq_len,
                        unsigned int vocab_size, unsigned int max_samples = 0,
                        unsigned int seed = 42);

  /**
   * @brief ml::train GENERATOR dataset callback signature.
   * @details Fills one sample's (input, label) and reshuffles at each epoch
   *          boundary. `*last` is set true after the final sample of an
   *          epoch (matching ml::train::DatasetGenCb semantics).
   */
  int next(float **input, float **label, bool *last);

  /**
   * @brief Hybrid (GPU trunk / CPU head) mode: when set, next() hands the
   *        sample's input token ids to this writer, which fills input[0]
   *        (seq_len rows of the frozen trunk's hidden state) instead of the
   *        ids themselves. Runs on the dataset producer thread, so the trunk
   *        for the next sample overlaps the current training step.
   */
  std::function<void(const int32_t *ids, unsigned int used, float *input)>
    input_writer;

  /** @brief Number of samples loaded from the training file. */
  size_t size() const { return samples_.size(); }

  /**
   * @brief End every epoch on a whole batch.
   * @details nntrainer skips a trailing iteration it cannot fill to the batch
   *          size, so the samples in it are produced but never trained on --
   *          in hybrid mode after the GPU trunk has already run them. Ending
   *          the epoch after floor(size / batch) * batch samples means no
   *          partial iteration is ever formed. The reshuffle at every epoch
   *          changes which samples sit out.
   * @param batch_size the model's training batch size (0 is taken as 1)
   */
  void setBatchSize(unsigned int batch_size);

  /** @brief Number of samples next() yields per epoch. */
  size_t samplesPerEpoch() const;

private:
  struct Sample {
    std::vector<int32_t> ids; /**< full tokenized sample, unpadded */
  };

  std::vector<Sample> samples_;
  std::vector<size_t> index_order_;
  size_t cursor_ = 0;
  unsigned int batch_size_ = 1;
  unsigned int seq_len_;
  unsigned int vocab_size_;
  std::mt19937 rng_;

  void loadTextFile(const std::string &path, tokenizers::Tokenizer *tokenizer,
                    unsigned int max_samples);
  void reshuffle();
};

/**
 * @brief ml::train::createDataset(GENERATOR, ...)-compatible free-function
 *        callback; `user_data` must be a `TrainingDataGenerator*`.
 */
int trainingDataGenCb(float **input, float **label, bool *last,
                      void *user_data);

/**
 * @brief All tunable knobs for a single LoRA fine-tuning run.
 *
 * Mirrors the CLI flags accepted by nntr_lora_train (see train_qwen3_lora.cpp).
 * Fields left at zero / empty use the value from nntr_config.json.
 */
struct TrainParams {
  std::string model_path;      ///< directory containing config.json etc.
  std::string data_path;       ///< plain-text or chat-format training file
  float lr = 1e-4f;            ///< learning rate
  float lr_decay_rate = 0.0f;  ///< exponential lr decay rate (0 = constant)
  float lr_decay_steps = 0.0f; ///< decay interval in iterations
  unsigned int epochs = 1;     ///< number of training epochs
  /**
   * @brief Training batch size: how many samples the model consumes (and the
   *        optimizer updates on) per iteration. 0 = keep nntr_config.json.
   *
   * This is a model property only. The data generator stays sample-at-a-time:
   * nntrainer's IterationQueue fills one slot per generator call and forms the
   * batch itself, so a generator that wrote a whole batch into the single slot
   * it was handed would run off the end of that slot's buffer.
   */
  unsigned int batch_size = 0;
  std::string output_path;      ///< LoRA adapter output path
  std::string resume_lora_path; ///< resume from an existing adapter
  unsigned int lora_rank = 0;   ///< override nntr_config lora_rank (0 = config)
  unsigned int lora_alpha = 0; ///< override nntr_config lora_alpha (0 = config)
  unsigned int max_samples = 0;  ///< cap training samples (0 = no cap)
  unsigned int lora_last_n = 0;  ///< LoRA on last N layers (0 = all)
  unsigned int seq_len = 0;      ///< override init_seq_len (0 = config)
  std::string hybrid_trunk_path; ///< GPU trunk model dir (empty = CPU only)
  float clip_grad = -1.0f;       ///< gradient clip global norm (-1 = config)
  bool train_norms = false;      ///< also train RMSNorm gammas
  bool lora_qat = false;         ///< fake-quantize loraA/loraB to Q4_0
  bool lora_weight_q4 = false;   ///< also save a Q4_0 adapter
  float qat_ema_momentum = 0.0f; ///< EMA momentum for QAT scale update
  unsigned int seed = 42;        ///< RNG seed for epoch shuffling
};

/**
 * @brief Per-epoch progress reported through the training callback.
 */
struct TrainProgress {
  unsigned int epoch = 0;     ///< 1-based epoch index
  float train_loss = 0.0f;    ///< running training loss for this epoch
  float valid_loss = 0.0f;    ///< validation loss for this epoch
  float accuracy = 0.0f;      ///< top-1 accuracy (0 if unavailable)
  bool saved = false;         ///< whether the adapter was saved this epoch
  std::string output_path;    ///< current best-adapter output path
  std::string q4_output_path; ///< Q4_0 adapter path (empty if not saved)
  float best_loss =
    std::numeric_limits<float>::max(); ///< best valid loss so far
  unsigned int num_samples = 0;        ///< total training samples loaded
  double epoch_seconds = 0.0; ///< wall time of the epoch (train + validation)
  double train_seconds = 0.0; ///< training part of it; the rest is validation
};

/**
 * @brief Callback invoked after every epoch.
 * @param progress  snapshot of the training state
 * @param user_data opaque pointer forwarded from the caller
 */
using TrainProgressCb = void (*)(const TrainProgress &progress,
                                 void *user_data);

/**
 * @brief Callback invoked after every training iteration.
 * @param iteration  0-based iteration index within the current epoch
 * @param epoch      1-based epoch index
 * @param loss       training loss for this iteration
 * @param user_data  opaque pointer forwarded from the caller
 */
using IterationProgressCb = void (*)(unsigned int iteration, unsigned int epoch,
                                     float loss, void *user_data);

/**
 * @brief Run a complete LoRA fine-tuning training loop.
 *
 * This is the library entry point extracted from the nntr_lora_train CLI.
 * It loads the model, builds the dataset generator, trains for the
 * requested number of epochs, and saves the best adapter. After each
 * epoch @p callback is invoked (if non-null) with the current progress.
 * After every training iteration @p iter_cb is invoked (if non-null) with
 * the iteration index, epoch, and current loss.
 *
 * @param params    fully populated training parameters
 * @param callback  per-epoch progress callback (may be nullptr)
 * @param user_data opaque pointer forwarded to @p callback
 * @param iter_cb   per-iteration progress callback (may be nullptr)
 * @param iter_data opaque pointer forwarded to @p iter_cb
 * @return 0 on success, non-zero on failure
 */
int runLoraTraining(const TrainParams &params,
                    TrainProgressCb callback = nullptr,
                    void *user_data = nullptr,
                    IterationProgressCb iter_cb = nullptr,
                    void *iter_data = nullptr);

/**
 * @brief Thread-safe flag for cancelling an in-progress LoRA training run.
 *
 * @return Reference to a static `std::atomic<bool>`. When set to true by
 *         `stopLoraTraining()` (from any thread), the training loop checks
 *         it after every iteration and aborts.
 */
std::atomic<bool> &loraTrainCancelledFlag();

} // namespace causallm

#endif /* __LORA_TRAIN_H__ */
