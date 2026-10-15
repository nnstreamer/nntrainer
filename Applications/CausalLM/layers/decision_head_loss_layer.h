// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   decision_head_loss_layer.h
 * @date   12 October 2026
 * @brief  Decision-model head (26 letter rows of the tied embedding) fused
 *         with its training loss: supervised (sft) or RLCD / RLVR
 *         REINFORCE under bandit feedback.
 * @see    https://huggingface.co/anthonym21/qwen3-0.6b-rlcd-decision
 * @bug    No known bugs except for NYI items
 *
 * @details Input  x: (B, 1, L, hidden) final hidden states.
 *          Label  t: (B, 1, 1, 27). t[0..25] per letter: 1 for the correct
 *                    option, 0 for another declared option, -1 beyond the
 *                    option count K (masked). t[26]: the readout row (the
 *                    last prompt token).
 *          Output  : (B, 1, 1, 27). [0..25] the option probabilities
 *                    softmax(x[row] W_letters^T) over the first K letters,
 *                    0 beyond; [26] the sample's -log p(correct).
 *
 *          objective=sft   loss = mean_b -log p_b(answer)
 *          objective=rlcd  per sample, draw `group` actions a ~ p, reward
 *                          r = 1[a correct] - p(a), advantage = r minus the
 *                          leave-one-out mean of the group's rewards,
 *                          loss = -mean_{b,g} advantage * log p(a)
 *                          (eve-rlcd's policy_gradient_loss; the label only
 *                          supplies the environment's outcome)
 *          objective=rlvr  the same with r = 1[a correct]
 *
 *          The reported loss is always the mean -log p(correct), whatever
 *          the objective, so training and validation curves read the same.
 *          W is the tied embedding (shared_from the embedding), frozen; only
 *          its 26 letter rows are read.
 */
#ifndef __DECISION_HEAD_LOSS_LAYER_H__
#define __DECISION_HEAD_LOSS_LAYER_H__
#ifdef __cplusplus

#pragma once
#ifdef _WIN32
#define WIN_EXPORT __declspec(dllexport)
#else
#define WIN_EXPORT
#endif

#include <base_properties.h>
#include <causallm_common_properties.h>
#include <common_properties.h>
#include <layer_context.h>
#include <layer_devel.h>
#include <node_exporter.h>

#include <random>
#include <vector>

namespace causallm {

namespace props {

/** @brief comma-separated token ids of the letters " A".." Z" */
class LetterIds : public nntrainer::Property<std::string> {
public:
  static constexpr const char *key = "letter_ids";
  using prop_tag = nntrainer::str_prop_tag;
};

/** @brief training objective: sft | rlcd | rlvr */
class DecisionObjective : public nntrainer::Property<std::string> {
public:
  DecisionObjective(const std::string &value = "sft") { set(value); }
  static constexpr const char *key = "objective";
  using prop_tag = nntrainer::str_prop_tag;
};

/** @brief actions sampled per sample by the REINFORCE objectives */
class DecisionGroup : public nntrainer::Property<unsigned int> {
public:
  DecisionGroup(unsigned int value = 4) { set(value); }
  static constexpr const char *key = "group";
  using prop_tag = nntrainer::uint_prop_tag;
};

/** @brief seed of the action sampler */
class DecisionSeed : public nntrainer::Property<unsigned int> {
public:
  DecisionSeed(unsigned int value = 0) { set(value); }
  static constexpr const char *key = "seed";
  using prop_tag = nntrainer::uint_prop_tag;
};

} // namespace props

/**
 * @class   DecisionHeadLossLayer
 * @brief   26-letter decision head + sft / rlcd / rlvr loss
 */
WIN_EXPORT class DecisionHeadLossLayer final : public nntrainer::Layer {
public:
  /** @brief Constructor */
  WIN_EXPORT DecisionHeadLossLayer();

  /** @brief Destructor */
  WIN_EXPORT ~DecisionHeadLossLayer() = default;

  /** @copydoc Layer::finalize(InitLayerContext &context) */
  WIN_EXPORT void finalize(nntrainer::InitLayerContext &context) override;

  /** @copydoc Layer::forwarding(RunLayerContext &context, bool training) */
  WIN_EXPORT void forwarding(nntrainer::RunLayerContext &context,
                             bool training) override;

  /** @copydoc Layer::calcDerivative(RunLayerContext &context) */
  WIN_EXPORT void calcDerivative(nntrainer::RunLayerContext &context) override;

  /** @copydoc Layer::supportBackwarding() */
  WIN_EXPORT bool supportBackwarding() const override { return true; }

  /** @copydoc Layer::requireLabel() */
  WIN_EXPORT bool requireLabel() const override { return true; }

  /** @copydoc Layer::exportTo() */
  WIN_EXPORT void
  exportTo(nntrainer::Exporter &exporter,
           const ml::train::ExportMethods &method) const override {}

  /** @copydoc Layer::getType() */
  WIN_EXPORT const std::string getType() const override {
    return DecisionHeadLossLayer::type;
  }

  /** @copydoc Layer::setProperty(const std::vector<std::string> &values) */
  WIN_EXPORT void setProperty(const std::vector<std::string> &values) override;

  static constexpr const char *type = "decision_head_loss";

  /** @brief label / output width: 26 letters + the readout row */
  static constexpr unsigned int WIDTH = 27;

private:
  std::tuple<nntrainer::props::Unit, props::LetterIds, props::DecisionObjective,
             props::DecisionGroup, props::DecisionSeed,
             props::ReleaseHostWeight>
    props;
  unsigned int weight_idx = 0;
  std::vector<unsigned int> letter_ids_;
  /** the letter rows of the frozen W (LETTERS x hidden), copied on the first
   * forward: the only part of the vocabulary-sized W this head reads */
  std::vector<float> letter_w_;
  std::mt19937 rng_;
  bool rng_seeded_ = false;
  std::vector<float> dx_; /**< dL/dx from the last training forward */
};

} // namespace causallm

#endif /* __cplusplus */
#endif /* __DECISION_HEAD_LOSS_LAYER_H__ */
