// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   decision_head_loss_layer.cpp
 * @date   12 October 2026
 * @brief  Decision-model head fused with its sft / rlcd / rlvr loss
 * @see    https://huggingface.co/anthonym21/qwen3-0.6b-rlcd-decision
 * @bug    No known bugs except for NYI items
 */

#include <decision_head_loss_layer.h>

#include <cpu_backend.h>
#include <nntrainer_error.h>
#include <tie_word_embedding.h>
#include <util_func.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <sstream>

namespace causallm {

namespace {
constexpr unsigned int SINGLE_INOUT_IDX = 0;
constexpr unsigned int LETTERS = 26;
} // namespace

DecisionHeadLossLayer::DecisionHeadLossLayer() :
  Layer(),
  props(nntrainer::props::Unit(), props::LetterIds(),
        props::DecisionObjective(), props::DecisionGroup(),
        props::DecisionSeed(), props::ReleaseHostWeight()) {}

void DecisionHeadLossLayer::setProperty(
  const std::vector<std::string> &values) {
  auto remain = nntrainer::loadProperties(values, props);
  NNTR_THROW_IF(!remain.empty(), std::invalid_argument)
    << "[decision_head_loss] unknown property: " << remain.front();
}

void DecisionHeadLossLayer::finalize(nntrainer::InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "[decision_head_loss] takes exactly one input";
  NNTR_THROW_IF(std::get<nntrainer::props::Unit>(props).empty(),
                std::invalid_argument)
    << "[decision_head_loss] unit (vocabulary size) is required";
  const unsigned int vocab = std::get<nntrainer::props::Unit>(props).get();

  NNTR_THROW_IF(std::get<props::LetterIds>(props).empty(),
                std::invalid_argument)
    << "[decision_head_loss] letter_ids is required";
  NNTR_THROW_IF(std::get<props::DecisionGroup>(props).get() == 0,
                std::invalid_argument)
    << "[decision_head_loss] group must be positive";
  letter_ids_.clear();
  std::stringstream ss(std::get<props::LetterIds>(props).get());
  for (std::string tok; std::getline(ss, tok, ',');)
    letter_ids_.push_back(static_cast<unsigned int>(std::stoul(tok)));
  NNTR_THROW_IF(letter_ids_.size() != LETTERS, std::invalid_argument)
    << "[decision_head_loss] letter_ids needs 26 ids, got "
    << letter_ids_.size();
  for (unsigned int id : letter_ids_)
    NNTR_THROW_IF(id >= vocab, std::invalid_argument)
      << "[decision_head_loss] letter id " << id << " >= vocab " << vocab;

  const std::string &obj = std::get<props::DecisionObjective>(props).get();
  NNTR_THROW_IF(obj != "sft" && obj != "soft" && obj != "rlcd" && obj != "rlvr",
                std::invalid_argument)
    << "[decision_head_loss] objective must be sft, soft, rlcd or rlvr, got "
    << obj;

  const ml::train::TensorDim &in_dim = context.getInputDimensions()[0];
  NNTR_THROW_IF(in_dim.channel() != 1, std::invalid_argument)
    << "[decision_head_loss] input must be (B, 1, L, hidden)";
  context.setOutputDimensions({ml::train::TensorDim(
    in_dim.batch(), 1, 1, WIDTH,
    {context.getFormat(), context.getActivationDataType()})});

  // the tied embedding, as the tied lm_head requests it (shared_from)
  ml::train::TensorDim w_dim(
    1, 1, vocab, in_dim.width(),
    {context.getFormat(), context.getWeightDataType()});
  weight_idx = context.requestWeight(w_dim, nntrainer::Initializer::NONE,
                                     nntrainer::WeightRegularizer::NONE, 1.0f,
                                     0.0f, "Embedding", /*trainable=*/false);
}

void DecisionHeadLossLayer::forwarding(nntrainer::RunLayerContext &context,
                                       bool training) {
  using DT = ml::train::TensorDim::DataType;
  const nntrainer::Tensor &x = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &out = context.getOutput(SINGLE_INOUT_IDX);
  const nntrainer::Tensor &w = context.getWeight(weight_idx);
  const bool w_q4_0 = w.getDataType() == DT::Q4_0;
  NNTR_THROW_IF(x.getDataType() != DT::FP32 ||
                  (w.getDataType() != DT::FP32 && !w_q4_0),
                std::invalid_argument)
    << "[decision_head_loss] FP32 input and an FP32 or Q4_0 weight only";

  const unsigned int batch = x.getDim().batch();
  const unsigned int len = x.getDim().height();
  const unsigned int hidden = x.getDim().width();
  const float *x_data = x.getData<float>();
  float *out_data = out.getData<float>();
  std::fill(out_data, out_data + out.size(), 0.0f);
  if (training)
    dx_.assign((size_t)batch * len * hidden, 0.0f);

  // The head reads LETTERS rows of the frozen W: keep those, and when this
  // layer owns W (release_host_weight) hand the rest of its pages back.
  if (letter_w_.empty() && w_q4_0) {
    // a Q4_0 tied table (plain rows, or repacked in place by a tied head)
    const uint8_t *wq = w.getData<uint8_t>();
    const bool x4 = nntrainer::TieWordEmbedding::isQ4_0x4Table(wq);
    const size_t row_bytes = (size_t)(hidden / 32) * (sizeof(uint16_t) + 16);
    letter_w_.resize((size_t)LETTERS * hidden);
    for (unsigned int i = 0; i < LETTERS; ++i) {
      float *dst = letter_w_.data() + (size_t)i * hidden;
      if (x4)
        nntrainer::TieWordEmbedding::dequantizeRowQ4_0x4(wq, letter_ids_[i],
                                                         hidden, dst);
      else
        nntrainer::dequantize_row_q4_0(wq + (size_t)letter_ids_[i] * row_bytes,
                                       dst, hidden);
    }
  }
  if (letter_w_.empty()) {
    const float *w_data = w.getData<float>();
    letter_w_.resize((size_t)LETTERS * hidden);
    for (unsigned int i = 0; i < LETTERS; ++i)
      std::memcpy(letter_w_.data() + (size_t)i * hidden,
                  w_data + (size_t)letter_ids_[i] * hidden,
                  sizeof(float) * hidden);
    if (std::get<props::ReleaseHostWeight>(props).get())
      nntrainer::releaseHostPages(w_data, w.getMemoryBytes());
  }

  if (!context.isLabelAvailable(SINGLE_INOUT_IDX)) {
    context.setLoss(0.0f);
    return;
  }
  const float *label = context.getLabel(SINGLE_INOUT_IDX).getData<float>();

  const std::string &obj = std::get<props::DecisionObjective>(props).get();
  const unsigned int group = std::get<props::DecisionGroup>(props).get();
  if (!rng_seeded_) {
    rng_.seed(std::get<props::DecisionSeed>(props).get());
    rng_seeded_ = true;
  }

  double nll_sum = 0.0;
  std::vector<double> logits(LETTERS), p(LETTERS), dz(LETTERS);
  for (unsigned int b = 0; b < batch; ++b) {
    const float *t = label + (size_t)b * WIDTH;
    unsigned int k = 0, answer = LETTERS;
    for (unsigned int i = 0; i < LETTERS && t[i] >= 0.0f; ++i, ++k)
      if (t[i] > 0.5f)
        answer = i;
    if (obj == "soft") {
      double tsum = 0.0;
      for (unsigned int i = 0; i < k; ++i)
        tsum += t[i];
      NNTR_THROW_IF(k < 1 || tsum <= 0.0, std::invalid_argument)
        << "[decision_head_loss] sample " << b << ": " << k
        << " options, target mass " << tsum;
    } else {
      NNTR_THROW_IF(k < 2 || answer >= k, std::invalid_argument)
        << "[decision_head_loss] sample " << b << ": " << k
        << " options, answer " << answer;
    }
    const unsigned int row =
      std::min(static_cast<unsigned int>(t[LETTERS]), len - 1);
    const float *h = x_data + ((size_t)b * len + row) * hidden;

    // logits of the K declared letters, softmax over them
    double mx = -INFINITY;
    for (unsigned int i = 0; i < k; ++i) {
      const float *wr = letter_w_.data() + (size_t)i * hidden;
      double d = 0.0;
      for (unsigned int j = 0; j < hidden; ++j)
        d += (double)h[j] * wr[j];
      logits[i] = d;
      mx = std::max(mx, d);
    }
    double se = 0.0;
    for (unsigned int i = 0; i < k; ++i)
      se += std::exp(logits[i] - mx);
    const double lse = mx + std::log(se);
    float *o = out_data + (size_t)b * WIDTH;
    for (unsigned int i = 0; i < k; ++i) {
      p[i] = std::exp(logits[i] - lse);
      o[i] = static_cast<float>(p[i]);
    }
    double nll;
    if (obj == "soft") {
      // soft targets: loss = -sum_i t_i log p_i = tsum*lse - sum_i t_i z_i
      double tsum = 0.0, tz = 0.0;
      for (unsigned int i = 0; i < k; ++i) {
        tsum += t[i];
        tz += t[i] * logits[i];
      }
      nll = tsum * lse - tz;
    } else {
      nll = lse - logits[answer];
    }
    o[LETTERS] = static_cast<float>(nll);
    nll_sum += nll;

    if (!training)
      continue;

    // dL/dz over the K logits of this sample, for a batch-mean loss
    std::fill(dz.begin(), dz.end(), 0.0);
    if (obj == "sft") {
      for (unsigned int i = 0; i < k; ++i)
        dz[i] = p[i];
      dz[answer] -= 1.0;
    } else if (obj == "soft") {
      // d(-sum_i t_i log p_i)/dz_i = tsum*p_i - t_i (tsum = 1 for a
      // normalized target, where this is p_i - t_i)
      double tsum = 0.0;
      for (unsigned int i = 0; i < k; ++i)
        tsum += t[i];
      for (unsigned int i = 0; i < k; ++i)
        dz[i] = tsum * p[i] - t[i];
    } else {
      // REINFORCE: loss = -1/G sum_g adv_g log p(a_g);
      // d(-log p(a))/dz = p - onehot(a)
      std::discrete_distribution<unsigned int> pick(p.begin(), p.begin() + k);
      std::vector<unsigned int> acts(group);
      std::vector<double> reward(group);
      double rsum = 0.0;
      for (unsigned int g = 0; g < group; ++g) {
        acts[g] = pick(rng_);
        const double c = acts[g] == answer ? 1.0 : 0.0;
        reward[g] = obj == "rlcd" ? c - p[acts[g]] : c;
        rsum += reward[g];
      }
      for (unsigned int g = 0; g < group; ++g) {
        const double base = group > 1 ? (rsum - reward[g]) / (group - 1) : 0.0;
        const double adv = (reward[g] - base) / group;
        for (unsigned int i = 0; i < k; ++i)
          dz[i] += adv * p[i];
        dz[acts[g]] -= adv;
      }
    }

    // dL/dx[row] = (1/B) sum_i dz_i W_letter_i
    float *dxr = dx_.data() + ((size_t)b * len + row) * hidden;
    for (unsigned int i = 0; i < k; ++i) {
      const double c = dz[i] / batch;
      if (c == 0.0)
        continue;
      const float *wr = letter_w_.data() + (size_t)i * hidden;
      for (unsigned int j = 0; j < hidden; ++j)
        dxr[j] += static_cast<float>(c * wr[j]);
    }
  }
  context.setLoss(static_cast<float>(nll_sum / batch));
}

void DecisionHeadLossLayer::calcDerivative(
  nntrainer::RunLayerContext &context) {
  nntrainer::Tensor &dx = context.getOutgoingDerivative(SINGLE_INOUT_IDX);
  NNTR_THROW_IF(dx_.size() != dx.size(), std::runtime_error)
    << "[decision_head_loss] calcDerivative without a training forward";
  std::memcpy(dx.getData<float>(), dx_.data(), sizeof(float) * dx_.size());
  const float loss_scale = context.getLossScale();
  if (loss_scale != 1.0f)
    dx.multiply_i(loss_scale);
}

} // namespace causallm
