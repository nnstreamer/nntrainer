// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   lm_head_loss_layer.h
 * @date   12 October 2026
 * @brief  LM head fused with a masked next-token cross-entropy, for training
 *         a causal LM on every position of a sample.
 * @see    https://github.com/nntrainer/nntrainer
 * @bug    No known bugs except for NYI items
 *
 * @details Input  x: (B, 1, L, hidden) final hidden states.
 *          Label  t: (B, 1, 1, L) next-token id of each position, or -1 for a
 *                    position that is not scored (right padding, the last
 *                    real token).
 *          Output  : (B, 1, 1, L) per-position loss (0 where t = -1).
 *          Loss    : mean over the scored positions of the whole batch of
 *                    -log softmax(x W^T)[t], W the (vocab, hidden) tied
 *                    embedding (shared_from the embedding layer, frozen).
 *
 *          The (L, vocab) logits are never planned as a tensor -- at vocab
 *          151936 that is 78 MB per sample at L = 128. forwarding() runs the
 *          scored rows through the head a chunk at a time, and when training
 *          turns each chunk's logits into dL/dlogits on the spot and
 *          multiplies it back through W. The resulting dL/dx (L, hidden) is
 *          kept until calcDerivative() hands it on. W is frozen, so no dW.
 */
#ifndef __LM_HEAD_LOSS_LAYER_H__
#define __LM_HEAD_LOSS_LAYER_H__
#ifdef __cplusplus

#pragma once
#ifdef _WIN32
#define WIN_EXPORT __declspec(dllexport)
#else
#define WIN_EXPORT
#endif

#include <common_properties.h>
#include <layer_context.h>
#include <layer_devel.h>
#include <node_exporter.h>

#include <vector>

namespace causallm {

/**
 * @class   LmHeadLossLayer
 * @brief   Tied LM head + masked next-token cross-entropy loss
 */
WIN_EXPORT class LmHeadLossLayer final : public nntrainer::Layer {
public:
  /** @brief Constructor */
  WIN_EXPORT LmHeadLossLayer();

  /** @brief Destructor */
  WIN_EXPORT ~LmHeadLossLayer() = default;

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
    return LmHeadLossLayer::type;
  }

  /** @copydoc Layer::setProperty(const std::vector<std::string> &values) */
  WIN_EXPORT void setProperty(const std::vector<std::string> &values) override;

  static constexpr const char *type = "lm_head_loss";

private:
  std::tuple<nntrainer::props::Unit> props;
  unsigned int weight_idx = 0;
  std::vector<float> dx_;     /**< dL/dx from the last training forward */
  std::vector<float> logits_; /**< one chunk of logits / dL/dlogits */
  std::vector<float> x_rows_; /**< the chunk's input rows */
  std::vector<float> dx_rows_;
};

} // namespace causallm

#endif /* __cplusplus */
#endif /* __LM_HEAD_LOSS_LAYER_H__ */
