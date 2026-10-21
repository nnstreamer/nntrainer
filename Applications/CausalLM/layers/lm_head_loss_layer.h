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

#include <causallm_common_properties.h>
#include <common_properties.h>
#include <layer_context.h>
#include <layer_devel.h>
#include <node_exporter.h>

#include <cstdint>
#include <string>
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

  /** @brief Destructor: drop this layer's entry from the staged-int8
   * registry (a staged copy that no forward adopted) */
  WIN_EXPORT ~LmHeadLossLayer();

  /**
   * @brief Build this layer's two int8 copies of W from @a src (the
   * checkpoint's FP32 weight) ahead of initialization, keyed by @a layer_name.
   * The layer adopts them on its first forward and never fills its own FP32 W.
   * @retval false the head will not run as int8 (or NNTR_LMHEAD_KEEP_FP32 is
   *               set): the caller must load the FP32 weight normally
   */
  static bool stageInt8Weight(const std::string &layer_name, const float *src,
                              unsigned int vocab, unsigned int hidden);

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
  std::tuple<nntrainer::props::Unit, props::ReleaseHostWeight> props;
  unsigned int weight_idx = 0;
  std::vector<float> dx_;     /**< dL/dx from the last training forward */
  std::vector<float> logits_; /**< one chunk of logits / dL/dlogits */
  std::vector<float> x_rows_; /**< the chunk's input rows */
  std::vector<float> dx_rows_;
  /** int8 head (NNTR_LMHEAD_Q8): the frozen W as q8_0x4, built once */
  std::vector<uint8_t> w_q8_;  /**< W   (vocab x hidden), blocks on hidden */
  std::vector<uint8_t> wt_q8_; /**< W^T (hidden x vocab), blocks on vocab */
  const float *w_q8_src_ = nullptr; /**< the W the copies were built from */
  bool w_is_q4_ = false;            /**< w_q8_ holds q4_0x4 instead of q8_0x4 */
  /** a Q4_0 W repacked in place to q4_0x4: the logits GEMM operand */
  const uint8_t *w_q4_table_ = nullptr;
  bool wt_is_q4_ = false; /**< wt_q8_ holds q4_0x4 instead of q8_0x4 */
  /** this layer's name (set in finalize): key into the staged-int8 map */
  std::string layer_name_;
  /** the FP32 W's pages were released (staged or release_host_weight): the
   * FP32 head cannot run anymore */
  bool w_released_ = false;
  /** [PERF-LMHEAD] NNTR_TRAIN_PROF: accumulated ms per stage */
  double prof_ms_[3] = {0.0, 0.0, 0.0};
  /** [PERF-LMHEAD] rows and forwards in the current profiling window */
  size_t prof_rows_ = 0;
  unsigned int prof_calls_ = 0;
  void buildQ8(const float *w, unsigned int vocab, unsigned int hidden);
};

} // namespace causallm

#endif /* __cplusplus */
#endif /* __LM_HEAD_LOSS_LAYER_H__ */
