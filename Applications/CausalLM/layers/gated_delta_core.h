// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   gated_delta_core.h
 * @date   2026
 * @see    https://github.com/nntrainer/nntrainer
 * @brief  Gated delta rule recurrent core for Qwen3.5 linear attention
 * @bug    No known bugs except for NYI items
 *
 * Implements one Qwen3.5 linear-attention block body (per HF
 * modeling_qwen3_5.py gated delta net) with all weights frozen. The
 * surrounding graph provides q/k/v (post conv+silu), the z gate, and the
 * b/a scalars; this layer runs the recurrence and the gated norm:
 *
 * Inputs (all [B, 1, T, W]):
 *   idx 0: q  (W = num_heads * head_dim)
 *   idx 1: k  (W = num_heads * head_dim)
 *   idx 2: v  (W = num_heads * head_dim)
 *   idx 3: z  (W = num_heads * head_dim)  -- gated-norm gate
 *   idx 4: b  (W = num_heads)
 *   idx 5: a  (W = num_heads)
 *
 * Frozen weights (FP32, trainable=false):
 *   A_log      [num_heads]  (raw)
 *   dt_bias    [num_heads]  (raw)
 *   norm_w     [head_dim]   (gated-norm weight, raw -- NOT the (1+w) form)
 *
 * Recurrent state (MAX_LIFESPAN tensor): S [B, num_heads, head_dim,
 * head_dim] FP32, zeroed on every full-sequence forward and on every
 * prefill starting at position 0. S[c][d]: c indexes the key channel,
 * d the value channel.
 *
 * Per-step math (fp32, per head h):
 *   qh = l2norm(q_t[h]) * 1/sqrt(head_dim);  kh = l2norm(k_t[h])
 *   beta = sigmoid(b_t[h]);  g = -exp(A_log[h]) * softplus(a_t[h]+dt_bias[h])
 *   S <- S * exp(g)
 *   kv[d] = sum_c S[c][d] * kh[c];  delta[d] = (v_t[d] - kv[d]) * beta
 *   S <- S + kh (outer) delta
 *   o[d] = sum_c S[c][d] * qh[c]
 *   y = norm_w * rmsnorm(o) * silu(z)        (rms over head_dim, eps 1e-6)
 *
 * The node must be built with trainable=false (nothing to train); the
 * layer only propagates input gradients so LoRA adapters feeding q/k/v/z
 * receive their dX.
 *
 * Training backward runs a reverse scan. Full-sequence training forwards
 * checkpoint the recurrent state every CHECKPOINT_SEG steps; the reverse
 * scan re-runs short forward segments from those checkpoints so per-step
 * states are regenerated instead of stored (O(C) memory per head, not
 * O(T)).
 */

#ifndef __CAUSALLM_GATED_DELTA_CORE_H__
#define __CAUSALLM_GATED_DELTA_CORE_H__

#pragma once
#ifdef _WIN32
#define WIN_EXPORT __declspec(dllexport)
#else
#define WIN_EXPORT
#endif

#include <array>
#include <string>
#include <tuple>

#include <common_properties.h>
#include <layer_context.h>
#include <layer_devel.h>
#include <node_exporter.h>

namespace causallm {
namespace props {

/** Number of recurrence heads (default 16 = Qwen3.5-0.8B). */
class GDeltaNumHeads : public nntrainer::Property<unsigned int> {
public:
  GDeltaNumHeads(unsigned int value = 16) { set(value); }
  static constexpr const char *key = "gdelta_num_heads";
  using prop_tag = nntrainer::uint_prop_tag;
};

/** Per-head recurrence dimension (default 128 = Qwen3.5-0.8B). */
class GDeltaHeadDim : public nntrainer::Property<unsigned int> {
public:
  GDeltaHeadDim(unsigned int value = 128) { set(value); }
  static constexpr const char *key = "gdelta_head_dim";
  using prop_tag = nntrainer::uint_prop_tag;
};

/**
 * @brief Request the training backward caches: per-segment recurrent-state
 * checkpoints and the pre-gate core output. Inference graphs leave it off
 * (the default) so neither tensor is allocated; a training graph turns it
 * on (the model builder passes it when FOR_TRAINING).
 */
class GDeltaBackwardCache : public nntrainer::Property<bool> {
public:
  GDeltaBackwardCache(bool value = false) { set(value); }
  static constexpr const char *key = "gdelta_backward_cache";
  using prop_tag = nntrainer::bool_prop_tag;
};

} // namespace props

class GatedDeltaCoreLayer final : public nntrainer::Layer {
public:
  WIN_EXPORT GatedDeltaCoreLayer();
  WIN_EXPORT ~GatedDeltaCoreLayer() {}

  WIN_EXPORT void finalize(nntrainer::InitLayerContext &context) override;
  WIN_EXPORT void forwarding(nntrainer::RunLayerContext &context,
                             bool training) override;
  WIN_EXPORT void incremental_forwarding(nntrainer::RunLayerContext &context,
                                         unsigned int from, unsigned int to,
                                         bool training) override;
  WIN_EXPORT void calcDerivative(nntrainer::RunLayerContext &context) override;
  WIN_EXPORT void calcGradient(nntrainer::RunLayerContext &context) override;

  /** Input gradients flow through; the frozen weights take none. */
  WIN_EXPORT bool supportBackwarding() const override { return true; }

  WIN_EXPORT void
  exportTo(nntrainer::Exporter &exporter,
           const ml::train::ExportMethods &method) const override {}

  WIN_EXPORT const std::string getType() const override {
    return GatedDeltaCoreLayer::type;
  }

  WIN_EXPORT void setProperty(const std::vector<std::string> &values) override;

  WIN_EXPORT void updateTensorsByInputDimensions(
    nntrainer::RunLayerContext &context,
    std::vector<nntrainer::TensorDim> input_dimensions) override {
    (void)context;
    (void)input_dimensions;
  }

  inline static const std::string type = "gated_delta_core";

private:
  static constexpr size_t Q_IDX = 0;
  static constexpr size_t K_IDX = 1;
  static constexpr size_t V_IDX = 2;
  static constexpr size_t Z_IDX = 3;
  static constexpr size_t B_IDX = 4;
  static constexpr size_t A_IDX = 5;
  static constexpr size_t OUT_IDX = 0;

  enum WeightIdx { w_A_log = 0, w_dt_bias = 1, w_norm = 2, NUM_WEIGHTS };
  enum TensorIdx {
    t_state = 0, /**< [B, NH, HD, HD] recurrent state (MAX_LIFESPAN) */
    t_ckpt = 1,  /**< [B, NSEG, NH, HD*HD] training fwd checkpoints */
    t_o_raw = 2, /**< [B, 1, T, NH*HD] pre-gate core output */
    NUM_TENSORS
  };

  /** Checkpoint the recurrent state every this many steps (training). */
  static constexpr unsigned int CHECKPOINT_SEG = 32;

  std::tuple<props::GDeltaNumHeads, props::GDeltaHeadDim,
             props::GDeltaBackwardCache>
    gdelta_props;
  unsigned int num_heads = 16;
  unsigned int head_dim = 128;
  bool backward_cache = false;

  std::array<unsigned int, NUM_WEIGHTS> weight_idx;
  std::array<unsigned int, NUM_TENSORS> tensor_idx;
};

} // namespace causallm

#endif // __CAUSALLM_GATED_DELTA_CORE_H__
