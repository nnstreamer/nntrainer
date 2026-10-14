// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Hyeong-Gwon Hong
 *
 * @file   causal_conv1d_layer.h
 * @date   01 April 2026
 * @brief  Causal depthwise Conv1D layer with conv-state cache for CausalLM
 * @see    https://github.com/nntrainer/nntrainer
 * @author Hyeong-Gwon Hong
 * @bug    No known bugs except for NYI items
 *
 * Input/output tensor layout: [B, 1, T, W]
 *   B = batch, T = sequence length (height), W = features/channels (width)
 *
 * Weight layout: [1, 1, K, W] FP32, K = kernel size ("kernel_size" property,
 *   default 3, supported values {3, 4}; Qwen3.5 linear attention uses 4)
 *   position 0 (w0): applied to the current token x_t
 *   position 1 (w1): applied to x_{t-1}
 *   ...
 *
 * Conv-state cache: [B, 1, K-1, W] FP32 (MAX_LIFESPAN)
 *   state[b, 0, 0, :] = oldest cached token
 *   state[b, 0, K-2, :] = newest cached token (x_{t-1})
 *
 * During single-token decode (to - from == 1):
 *   output = w0*x_t + sum_k w_k * state[K-1-k]  (state[0] oldest)
 *   then: state shifts by one, newest = x_t  (O(W) update)
 *
 * During prefill (to > from + 1) and full-sequence forwarding() (training):
 *   Computes all positions and saves the last K-1 tokens to state.
 *
 * Backwarding: the conv weight stays frozen; calcDerivative propagates dX
 * (an anti-causal conv of dY, computed via time-reversal + the prefill
 * kernel) so gradients reach trainable projections feeding the conv.
 */

#ifndef __CAUSAL_LM_CAUSAL_CONV1D_LAYER_H__
#define __CAUSAL_LM_CAUSAL_CONV1D_LAYER_H__

#include <array>
#include <limits>
#include <string>
#include <tuple>
#include <vector>

#include <common_properties.h>
#include <layer_devel.h>
#include <layer_impl.h>
#include <tensor_dim.h>

namespace causallm {

/**
 * @brief Causal depthwise Conv1D layer with conv-state cache for CausalLM
 */
class CausalConv1DLayer : public nntrainer::LayerImpl {
public:
  CausalConv1DLayer();
  ~CausalConv1DLayer() override = default;

  void finalize(nntrainer::InitLayerContext &context) override;

  /** Full-sequence forward (training path); decode uses incremental_forwarding.
   */
  void forwarding(nntrainer::RunLayerContext &context, bool training) override;

  /**
   * @brief Incremental forward.
   *  - Prefill  (to - from > 1): compute positions [0, to) and save state.
   *  - Decode   (to - from == 1): O(1) single-token using cached state.
   */
  void incremental_forwarding(nntrainer::RunLayerContext &context,
                              unsigned int from, unsigned int to,
                              bool training) override;

  void calcDerivative(nntrainer::RunLayerContext &context) override;
  void calcGradient(nntrainer::RunLayerContext &context) override;

  /**
   * @brief Backwarding is supported for LoRA training through the conv.
   * The conv weight itself stays frozen; calcDerivative propagates dX to the
   * trainable projection feeding the conv.
   */
  bool supportBackwarding() const override { return true; }

  void updateTensorsByInputDimensions(
    nntrainer::RunLayerContext &context,
    std::vector<nntrainer::TensorDim> input_dimensions) override;

  const std::string getType() const override { return type; }

  void exportTo(nntrainer::Exporter &exporter,
                const ml::train::ExportMethods &method) const override;

  void setProperty(const std::vector<std::string> &values) override;

  inline static const std::string type = "causal_conv1d";

private:
  /** Default kernel size 3 (matches LFM2 conv_L_cache=3). */
  unsigned int kernel_size = 3;
  /** Parsed "kernel_size" property (validated in setProperty). */
  std::tuple<nntrainer::props::KernelSize> conv_props;
  static constexpr size_t SINGLE_INOUT_IDX = 0;

  enum WeightIdx { weight = 0, NUM_WEIGHTS };
  enum TensorIdx { conv_state = 0, NUM_TENSORS };

  std::array<unsigned int, NUM_WEIGHTS> weight_idx;
  std::array<unsigned int, NUM_TENSORS> tensor_idx;

  void validateInputShape(const nntrainer::TensorDim &input_dim) const;
};

} // namespace causallm

#endif // __CAUSAL_LM_CAUSAL_CONV1D_LAYER_H__
