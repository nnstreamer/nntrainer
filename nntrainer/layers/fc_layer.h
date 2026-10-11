// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   fc_layer.h
 * @date   14 May 2020
 * @brief  This is Fully Connected Layer Class of Neural Network
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @author Pranjal Thapliyal <p.thapliyal@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __FC_LAYER_H__
#define __FC_LAYER_H__
#ifdef __cplusplus

#include <acti_func.h>
#include <common_properties.h>
#include <layer_impl.h>

#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

namespace nntrainer {

class ComputeOps;

/**
 * @class   FullyConnecedLayer
 * @brief   fully connected layer
 */
class FullyConnectedLayer : public LayerImpl {
public:
  /**
   * @brief     Constructor of Fully Connected Layer
   */
  FullyConnectedLayer();

  /**
   * @brief     Destructor of Fully Connected Layer
   */
  ~FullyConnectedLayer() = default;

  /**
   *  @brief  Move constructor.
   *  @param[in] FullyConnected &&
   */
  FullyConnectedLayer(FullyConnectedLayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs FullyConnectedLayer to be moved.
   */
  FullyConnectedLayer &operator=(FullyConnectedLayer &&rhs) = default;

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  void finalize(InitLayerContext &context) override;

  /**
   * @copydoc Layer::forwarding(RunLayerContext &context, bool training)
   */
  void forwarding(RunLayerContext &context, bool training) override;

  /**
￼   * @copydoc Layer::incremental_forwarding(RunLayerContext &context, unsigned
￼   * int from, unsigned int to, bool training)
￼   */
  void incremental_forwarding(RunLayerContext &context, unsigned int from,
                              unsigned int to, bool training) override;

  /**
   * @copydoc Layer::calcDerivative(RunLayerContext &context)
   */
  void calcDerivative(RunLayerContext &context) override;

  /**
   * @copydoc Layer::calcGradient(RunLayerContext &context)
   * @note
   * [note for LoRA] implicit calcDerivative is implicitly applied.
   * The weight is already updated with the LoRA's (W = W + W_lora)
   */
  void calcGradient(RunLayerContext &context) override;

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, ml::train::ExportMethods
   * method)
   */
  void exportTo(Exporter &exporter,
                const ml::train::ExportMethods &method) const override;

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override {
    return FullyConnectedLayer::type;
  };

  /**
   * @copydoc Layer::supportBackwarding()
   */
  bool supportBackwarding() const override { return true; }

  /**
   * @copydoc Layer::setProperty(const PropertyType type, const std::string
   * &value)
   */
  void setProperty(const std::vector<std::string> &values) override;

  /**
   * @copydoc Layer::setBatch(RunLayerContext &context, unsigned int batch)
   */
  void setBatch(nntrainer::RunLayerContext &context,
                unsigned int batch) override;

  /**
   * @copydoc Layer::pack(RunLayerContext &context)
   */
  void pack(RunLayerContext &context) override;

  static constexpr const char *type = "fully_connected";

  /** Per-layer Q4_0 QAT calibration snapshot, updated on every QAT forward. */
  struct LoRAQATStats {
    float a_min = 0, a_max = 0, a_scale = 0;
    float b_min = 0, b_max = 0, b_scale = 0;
    bool valid = false;
  };

  /**
   * @brief Look up a layer's most recent QAT calibration stats by name.
   * Thread-safe. Returns a default-constructed (valid=false) instance if the
   * named layer hasn't run a QAT forward pass yet.
   */
  static LoRAQATStats getRegisteredStats(const std::string &layer_name);

  /**
   * @brief Look up a layer's current per-block EMA scales (loraA, loraB),
   * each indexed in N x K layout to match fakeQuantizeQ4_0's block
   * numbering. Thread-safe. Returns empty vectors if the named layer hasn't
   * run a QAT forward pass yet.
   */
  static std::pair<std::vector<float>, std::vector<float>>
  getRegisteredBlockScales(const std::string &layer_name);

private:
  static std::mutex s_registry_mutex;
  static std::unordered_map<std::string, LoRAQATStats> s_qat_registry;
  static std::unordered_map<std::string,
                            std::pair<std::vector<float>, std::vector<float>>>
    s_block_d_registry;

  float lora_scaling;
  std::tuple<props::Unit, props::LoraRank, props::LoraAlpha, props::LoraQAT,
             props::LoraWeightQ4, props::LoraQatEmaMomentum, props::LoraGPU,
             props::FusedActivation, props::PlanLastRowOnly>
    fc_props;                             /**< fc layer properties :
                                                unit - number of output neurons,
                                                lora_rank - rank of lora (optional)
                                                lora_scaling - scaling factor of LoRA apply, i.e.,
                                             lora_scaling = alpha / lora_rank
                                                fused_activation - inline activation
                                             applied after GEMM+bias
                                                plan_last_row_only - plan the output at height 1
                                             because only the last row is ever produced
                                                lora_qat - enable per-block Q4_0 fake-quant
                                                lora_weight_q4 - store/load adapters as real Q4_0
                                                lora_qat_ema_momentum - EMA momentum for per-block
                                             QAT scale update (optional, default 0.1)
                                                lora_gpu - offload FP32 GEMMs to the accelerator */
  std::array<unsigned int, 2> weight_idx; /**< indices of the weights */
  std::array<unsigned int, 4> lora_idx;   /**< indices of the lora weights */
  std::unique_ptr<nntrainer::Quantizer> quantizer;
  bool skip_prefill = false;
  ActiFunc acti_func; /**< inline fused activation; inert unless
                           fused_activation is set by the FusionRealizer */

  float momentum = 0.1f; /**< EMA momentum for per-block QAT scale update */
  ComputeOps *gemm_ops_ =
    nullptr; /**< accelerator table for lora_gpu, else null */

  /**
   * @brief C = op(A) * op(B) + beta * C for the LoRA FP32 GEMMs.
   *
   * With lora_gpu the product goes to the accelerator table's host-pointer
   * GEMM (see ComputeOps::try_sgemm_fp32_host) when every operand is a
   * contiguous FP32 matrix and the table accepts the shape; otherwise (and
   * for the accumulate case beta != 0 beyond a plain add) it is exactly
   * Tensor::dot, so the CPU path is bit-for-bit what it was.
   */
  void gemm(const Tensor &A, const Tensor &B, Tensor &C, bool trans_a,
            bool trans_b, float beta = 0.0f);
  std::string
    layer_name_; /**< captured in finalize(); keys the QAT registries */

  /** Per-block EMA scales for loraA/loraB (one float per 32-element block,
   * N x K layout), lazily sized on the first QAT forward pass. */
  std::vector<float> lora_a_block_d;
  std::vector<float> lora_b_block_d;

  /** Fake-quantized loraA/loraB from the most recent forward pass, cached
   * for calcDerivative/calcGradient's straight-through backward. */
  Tensor a_fq, b_fq;

  /** Frozen base weight W (K x N) re-quantized to Q8_0 along N (the dX
   * reduction axis) and repacked q8_0x4 (4x8 interleave), built once on the
   * first LoRA calcDerivative. Lets dX = dY * W^T run as an int8 GEMM on the
   * same i8mm kernel family the forward uses, instead of an FP32 GEMM over a
   * per-call dequantized copy. Empty when the FP32 path is used. */
  std::vector<uint8_t> dx_wt_q8x4_;

  /** QS4CX base only: the same int4 codes transposed to (N_out x K_in) with
   * unit scales -- the per-output-channel scale s is folded into dY instead
   * (dX = (dY * s) . Q^T), so the copy is lossless and dX runs on the same
   * KleidiAI qai8dxp x qsi4cxp GEMM as the forward. */
  Tensor dx_wt_qs4t_;
  Tensor dx_scaled_dy_; /**< (dY * s) scratch */
  /** calcGradient left s * dY . B^T (the loraTmp gradient) for this step's
   * calcDerivative to reuse for the LoRA part of dX */
  bool lora_dtmp_ready_ = false;
  /** private copy of that gradient: the loraTmp tensor's lifespan
   * (FORWARD_GRAD) does not cover calcDerivative */
  std::vector<float> lora_dtmp_copy_;

  /**
   * @brief Per-block Q4_0 fake-quantization with EMA-tracked block scales.
   *
   * Blocks are defined in N x K layout (the transposed layout the Q4_0
   * GEMM kernel and the save-time codec use), so the EMA scales computed
   * here line up exactly with the blocks force-fed at save time.
   *
   * Training: computes a fresh per-block scale, folds it into the running
   * EMA (bootstrapping on first use), and quantizes with the updated EMA
   * scale. Validation/inference: quantizes with the current EMA without
   * updating it. Backward is a straight-through estimator: the gradient
   * passes through the round+clamp unchanged.
   *
   * @param x the FP32 tensor to fake-quantize (loraA or loraB)
   * @param block_d persistent per-block EMA scale storage for this tensor
   * @param training whether to update the EMA (true) or just apply it (false)
   */
  Tensor fakeQuantizeQ4_0(const Tensor &x, std::vector<float> &block_d,
                          bool training);
};
} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __FC_LAYER_H__ */
