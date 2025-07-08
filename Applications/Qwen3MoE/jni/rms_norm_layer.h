// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file rms_norm_layer.h
 * @date 09 January 2025
 * @brief RMS Normalization Layer for Qwen3 MoE
 * @see https://github.com/nnstreamer/nntrainer
 * @author Samsung Electronics
 * @bug No known bugs except for NYI items
 */

#ifndef __QWEN3_RMS_NORM_LAYER_H__
#define __QWEN3_RMS_NORM_LAYER_H__

#include <layer_impl.h>
#include <base_properties.h>

namespace nntrainer {

/**
 * @class   RMSNormLayer
 * @brief   Root Mean Square Normalization Layer for Qwen3 MoE
 * 
 * RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight
 * 
 * This is different from LayerNorm which also subtracts the mean.
 * RMSNorm only normalizes by the root mean square.
 */
class RMSNormLayer final : public LayerImpl {
public:
  /**
   * @brief Construct a new RMS Norm Layer object
   */
  RMSNormLayer();

  /**
   * @brief Destroy the RMS Norm Layer object
   */
  ~RMSNormLayer() = default;

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  void finalize(InitLayerContext &context) override;

  /**
   * @copydoc Layer::forwarding(RunLayerContext &context, bool training)
   */
  void forwarding(RunLayerContext &context, bool training) override;

  /**
   * @copydoc Layer::calcDerivative(RunLayerContext &context)
   */
  void calcDerivative(RunLayerContext &context) override;

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, ml_train_format_e format)
   */
  void exportTo(Exporter &exporter,
                const ml_train_format_e format) const override;

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override { return RMSNormLayer::type; };

  /**
   * @copydoc Layer::supportBackwarding()
   */
  bool supportBackwarding() const override { return true; }

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  void setProperty(const std::vector<std::string> &values) override;

  /**
   * @copydoc Layer::clone()
   */
  std::unique_ptr<Layer> clone() const override;

  inline static const std::string type = "rms_norm";

private:
  static constexpr unsigned int WEIGHT_IDX = 0; /**< weight index */

  /**
   * @brief RMS normalization properties
   */
  std::tuple<props::Epsilon> rms_norm_props;

  /**
   * @brief Calculate RMS (Root Mean Square)
   * @param input Input tensor
   * @param eps Epsilon for numerical stability
   * @return RMS value
   */
  float calculateRMS(const Tensor &input, float eps) const;

  /**
   * @brief Apply RMS normalization
   * @param input Input tensor
   * @param weight Weight tensor
   * @param eps Epsilon value
   * @param output Output tensor
   */
  void applyRMSNorm(const Tensor &input, const Tensor &weight, 
                    float eps, Tensor &output) const;
};

} // namespace nntrainer

#endif /* __QWEN3_RMS_NORM_LAYER_H__ */