// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file qwen3_moe_mlp_layer.h
 * @date 09 January 2025
 * @brief Qwen3 MoE MLP Layer
 * @see https://github.com/nnstreamer/nntrainer
 * @author Samsung Electronics
 * @bug No known bugs except for NYI items
 */

#ifndef __QWEN3_MOE_MLP_LAYER_H__
#define __QWEN3_MOE_MLP_LAYER_H__

#include <layer_impl.h>
#include <base_properties.h>

namespace nntrainer {

/**
 * @class   Qwen3MoeMlpLayer
 * @brief   MLP Layer for Qwen3 MoE
 * 
 * This layer implements the MLP structure used in Qwen3:
 * - Gate projection: Linear(hidden_size, intermediate_size)
 * - Up projection: Linear(hidden_size, intermediate_size)  
 * - SiLU activation applied to gate projection
 * - Element-wise multiplication: gate_output * up_output
 * - Down projection: Linear(intermediate_size, hidden_size)
 */
class Qwen3MoeMlpLayer final : public LayerImpl {
public:
  /**
   * @brief Construct a new Qwen3 MoE MLP Layer object
   */
  Qwen3MoeMlpLayer();

  /**
   * @brief Destroy the Qwen3 MoE MLP Layer object
   */
  ~Qwen3MoeMlpLayer() = default;

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
  const std::string getType() const override { return Qwen3MoeMlpLayer::type; };

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

  inline static const std::string type = "qwen3_moe_mlp";

private:
  static constexpr unsigned int GATE_WEIGHT_IDX = 0;   /**< gate projection weight */
  static constexpr unsigned int UP_WEIGHT_IDX = 1;     /**< up projection weight */  
  static constexpr unsigned int DOWN_WEIGHT_IDX = 2;   /**< down projection weight */

  /**
   * @brief MLP layer properties
   */
  struct MlpProps {
    unsigned int hidden_size = 2048;
    unsigned int intermediate_size = 6144;
  } mlp_props;

  /**
   * @brief Apply SiLU activation function
   * @param input Input tensor
   * @param output Output tensor
   */
  void applySiLU(const Tensor &input, Tensor &output) const;

  /**
   * @brief Apply SiLU derivative
   * @param input Input tensor
   * @param output Output tensor
   */
  void applySiLUDerivative(const Tensor &input, Tensor &output) const;

  /**
   * @brief SiLU function implementation
   * @param x Input value
   * @return SiLU(x)
   */
  static float silu(float x);

  /**
   * @brief SiLU derivative implementation  
   * @param x Input value
   * @return SiLU'(x)
   */
  static float silu_derivative(float x);

  /**
   * @brief Sigmoid function implementation
   * @param x Input value
   * @return sigmoid(x)
   */
  static float sigmoid(float x);
};

} // namespace nntrainer

#endif /* __QWEN3_MOE_MLP_LAYER_H__ */