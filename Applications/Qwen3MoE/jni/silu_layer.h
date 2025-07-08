// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file silu_layer.h
 * @date 09 January 2025
 * @brief SiLU (Sigmoid Linear Unit) Activation Layer
 * @see https://github.com/nnstreamer/nntrainer
 * @author Samsung Electronics
 * @bug No known bugs except for NYI items
 */

#ifndef __SILU_LAYER_H__
#define __SILU_LAYER_H__

#include <layer_impl.h>

namespace nntrainer {

/**
 * @class   SiLULayer
 * @brief   SiLU (Sigmoid Linear Unit) activation layer
 * 
 * SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
 * Also known as Swish activation function.
 */
class SiLULayer final : public LayerImpl {
public:
  /**
   * @brief Construct a new SiLU Layer object
   */
  SiLULayer();

  /**
   * @brief Destroy the SiLU Layer object
   */
  ~SiLULayer() = default;

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
  const std::string getType() const override { return SiLULayer::type; };

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

  inline static const std::string type = "silu";

private:
  /**
   * @brief Compute SiLU function: x * sigmoid(x)
   * @param x input value
   * @return SiLU(x)
   */
  static float silu(float x);

  /**
   * @brief Compute SiLU derivative: sigmoid(x) * (1 + x * (1 - sigmoid(x)))
   * @param x input value
   * @return SiLU'(x)
   */
  static float silu_derivative(float x);

  /**
   * @brief Compute sigmoid function
   * @param x input value
   * @return sigmoid(x)
   */
  static float sigmoid(float x);
};

} // namespace nntrainer

#endif /* __SILU_LAYER_H__ */