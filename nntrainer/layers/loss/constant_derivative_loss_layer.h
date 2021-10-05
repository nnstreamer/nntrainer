// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file constant_deriv_loss_layer.h
 * @date 05 Oct 2021
 * @brief This patch contains constant derivative loss implementation
 * @note This is special type of loss to feed an arbitrary derivative value to
 * the last layer.
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug No known bugs except for NYI items
 */
#ifndef __CONSTANT_DERIVATIVE_LOSS_LAYER_H__
#define __CONSTANT_DERIVATIVE_LOSS_LAYER_H__
#ifdef __cplusplus

#include <loss_layer.h>

namespace nntrainer {

/**
 * @class   ConstantDerivativeLossLayer
 * @brief   Constant Loss Layer
 */
class ConstantDerivativeLossLayer final : public LossLayer {
public:
  /**
   * @brief     Constructor of Constant Loss Layer
   */
  ConstantDerivativeLossLayer();

  /**
   * @brief     Destructor of MSE Loss Layer
   */
  ~ConstantDerivativeLossLayer();

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  void setProperty(const std::vector<std::string> &values) override;

  /**
   * @copydoc Layer::forwarding(RunLayerContext &context, bool training)
   */
  void forwarding(RunLayerContext &context, bool training) override;

  /**
   * @copydoc Layer::calcDerivative(RunLayerContext &context)
   */
  void calcDerivative(RunLayerContext &context) override;

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override {
    return ConstantDerivativeLossLayer::type;
  };

  inline static const std::string type = "constant_derivative";
};
} // namespace nntrainer

#endif /* __cplusplus */

#endif // __CONSTANT_DERIVATIVE_LOSS_LAYER_H__
