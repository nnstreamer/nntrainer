// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   kld_loss_layer.h
 * @date   25 November 2021
 * @brief  KLD (Kullback-Leibler Divergence) loss implementation
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */
#ifndef __KLD_LOSS_LAYER_H__
#define __KLD_LOSS_LAYER_H__

#ifdef __cplusplus

#include <loss_layer.h>
#include <string>
#include <vector>

namespace nntrainer {

/**
 * @class   KLD (Kullback-Leibler Divergence) Loss layer
 * @brief   kld loss layer
 */
class KLDLossLayer : public LossLayer {
public:
  /**
   * @brief     Constructor of Constant Loss Layer
   */
  NNTR_EXPORT KLDLossLayer() : LossLayer() {}

  /**
   * @brief     Destructor of MSE Loss Layer
   */
  NNTR_EXPORT ~KLDLossLayer() = default;

  /**
   * @copydoc Layer::forwarding(RunLayerContext &context, bool training)
   */
  NNTR_EXPORT void forwarding(RunLayerContext &context, bool training) override;

  /**
   * @copydoc Layer::calcDerivative(RunLayerContext &context)
   */
  NNTR_EXPORT void calcDerivative(RunLayerContext &context) override;

  /**
   * @copydoc Layer::getType()
   */
  NNTR_EXPORT const std::string getType() const override { return KLDLossLayer::type; }

  static constexpr const char *type = "kld";
};
} // namespace nntrainer

#endif /* __cplusplus */
#endif // __KLD_LOSS_LAYER_H__
