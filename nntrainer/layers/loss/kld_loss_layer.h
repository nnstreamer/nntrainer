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
class KLDLossLayer final : public LossLayer {
public:
  /**
   * @brief     Constructor of Constant Loss Layer
   */
  KLDLossLayer();

  /**
   * @brief     Destructor of MSE Loss Layer
   */
  ~KLDLossLayer();

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
  const std::string getType() const override { return KLDLossLayer::type; }

  /**
   * @copydoc Layer::setBatch(RunLayerContext &context, unsigned int batch)
   */
  void setBatch(nntrainer::RunLayerContext &context,
                unsigned int batch) override;

  inline static const std::string type = "kld";

private:
  unsigned before_sum_idx;
  unsigned temp_idx;
};
} // namespace nntrainer

#endif /* __cplusplus */

#endif // __KLD_LOSS_LAYER_H__
