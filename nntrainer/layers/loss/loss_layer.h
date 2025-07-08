// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   loss_layer.h
 * @date   12 June 2020
 * @brief  This is Loss Layer Class of Neural Network
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __LOSS_LAYER_H__
#define __LOSS_LAYER_H__
#ifdef __cplusplus

#include <layer_devel.h>

#include <tensor.h>

namespace nntrainer {

/**
 * @class   LossLayer
 * @brief   loss layer
 */
class LossLayer : public Layer {
public:
  /**
   * @brief     Destructor of Loss Layer
   */
  NNTR_API virtual ~LossLayer() = default;

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  NNTR_API virtual void finalize(InitLayerContext &context) override;

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  NNTR_API virtual void
  setProperty(const std::vector<std::string> &values) override;

  /**
   * @copydoc Layer::supportBackwarding()
   */
  NNTR_API virtual bool supportBackwarding() const override { return true; }

  /**
   * @copydoc Layer::requireLabel()
   */
  NNTR_API bool requireLabel() const override { return true; }

protected:
  /**
   * @brief     update loss
   * @param     context Run context to update loss in
   * @param     l Tensor data to calculate
   */
  NNTR_API void updateLoss(RunLayerContext &context, const Tensor &l);

  /**
   * @brief     update return derivative with loss scale
   * @param     context Run context to update
   * @param     return_dev Tensor data to calculate
   */
  NNTR_API void applyLossScale(RunLayerContext &context, Tensor &l);

  Tensor
    l; /**< loss tensor to store intermediate value to calculate loss value */
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __LOSS_LAYER_H__ */
