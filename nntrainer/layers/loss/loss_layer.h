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
  virtual ~LossLayer() = default;

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  virtual void finalize(InitLayerContext &context) override;

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  virtual void setProperty(const std::vector<std::string> &values) override;

  /**
   * @copydoc Layer::supportBackwarding()
   */
  virtual bool supportBackwarding() const override { return true; }

  bool supportInPlace() const override { return is_inplace; }

  /**
   * @copydoc Layer::requireLabel()
   */
  bool requireLabel() const override { return true; }

protected:
  /**
   * @brief     update loss
   * @param     context Run context to update loss in
   * @param     l Tensor data to calculate
   */
  void updateLoss(RunLayerContext &context, const Tensor &l);

  /**
   * @brief     update return derivative with loss scale
   * @param     context Run context to update
   * @param     return_dev Tensor data to calculate
   */
  void applyLossScale(RunLayerContext &context, Tensor &l);

  Tensor
    l; /**< loss tensor to store intermediate value to calculate loss value */

  bool is_inplace;
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __LOSS_LAYER_H__ */
