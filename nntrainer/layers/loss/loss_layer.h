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

namespace nntrainer {


/**
 * @brief     Enumeration of loss function type
 */
enum class LossType {
  LOSS_MSE,             /** Mean Squared Error */
  LOSS_ENTROPY,         /** Cross Entropy */
  LOSS_NONE,            /** No loss for this model */
  LOSS_ENTROPY_SIGMOID, /** Cross Entropy amalgamated with sigmoid for stability
                         */
  LOSS_ENTROPY_SOFTMAX, /** Cross Entropy amalgamated with softmax for stability
                         */
  LOSS_UNKNOWN          /** Unknown */
};

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
  virtual void finalize(InitLayerContext &context) override {
    context.setOutputDimensions(context.getInputDimensions());
  }

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  virtual void setProperty(const std::vector<std::string> &values) override {}

  /**
   * @copydoc Layer::supportBackwarding()
   */
  virtual bool supportBackwarding() const override { return true; }

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

  Tensor l; /**< loss tensor to store intermediate value to calculate loss value */
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __LOSS_LAYER_H__ */
