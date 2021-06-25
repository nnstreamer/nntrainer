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

#include <layer_internal.h>
#include <tensor.h>

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
   * @brief     Constructor of Loss Layer
   */
  LossLayer() : Layer(), loss_type(LossType::LOSS_UNKNOWN) {}

  /**
   * @brief     Destructor of Loss Layer
   */
  ~LossLayer() = default();

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
   * @copydoc Layer::calcGradient(RunLayerContext &context)
   */
  void calcGradient(RunLayerContext &context) override;

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, ExportMethods method)
   */
  void exportTo(Exporter &exporter, const ExportMethods &method) const override;

  /**
   * @brief     set loss function
   * @param[in] l loss type
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int setLoss(LossType l);

  /**
   * @copydoc Layer::requireLabel()
   */
  bool requireLabel() const override { return true; }

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override { return LossLayer::type; };

  static const std::string type;

private:
  LossType loss_type; /**< loss type of loss layer */

  /**
   * @brief     update loss
   * @param[in] l Tensor data to calculate
   */
  void updateLoss(const Tensor &l);
};
} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __LOSS_LAYER_H__ */
