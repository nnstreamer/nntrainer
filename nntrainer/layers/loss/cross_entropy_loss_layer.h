// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   cross_entropy_loss_layer.h
 * @date   24 June 2021
 * @brief  This is Cross Entropy Loss Layer Class of Neural Network
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __CROSS_ENTROPY_LOSS_LAYER_H__
#define __CROSS_ENTROPY_LOSS_LAYER_H__
#ifdef __cplusplus

#include <loss_layer.h>
#include <nntrainer_error.h>

namespace nntrainer {

/**
 * @class   CrossEntropyLossLayer
 * @brief   Cross Entropy Loss Layer
 */
class CrossEntropyLossLayer : public LossLayer {
public:
  /**
   * @brief     Constructor of Cross Entropy Loss Layer
   */
  CrossEntropyLossLayer() : LossLayer() {}

  /**
   * @brief     Destructor of Cross Entropy Loss Layer
   */
  ~CrossEntropyLossLayer() = default;

  /**
   * @copydoc Layer::forwarding(RunLayerContext &context, bool training)
   */
  void finalize(InitLayerContext &context) override {
    throw exception::not_supported(
      "Cross Entropy not supported without softmax or sigmoid");
  }

  /**
   * @copydoc Layer::forwarding(RunLayerContext &context, bool training)
   */
  void forwarding(RunLayerContext &context, bool training) override {
    throw exception::not_supported(
      "Cross Entropy not supported without softmax or sigmoid");
  }

  /**
   * @copydoc Layer::calcDerivative(RunLayerContext &context)
   */
  void calcDerivative(RunLayerContext &context) override {
    throw exception::not_supported(
      "Cross Entropy not supported without softmax or sigmoid");
  }

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override {
    return CrossEntropyLossLayer::type;
  };

  inline static const std::string type = "cross";
};
} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __CROSS_ENTROPY_LOSS_LAYER_H__ */
