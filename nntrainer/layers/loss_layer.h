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
class LossLayer : public LayerV1 {
public:
  /**
   * @brief     Constructor of Loss Layer
   */
  template <typename... Args>
  LossLayer(LossType loss_type_ = LossType::LOSS_UNKNOWN, Args... args) :
    LayerV1(args...),
    loss_type(loss_type_) {}

  /**
   * @brief     Destructor of Loss Layer
   */
  ~LossLayer(){};

  /**
   * @copydoc Layer::forwarding(bool training)
   */
  void forwarding(bool training = true) override;

  /**
   * @copydoc Layer::calcDerivative()
   */
  void calcDerivative() override;

  /**
   * @brief     read layer Weight & Bias data from file
   * @param[in] file input file stream
   */
  void read(std::ifstream &file) override {}

  /**
   * @brief     save layer Weight & Bias data from file
   * @param[in] file output file stream
   */
  void save(std::ofstream &file) override {}

  /**
   * @brief     copy layer
   * @param[in] l layer to copy
   */
  void copy(std::shared_ptr<LayerV1> l) override;

  /**
   * @brief     initialize layer
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int initialize(Manager &manager) override;

  /**
   * @copydoc Layer::requireLabel()
   */
  bool requireLabel() const override { return true; }

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override { return LossLayer::type; };

  /**
   * @brief     set loss function
   * @param[in] l loss type
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int setLoss(LossType l);

  static const std::string type;

  /**
   * @brief     get loss function
   * @retval loss type.
   */
  LossType getLossType() const noexcept { return loss_type; }

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
