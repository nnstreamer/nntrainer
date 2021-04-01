// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file	time_dist.h
 * @date	01 April 2021
 * @brief	This is Time Distributed Layer Class of Neural Network
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#ifndef __TIME_DIST_H__
#define __TIME_DIST_H__
#ifdef __cplusplus

#include <tensor.h>
#include <time_dist.h>

namespace nntrainer {

/**
 * @class   TimeDistLayer
 * @brief   Time Distribution Layer
 */
class TimeDistLayer : public Layer {
public:
  /**
   * @brief     Constructor of Time Distribution Layer
   */
  template <typename... Args> TimeDistLayer(Args... args) : Layer(args...) {}

  /**
   * @brief     Destructor of Time Distributed Layer
   */
  ~TimeDistLayer(){};

  /**
   *  @brief  Move constructor.
   *  @param[in] TimeDistLayer &&
   */
  TimeDistLayer(TimeDistLayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs TimeDistLayer to be moved.
   */
  TimeDistLayer &operator=(TimeDistLayer &&rhs) = default;

  /**
   * @copydoc Layer::forwarding(bool training)
   */
  void forwarding(bool training = true) override;

  /**
   * @copydoc Layer::calcDerivative()
   */
  void calcDerivative() override;

  /**
   * @copydoc Layer::calcGradient()
   */
  void calcGradient() override;

  /**
   * @brief     copy layer
   * @param[in] l layer to copy
   */
  void copy(std::shared_ptr<Layer> l) override;

  /**
   * @brief     initialize layer
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int initialize(Manager &manager) override;

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override { return TimeDistLayer::type; };

  static const std::string type;

private:
  /* @brief Layer to be distributed through time */

  std::shared_ptr<Layer> dist_layer;
};
} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __FC_LAYER_H__ */
