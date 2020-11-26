// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   pow.h
 * @date   16 November 2020
 * @brief  This file contains the simple pow2 layer which squares input
 * elements.
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __POW_LAYER_H__
#define __POW_LAYER_H__

/// @todo migrate these to API
#include <layer_internal.h>
#include <tensor.h>

namespace custom {

/// @todo inherit this to API
// class PowLayer : public ml::train::Layer {
class PowLayer : public nntrainer::Layer {
public:
  /**
   * @brief Construct a new Pow Layer object that does elementwise power
   *
   * @param exponent_ exponent
   */
  PowLayer(float exponent_ = 1) : Layer(), exponent(exponent_) {}

  /**
   * @brief Destroy the Pow Layer object
   *
   */
  ~PowLayer() {}

  using nntrainer::Layer::setProperty;

  /**
   * @brief     set Property of layer, currently only "exponent is accepted"
   * @param[in] values values of property
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int setProperty(std::vector<std::string> values);

  /**
   * @brief initializing nntrainer
   *
   * @return int ML_ERROR_NONE if success
   */
  int initialize();

  /**
   * @brief nntrainer forwarding function
   *
   * @param in input tensors
   */
  void forwarding(nntrainer::sharedConstTensors in = {});

  /**
   * @brief     calc the derivative to be passed to the previous layer
   * @param[in] in List of Derivative Tensor from the next layer
   * @retval    Derivative List of Tensor for the previous layer
   */
  void calcDerivative(nntrainer::sharedConstTensors in = {});

  /**
   * @brief Get the Type object
   *
   * @return const std::string
   */
  const std::string getType() const { return PowLayer::type; }

  static const std::string type;

private:
  float exponent;
};

} // namespace custom

#endif /* __POW_LAYER_H__ */
