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

/// @todo migrate these to API(#987)
#include <layer_internal.h>
#include <manager.h>

#include <tensor.h>

namespace custom {

/**
 * @brief layer class that calculates f(x) = x ^ exponent (exponent is
 * configurable by PowLayer::setProperty)
 *
 */
class PowLayer : public nntrainer::LayerV1 {
public:
  /**
   * @brief Construct a new Pow Layer object that does elementwise power
   *
   * @param exponent_ exponentLayerV1
   */
  PowLayer(float exponent_ = 1) : LayerV1(), exponent(exponent_) {}

  /**LayerV1
   * @brief Destroy the Pow Layer object
   *
   */
  ~PowLayer() {}

  using nntrainer::LayerV1::setProperty;

  /**LayerV1
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
  int initialize(nntrainer::Manager &manager);

  /**
   * @brief nntrainer forwarding function
   * @param[in] training true if forwarding is on training
   */
  void forwarding(bool training = true) override;

  /**
   * @brief     calc the derivative to be passed to the previous layer
   */
  void calcDerivative();

  /**
   * @brief Get the Type object
   *
   * @return const std::string
   */
  const std::string getType() const { return PowLayer::type; }

  inline static const std::string type = "pow";

private:
  float exponent;
};

} // namespace custom

#endif /* __POW_LAYER_H__ */
