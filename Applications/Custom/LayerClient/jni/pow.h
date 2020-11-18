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

#include <layer.h>
#include <tensor.h>

namespace custom {
class PowLayer : public ml::train::Layer {
public:
  /**
   * @brief Construct a new Pow Layer object that does elementwise power
   *
   * @param exponent_ exponent
   */
  PowLayer(float exponent_ = 1) : exponent(exponent_) {}

  /**
   * @brief Destroy the Pow Layer object
   *
   */
  ~PowLayer();

  /**
   * @brief     set Property of layer
   * @param[in] values values of property
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int setProperty(std::vector<std::string> values) {
    /**< NYI */
    return 1;
  }

  /**
   * @brief Set the Property by propertyType, this is not used in this demo
   *
   * @param type property type
   * @param value value
   */
  void setProperty(const ml::train::Layer::PropertyType type,
                   const std::string &value = "") {
    /**< NOT USED */
  }

  /**
   * @brief check if hyperparameter is valid
   *
   * @return int ML_ERROR_NONE if successful
   */
  int checkValidataion() { return ML_ERROR_NONE; }

  /**
   * @brief Get the Loss bound to the object
   *
   * @return float
   */
  float getLoss() { return 0.0f; }

  /**
   * @brief nntrainer forwarding function
   *
   * @param in input tensors
   * @return nntrainer::sharedConstTensors output tensors
   */
  nntrainer::sharedConstTensors forwarding(nntrainer::sharedConstTensors in) {
    return in;
  }

  /**
   * @brief nntrainer backwaridng function
   *
   * @param in input tensors
   * @param iteration number of iterations
   * @return nntrainer::sharedConstTensors output tensors
   */
  nntrainer::sharedConstTensors backwarding(nntrainer::sharedConstTensors in,
                                            int iteration) {
    return in;
  }

  /**
   * @brief initialize function
   *
   * @return int ML_ERROR_NONE if successful
   */
  int initialize() { return 1; }

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

const std::string PowLayer::type = "pow";

} // namespace custom

#endif /* __POW_LAYER_H__ */
