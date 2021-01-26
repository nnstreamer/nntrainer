// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file	preprocess_flip_layer.h
 * @date	20 January 2020
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug		No known bugs except for NYI items
 * @brief	This is Preprocess Random Flip Layer Class for Neural Network
 *
 */

#ifndef __PREPROCESS_FLIP_LAYER_H__
#define __PREPROCESS_FLIP_LAYER_H__
#ifdef __cplusplus

#include <random>

#include <layer_internal.h>
#include <tensor.h>

namespace nntrainer {

/**
 * @class   Preprocess FLip Layer
 * @brief   Preprocess FLip Layer
 */
class PreprocessFlipLayer : public Layer {
public:
  /**
   * @brief     Constructor of Preprocess FLip Layer
   */
  template <typename... Args>
  PreprocessFlipLayer(Args... args) :
    Layer(args...),
    flipdirection(FlipDirection::horizontal_and_vertical) {}

  /**
   * @brief     Destructor of Preprocess FLip Layer
   */
  ~PreprocessFlipLayer(){};

  /**
   *  @brief  Move constructor of PreprocessLayer.
   *  @param[in] PreprocessLayer &&
   */
  PreprocessFlipLayer(PreprocessFlipLayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs PreprocessLayer to be moved.
   */
  PreprocessFlipLayer &operator=(PreprocessFlipLayer &&rhs) = default;

  /**
   * @brief     initialize layer
   * @param[in] last last layer
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int initialize(Manager &manager);

  /**
   * @copydoc Layer::forwarding()
   */
  void forwarding(bool training = true) override;

  /**
   * @copydoc Layer::calcDerivative()
   */
  void calcDerivative() override;

  /**
   * @copydoc Layer::setTrainable(bool train)
   */
  void setTrainable(bool train) override;

  using Layer::setProperty;

  /**
   * @copydoc Layer::setProperty(const PropertyType type, const std::string
   * &value)
   */
  void setProperty(const PropertyType type,
                   const std::string &value = "") override;

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override {
    return PreprocessFlipLayer::type;
  }

  static const std::string type;

private:
  /** String names for the flip direction */
  static const std::string flip_horizontal;
  static const std::string flip_vertical;
  static const std::string flip_horizontal_vertical;

  /**
   * @brief Direction of the flip for data
   */
  enum class FlipDirection { horizontal, vertical, horizontal_and_vertical };

  std::mt19937 rng; /**< random number generator */
  std::uniform_real_distribution<float>
    flip_dist;                 /**< uniform random distribution */
  FlipDirection flipdirection; /**< direction of flip */
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __PREPROCESS_FLIP_LAYER_H__ */
