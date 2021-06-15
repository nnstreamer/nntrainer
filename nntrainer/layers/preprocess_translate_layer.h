// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   preprocess_layer.h
 * @date   31 December 2020
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is Preprocess Translate Layer Class for Neural Network
 *
 */

#ifndef __PREPROCESS_TRANSLATE_LAYER_H__
#define __PREPROCESS_TRANSLATE_LAYER_H__
#ifdef __cplusplus

#include <random>

#if defined(ENABLE_DATA_AUGMENTATION_OPENCV)
#include <opencv2/highgui/highgui.hpp>
#endif

#include <layer_internal.h>
#include <tensor.h>

namespace nntrainer {

/**
 * @class   PreprocessTranslate Layer
 * @brief   Preprocess Translate Layer
 */
class PreprocessTranslateLayer : public LayerV1 {
public:
  /**
   * @brief     Constructor of Preprocess Translate Layer
   */
  template <typename... Args>
  PreprocessTranslateLayer(Args... args) :
    LayerV1(args...),
    translation_factor(0.0),
    epsilon(1e-5) {
    trainable = false;
  }

  /**
   * @brief     Destructor of Preprocess Translate Layer
   */
  ~PreprocessTranslateLayer(){};

  /**
   *  @brief  Move constructor of PreprocessLayer.
   *  @param[in] PreprocessLayer &&
   */
  PreprocessTranslateLayer(PreprocessTranslateLayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs PreprocessLayer to be moved.
   */
  PreprocessTranslateLayer &operator=(PreprocessTranslateLayer &&rhs) = default;

  /**
   * @brief     initialize layer
   * @param[in] last last layer
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int initialize(Manager &manager) override;

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

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override {
    return PreprocessTranslateLayer::type;
  }

  using LayerV1::setProperty;

  /**
   * @copydoc Layer::setProperty(const PropertyType type, const std::string
   * &value)
   */
  void setProperty(const PropertyType type,
                   const std::string &value = "") override;

  static const std::string type;

private:
  float translation_factor;
  float epsilon;

  std::mt19937 rng; /**< random number generator */
  std::uniform_real_distribution<float>
    translate_dist; /**< uniform random distribution */

#if defined(ENABLE_DATA_AUGMENTATION_OPENCV)
  cv::Mat affine_transform_mat;
  cv::Mat input_mat, output_mat;

#endif
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __PREPROCESS_TRANSLATE_LAYER_H__ */
