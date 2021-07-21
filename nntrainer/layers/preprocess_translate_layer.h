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

#include <layer_devel.h>

namespace nntrainer {

/**
 * @class   PreprocessTranslate Layer
 * @brief   Preprocess Translate Layer
 */
class PreprocessTranslateLayer : public Layer {
public:
  /**
   * @brief     Constructor of Preprocess Translate Layer
   */
  PreprocessTranslateLayer() :
    Layer(),
    translation_factor(0.0),
    epsilon(1e-5) {}

  /**
   * @brief     Destructor of Preprocess Translate Layer
   */
  ~PreprocessTranslateLayer() = default;

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
   * @copydoc bool supportBackwarding() const
   */
  bool supportBackwarding() const override { return false; };

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, ExportMethods method)
   */
  void exportTo(Exporter &exporter,
                const ExportMethods &method) const override {}

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override {
    return PreprocessTranslateLayer::type;
  };

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  void setProperty(const std::vector<std::string> &values) override;

  inline static const std::string type = "preprocess_translate";

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

  /**
   * @brief setProperty by type and value separated
   * @param[in] type property type to be passed
   * @param[in] value value to be passed
   * @exception exception::not_supported     when property type is not valid for
   * the particular layer
   * @exception std::invalid_argument invalid argument
   */
  void setProperty(const std::string &type, const std::string &value);
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __PREPROCESS_TRANSLATE_LAYER_H__ */
