// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   preprocess_layer.cpp
 * @date   31 December 2020
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is Preprocess Translate Layer Class for Neural Network
 *
 * @todo   Add support without opencv
 */

#include <random>

#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <parse_util.h>
#include <preprocess_translate_layer.h>
#include <util_func.h>

#if defined(ENABLE_DATA_AUGMENTATION_OPENCV)
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif

namespace nntrainer {

void PreprocessTranslateLayer::finalize(InitLayerContext &context) {
  context.setOutputDimensions(context.getInputDimensions());
  const TensorDim input_dim_0 = context.getInputDimensions()[0];

  rng.seed(getSeed());

  // Made for 3 channel input
  if (translation_factor > epsilon) {
    if (input_dim_0.channel() > 3)
      throw exception::not_supported(
        "Preprocess translate layer not supported for over 3 channels");
    translate_dist = std::uniform_real_distribution<float>(-translation_factor,
                                                           translation_factor);

#if defined(ENABLE_DATA_AUGMENTATION_OPENCV)
    affine_transform_mat = cv::Mat::zeros(2, 3, CV_32FC1);
    affine_transform_mat.at<float>(0, 0) = 1;
    affine_transform_mat.at<float>(1, 1) = 1;

    input_mat =
      cv::Mat::zeros(input_dim_0.height(), input_dim_0.width(), CV_32FC3);
    output_mat =
      cv::Mat::zeros(input_dim_0.height(), input_dim_0.width(), CV_32FC3);
#else
    throw exception::not_supported(
      "Preprocess translate layer is not supported without opencv");
#endif
  }
}

void PreprocessTranslateLayer::setProperty(
  const std::vector<std::string> &values) {
  /// @todo: deprecate this in favor of loadProperties
  for (unsigned int i = 0; i < values.size(); ++i) {
    std::string key;
    std::string value;
    std::stringstream ss;

    if (getKeyValue(values[i], key, value) != ML_ERROR_NONE) {
      throw std::invalid_argument("Error parsing the property: " + values[i]);
    }

    if (value.empty()) {
      ss << "value is empty: key: " << key << ", value: " << value;
      throw std::invalid_argument(ss.str());
    }

    /// @note this calls derived setProperty if available
    setProperty(key, value);
  }
}

void PreprocessTranslateLayer::setProperty(const std::string &type_str,
                                           const std::string &value) {
  using PropertyType = nntrainer::Layer::PropertyType;
  int status = ML_ERROR_NONE;
  nntrainer::Layer::PropertyType type =
    static_cast<nntrainer::Layer::PropertyType>(parseLayerProperty(type_str));

  switch (type) {
  case PropertyType::random_translate: {
    status = setFloat(translation_factor, value);
    translation_factor = std::abs(translation_factor);
    throw_status(status);
  } break;
  default:
    std::string msg =
      "[PreprocessTranslateLayer] Unknown Layer Property Key for value " +
      std::string(value);
    throw exception::not_supported(msg);
  }
}

void PreprocessTranslateLayer::forwarding(RunLayerContext &context,
                                          bool training) {
  if (!training) {
    for (unsigned int idx = 0; idx < context.getNumInputs(); idx++) {
      /** TODO: tell the graph to not include this when not training */
      context.getOutput(idx) = context.getInput(idx);
    }

    return;
  }

  for (unsigned int idx = 0; idx < context.getNumInputs(); idx++) {
    Tensor &hidden_ = context.getOutput(idx);
    Tensor &input_ = context.getInput(idx);
    const TensorDim input_dim = input_.getDim();

    if (translation_factor < epsilon) {
      hidden_ = input_;
      continue;
    }

#if defined(ENABLE_DATA_AUGMENTATION_OPENCV)
    for (unsigned int b = 0; b < input_dim.batch(); b++) {

      /** random translation */
      float translate_x = translate_dist(rng) * input_dim.width();
      float translate_y = translate_dist(rng) * input_dim.height();
      affine_transform_mat.at<cv::Vec2f>(0, 0)[2] = translate_x;
      affine_transform_mat.at<cv::Vec2f>(1, 0)[2] = translate_y;

      for (unsigned int c = 0; c < input_dim.channel(); c++)
        for (unsigned int h = 0; h < input_dim.height(); h++)
          for (unsigned int w = 0; w < input_dim.width(); w++)
            input_mat.at<cv::Vec3f>(h, w)[c] = input_.getValue(b, c, h, w);

      cv::warpAffine(input_mat, output_mat, affine_transform_mat,
                     output_mat.size(), cv::WARP_INVERSE_MAP,
                     cv::BORDER_REFLECT);

      for (unsigned int c = 0; c < input_dim.channel(); c++)
        for (unsigned int h = 0; h < input_dim.height(); h++)
          for (unsigned int w = 0; w < input_dim.width(); w++)
            input_.setValue(b, c, h, w, output_mat.at<cv::Vec3f>(h, w)[c]);
    }

    hidden_ = input_;
#else
    throw exception::not_supported(
      "Preprocess translate layer is not supported without opencv");
#endif
  }
}

void PreprocessTranslateLayer::calcDerivative(RunLayerContext &context) {
  throw exception::not_supported(
    "calcDerivative for preprocess layer is not supported");
}

} /* namespace nntrainer */
