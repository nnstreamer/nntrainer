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

const std::string PreprocessTranslateLayer::type = "preprocess_translate";

int PreprocessTranslateLayer::initialize(Manager &manager) {
  output_dim = input_dim;

  rng.seed(getSeed());

  // Made for 3 channel input
  if (translation_factor > epsilon) {
    if (input_dim[0].channel() > 3)
      throw exception::not_supported(
        "Preprocess translate layer not supported for over 3 channels");
    translate_dist = std::uniform_real_distribution<float>(-translation_factor,
                                                           translation_factor);

#if defined(ENABLE_DATA_AUGMENTATION_OPENCV)
    affine_transform_mat = cv::Mat::zeros(2, 3, CV_32FC1);
    affine_transform_mat.at<float>(0, 0) = 1;
    affine_transform_mat.at<float>(1, 1) = 1;

    input_mat =
      cv::Mat::zeros(input_dim[0].height(), input_dim[0].width(), CV_32FC3);
    output_mat =
      cv::Mat::zeros(input_dim[0].height(), input_dim[0].width(), CV_32FC3);
#else
    throw exception::not_supported(
      "Preprocess translate layer is not supported without opencv");
#endif
  }

  return ML_ERROR_NONE;
}

void PreprocessTranslateLayer::setProperty(const PropertyType type,
                                           const std::string &value) {
  int status = ML_ERROR_NONE;

  switch (type) {
  case PropertyType::random_translate:
    if (!value.empty()) {
      status = setFloat(translation_factor, value);
      translation_factor = std::abs(translation_factor);
      throw_status(status);
    }
    break;
  default:
    Layer::setProperty(type, value);
    break;
  }
}

void PreprocessTranslateLayer::forwarding(bool training) {
  if (!training) {
    for (unsigned int idx = 0; idx < input_dim.size(); idx++) {
      net_hidden[idx]->getVariableRef() = net_input[idx]->getVariableRef();
    }

    return;
  }

  for (unsigned int idx = 0; idx < input_dim.size(); idx++) {
    Tensor &hidden_ = net_hidden[idx]->getVariableRef();
    Tensor &input_ = net_input[idx]->getVariableRef();

    if (translation_factor < epsilon) {
      hidden_ = input_;
      continue;
    }

#if defined(ENABLE_DATA_AUGMENTATION_OPENCV)
    for (unsigned int b = 0; b < input_dim[idx].batch(); b++) {

      /** random translation */
      float translate_x = translate_dist(rng) * input_dim[idx].width();
      float translate_y = translate_dist(rng) * input_dim[idx].height();
      affine_transform_mat.at<cv::Vec2f>(0, 0)[2] = translate_x;
      affine_transform_mat.at<cv::Vec2f>(1, 0)[2] = translate_y;

      for (unsigned int c = 0; c < input_dim[idx].channel(); c++)
        for (unsigned int h = 0; h < input_dim[idx].height(); h++)
          for (unsigned int w = 0; w < input_dim[idx].width(); w++)
            input_mat.at<cv::Vec3f>(h, w)[c] = input_.getValue(b, c, h, w);

      cv::warpAffine(input_mat, output_mat, affine_transform_mat,
                     output_mat.size(), cv::WARP_INVERSE_MAP,
                     cv::BORDER_REFLECT);

      for (unsigned int c = 0; c < input_dim[idx].channel(); c++)
        for (unsigned int h = 0; h < input_dim[idx].height(); h++)
          for (unsigned int w = 0; w < input_dim[idx].width(); w++)
            input_.setValue(b, c, h, w, output_mat.at<cv::Vec3f>(h, w)[c]);
    }

    hidden_ = input_;
#else
    throw exception::not_supported(
      "Preprocess translate layer is not supported without opencv");
#endif
  }
}

void PreprocessTranslateLayer::calcDerivative() {
  throw exception::not_supported(
    "calcDerivative for preprocess layer is not supported");
}

void PreprocessTranslateLayer::setTrainable(bool train) {
  if (train)
    throw exception::not_supported(
      "Preprocessing layer does not support training");

  Layer::setTrainable(false);
}

} /* namespace nntrainer */
