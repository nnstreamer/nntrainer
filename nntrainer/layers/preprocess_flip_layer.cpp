// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file	preprocess_flip_layer.cpp
 * @date	20 January 2020
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug		No known bugs except for NYI items
 * @brief	This is Preprocess Random Flip Layer Class for Neural Network
 *
 */

#include <random>

#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <parse_util.h>
#include <preprocess_flip_layer.h>
#include <util_func.h>

namespace nntrainer {

const std::string PreprocessFlipLayer::type = "preprocess_flip";

const std::string PreprocessFlipLayer::flip_horizontal = "horizontal";
const std::string PreprocessFlipLayer::flip_vertical = "vertical";
const std::string PreprocessFlipLayer::flip_horizontal_vertical =
  "horizontal_and_vertical";

int PreprocessFlipLayer::initialize(Manager &manager) {
  output_dim = input_dim;

  rng.seed(getSeed());
  flip_dist = std::uniform_real_distribution<float>(0.0, 1.0);

  return ML_ERROR_NONE;
}

void PreprocessFlipLayer::setProperty(const PropertyType type,
                                      const std::string &value) {
  switch (type) {
  case PropertyType::flip_direction:
    if (!value.empty()) {
      if (istrequal(value, flip_horizontal)) {
        flipdirection = FlipDirection::horizontal;
      } else if (istrequal(value, flip_vertical)) {
        flipdirection = FlipDirection::vertical;
      } else if (istrequal(value, flip_horizontal_vertical)) {
        flipdirection = FlipDirection::horizontal_and_vertical;
      } else {
        throw std::invalid_argument("Argument flip direction is invalid");
      }
    }
    break;
  default:
    Layer::setProperty(type, value);
    break;
  }
}

void PreprocessFlipLayer::forwarding() {
  using std::swap;
  bool fliph, flipw;

  for (unsigned int idx = 0; idx < input_dim.size(); idx++) {
    Tensor &hidden_ = net_hidden[idx]->getVariableRef();
    Tensor &input_ = net_input[idx]->getVariableRef();
    unsigned int width = input_dim[idx].width();
    unsigned int height = input_dim[idx].height();

    for (unsigned int b = 0; b < input_dim[idx].batch(); b++) {
      fliph = flipw = false;
      if (flip_dist(rng) < 0.5 && flipdirection != FlipDirection::vertical)
        flipw = true;

      if (flip_dist(rng) < 0.5 && flipdirection != FlipDirection::horizontal)
        fliph = true;

      if (!flipw && !fliph)
        continue;

      if (flipw) {
        for (unsigned int c = 0; c < input_dim[idx].channel(); c++)
          for (unsigned int h = 0; h < input_dim[idx].height(); h++)
            for (unsigned int w = 0; w < input_dim[idx].width() / 2; w++)
              swap(*input_.getAddress(b, c, h, w),
                   *input_.getAddress(b, c, h, width - w - 1));
      }
      if (fliph) {
        for (unsigned int c = 0; c < input_dim[idx].channel(); c++)
          for (unsigned int h = 0; h < input_dim[idx].height() / 2; h++)
            for (unsigned int w = 0; w < input_dim[idx].width(); w++)
              swap(*input_.getAddress(b, c, h, w),
                   *input_.getAddress(b, c, height - h - 1, w));
      }
    }
    hidden_ = input_;
  }
}

void PreprocessFlipLayer::calcDerivative() {
  throw exception::not_supported(
    "calcDerivative for preprocess layer is not supported");
}

void PreprocessFlipLayer::setTrainable(bool train) {
  if (train)
    throw exception::not_supported(
      "Preprocessing layer does not support training");

  Layer::setTrainable(false);
}

} /* namespace nntrainer */
