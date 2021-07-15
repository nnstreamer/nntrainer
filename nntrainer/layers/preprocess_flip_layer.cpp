// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   preprocess_flip_layer.cpp
 * @date   20 January 2020
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is Preprocess Random Flip Layer Class for Neural Network
 *
 */

#include <random>

#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <parse_util.h>
#include <preprocess_flip_layer.h>
#include <util_func.h>

namespace nntrainer {

const std::string PreprocessFlipLayer::flip_horizontal = "horizontal";
const std::string PreprocessFlipLayer::flip_vertical = "vertical";
const std::string PreprocessFlipLayer::flip_horizontal_vertical =
  "horizontal_and_vertical";

void PreprocessFlipLayer::finalize(InitLayerContext &context) {
  context.setOutputDimensions(context.getInputDimensions());

  rng.seed(getSeed());
  flip_dist = std::uniform_real_distribution<float>(0.0, 1.0);
}

void PreprocessFlipLayer::setProperty(const std::vector<std::string> &values) {
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

void PreprocessFlipLayer::setProperty(const std::string &type_str,
                                      const std::string &value) {
  using PropertyType = nntrainer::Layer::PropertyType;
  nntrainer::Layer::PropertyType type =
    static_cast<nntrainer::Layer::PropertyType>(parseLayerProperty(type_str));

  switch (type) {
  case PropertyType::flip_direction: {
    if (istrequal(value, flip_horizontal)) {
      flipdirection = FlipDirection::horizontal;
    } else if (istrequal(value, flip_vertical)) {
      flipdirection = FlipDirection::vertical;
    } else if (istrequal(value, flip_horizontal_vertical)) {
      flipdirection = FlipDirection::horizontal_and_vertical;
    } else {
      throw std::invalid_argument("Argument flip direction is invalid");
    }
  } break;
  default:
    std::string msg =
      "[PreprocessFlipLayer] Unknown Layer Property Key for value " +
      std::string(value);
    throw exception::not_supported(msg);
  }
}

void PreprocessFlipLayer::forwarding(RunLayerContext &context, bool training) {
  if (!training) {
    for (unsigned int idx = 0; idx < context.getNumInputs(); idx++) {
      /** TODO: tell the graph to not include this when not training */
      context.getOutput(idx) = context.getInput(idx);
    }

    return;
  }

  using std::swap;
  bool fliph, flipw;

  for (unsigned int idx = 0; idx < context.getNumInputs(); idx++) {
    Tensor &hidden_ = context.getOutput(idx);
    Tensor &input_ = context.getInput(idx);
    const TensorDim input_dim = input_.getDim();
    unsigned int width = input_dim.width();
    unsigned int height = input_dim.height();

    for (unsigned int b = 0; b < input_dim.batch(); b++) {
      fliph = flipw = false;
      if (flip_dist(rng) < 0.5 && flipdirection != FlipDirection::vertical)
        flipw = true;

      if (flip_dist(rng) < 0.5 && flipdirection != FlipDirection::horizontal)
        fliph = true;

      if (!flipw && !fliph)
        continue;

      if (flipw) {
        for (unsigned int c = 0; c < input_dim.channel(); c++)
          for (unsigned int h = 0; h < input_dim.height(); h++)
            for (unsigned int w = 0; w < input_dim.width() / 2; w++)
              swap(*input_.getAddress(b, c, h, w),
                   *input_.getAddress(b, c, h, width - w - 1));
      }
      if (fliph) {
        for (unsigned int c = 0; c < input_dim.channel(); c++)
          for (unsigned int h = 0; h < input_dim.height() / 2; h++)
            for (unsigned int w = 0; w < input_dim.width(); w++)
              swap(*input_.getAddress(b, c, h, w),
                   *input_.getAddress(b, c, height - h - 1, w));
      }
    }
    /** @todo enable inPlace support for this layer */
    hidden_ = input_;
  }
}

void PreprocessFlipLayer::calcDerivative(RunLayerContext &context) {
  throw exception::not_supported(
    "calcDerivative for preprocess layer is not supported");
}

} /* namespace nntrainer */
