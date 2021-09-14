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
#include <node_exporter.h>
#include <preprocess_flip_layer.h>
#include <util_func.h>

namespace nntrainer {

PreprocessFlipLayer::PreprocessFlipLayer() :
  Layer(),
  preprocess_flip_props(props::FlipDirection()) {}

void PreprocessFlipLayer::finalize(InitLayerContext &context) {
  context.setOutputDimensions(context.getInputDimensions());

  rng.seed(getSeed());
  flip_dist = std::uniform_real_distribution<float>(0.0, 1.0);
}

void PreprocessFlipLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, preprocess_flip_props);
  NNTR_THROW_IF(!remain_props.empty(), std::invalid_argument)
    << "[PreprocessFilpLayer] Unknown Layer Properties count " +
         std::to_string(values.size());
}

void PreprocessFlipLayer::forwarding(RunLayerContext &context, bool training) {
  props::FlipDirectionInfo::Enum flipdirection =
    std::get<props::FlipDirection>(preprocess_flip_props).get();

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
      if (flip_dist(rng) < 0.5 &&
          flipdirection != props::FlipDirectionInfo::Enum::vertical)
        flipw = true;

      if (flip_dist(rng) < 0.5 &&
          flipdirection != props::FlipDirectionInfo::Enum::horizontal)
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

void PreprocessFlipLayer::exportTo(Exporter &exporter,
                                   const ExportMethods &method) const {
  exporter.saveResult(preprocess_flip_props, method, this);
}

} /* namespace nntrainer */
