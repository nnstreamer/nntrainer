// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 SeungBaek Hong <sb92.hong@samsung.com>
 *
 * @file   slice_layer.cpp
 * @date   02 April 2025
 * @see    https://github.com/nnstreamer/nntrainer
 * @author SeungBaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is slice layer class (operation layer)
 */

#include "common_properties.h"
#include "tensor_base.h"
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <slice_layer.h>
#include <stdexcept>
#include <util_func.h>

#include <layer_context.h>

namespace nntrainer {

void SliceLayer::finalize(InitLayerContext &context) {
  starts = std::get<props::StartDimensions>(slice_props).get();
  TensorDim ends = std::get<props::EndDimensions>(slice_props).get();
  TensorDim outputDim = context.getInputDimensions()[0];

  for (unsigned int i = 0; i < 3; ++i) {
    outputDim[i + 1] = ends[i] - starts[i];
  }

  context.setOutputDimensions({outputDim});
}

void SliceLayer::forwarding_operation(const Tensor &input, Tensor &output) {
  TensorDim outputDim = output.getDim();
  auto startsDim = starts.getDim();

  for (unsigned int b = 0; b < output.batch(); ++b) {
    for (unsigned int c = 0; c < output.channel(); ++c) {
      for (unsigned int h = 0; h < output.height(); ++h) {
        for (unsigned int w = 0; w < output.width(); ++w) {
          output.setValue(b, c, h, w,
                          input.getValue(b + startsDim[0], c + startsDim[1],
                                         h + startsDim[2], w + startsDim[3]));
        }
      }
    }
  }
}

void SliceLayer::calcDerivative(RunLayerContext &context) {
  const Tensor &inDeriv = context.getIncomingDerivative(SINGLE_INOUT_IDX);
  Tensor &outDeriv = context.getOutgoingDerivative(SINGLE_INOUT_IDX);
  auto startsDim = starts.getDim();

  for (unsigned int b = 0; b < inDeriv.batch(); ++b) {
    for (unsigned int c = 0; c < inDeriv.channel(); ++c) {
      for (unsigned int h = 0; h < inDeriv.height(); ++h) {
        for (unsigned int w = 0; w < inDeriv.width(); ++w) {
          outDeriv.setValue(b + startsDim[0], c + startsDim[1],
                            h + startsDim[2], w + startsDim[3],
                            inDeriv.getValue(b, c, h, w));
        }
      }
    }
  }
}

void SliceLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, slice_props);
  if (!remain_props.empty()) {
    std::string msg = "[SliceLayer] Unknown Layer Properties count " +
                      std::to_string(values.size());
    throw exception::not_supported(msg);
  }
}

} /* namespace nntrainer */
