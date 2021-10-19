// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   flatten_layer.cpp
 * @date   16 June 2020
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug	   No known bugs except for NYI items
 * @brief  This is Flatten Layer Class for Neural Network
 *
 * @todo Update flatten to work in-place properly.
 */

#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <reshape_layer.h>

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

void ReshapeLayer::finalize(InitLayerContext &context) {
  if (context.getNumInputs() != 1) {
    throw std::invalid_argument("Reshape only supports 1 input for now");
  }

  const TensorDim &in_dim = context.getInputDimensions()[0];

  auto &target_shape = std::get<props::TargetShape>(reshape_props);
  if (target_shape.empty())
    throw std::invalid_argument(
      "Reshape layer must be provided with target shape");
  TensorDim out_dim = target_shape.get();

  /** flatten sets the dimension to 1 to indicate to flatten the rest of the
   * dimensions */
  if ((int)out_dim.getDataLen() == -1) {
    out_dim.height(1);
    out_dim.channel(1);
    out_dim.width(in_dim.getFeatureLen());
  }

  out_dim.batch(in_dim.batch());

  context.setOutputDimensions({out_dim});
}

void ReshapeLayer::forwarding(RunLayerContext &context, bool training) {
  if (!context.executeInPlace()) {
    context.getOutput(SINGLE_INOUT_IDX)
      .copyData(context.getInput(SINGLE_INOUT_IDX));
  }
}

void ReshapeLayer::calcDerivative(RunLayerContext &context) {
  if (!context.executeInPlace()) {
    context.getOutgoingDerivative(SINGLE_INOUT_IDX)
      .copyData(context.getIncomingDerivative(SINGLE_INOUT_IDX));
  }
}

void ReshapeLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, reshape_props);
  if (!remain_props.empty()) {
    std::string msg = "[ReshapeLayer] Unknown Layer Properties count " +
                      std::to_string(remain_props.size());
    throw exception::not_supported(msg);
  }
}
} /* namespace nntrainer */
