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

#include <layer_context.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <reshape_layer.h>
#include <iostream>
namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

void ReshapeLayer::finalize(InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "Reshape only supports 1 input for now";

  const TensorDim &in_dim = context.getInputDimensions()[0];

  auto &target_shape = std::get<props::TargetShape>(reshape_props);
  NNTR_THROW_IF(target_shape.empty(), std::invalid_argument)
    << "Reshape layer must be provided with target shape";
  TensorDim out_dim = target_shape.get();

  if ((int)out_dim.getDataLen() == -1) {
    out_dim.height(1);
    out_dim.channel(1);
    out_dim.width(in_dim.getFeatureLen());
  } else if (out_dim.getFeatureLen() != in_dim.getFeatureLen()) {
    std::cout << context.getName() << "out dim: " << out_dim.batch() << ":" << out_dim.channel() << ":" << out_dim.height() << ":" << out_dim.width()
     << " in dim: "<< in_dim.batch() << ":" << in_dim.channel() << ":" << in_dim.height() << ":" << in_dim.width() << std::endl;

    throw std::invalid_argument(
      "Target and input size mismatch for reshape layer");
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

void ReshapeLayer::exportTo(Exporter &exporter,
                            const ml::train::ExportMethods &method) const {
  exporter.saveResult(reshape_props, method, this);
}

} /* namespace nntrainer */
