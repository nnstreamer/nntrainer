// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Sachin Singh <sachin.3@samsung.com>
 *
 * @file   unsqueeze_layer.cpp
 * @date   08 July 2025
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Sachin Singh <sachin.3@samsung.com>
 * @author Abhinav Dwivedi <abhinav.d@samsung.com>
 * @bug	   No known bugs except for NYI items
 * @brief  This is Unsqueeze Layer Class for Neural Network
 *
 */

#include <layer_context.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <unsqueeze_layer.h>
namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

void UnsqueezeLayer::finalize(InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "Unsqueeze only supports 1 input";

  auto &axis = std::get<props::Axis>(unsqueeze_props);
  unsigned int input_axis = axis.get();

  TensorDim out_dim = context.getInputDimensions()[0];

  for (unsigned int idx = 0; idx < input_axis; idx++) {
    out_dim[idx] = out_dim[idx + 1];
  }
  out_dim[input_axis] =
    1; // Setting output tensor dimension according to the axis.

  out_dim.setDataType(context.getActivationDataType());
  context.setOutputDimensions({out_dim});
}

void UnsqueezeLayer::forwarding(RunLayerContext &context, bool training) {

  if (!context.getInPlace()) {
    context.getOutput(SINGLE_INOUT_IDX)
      .copyData(context.getInput(SINGLE_INOUT_IDX));
  }
}

void UnsqueezeLayer::calcDerivative(RunLayerContext &context) {

  if (!context.getInPlace()) {
    context.getOutgoingDerivative(SINGLE_INOUT_IDX)
      .copyData(context.getIncomingDerivative(SINGLE_INOUT_IDX));
  }
}

void UnsqueezeLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, unsqueeze_props);
  if (!remain_props.empty()) {
    std::string msg = "[UnsqueezeLayer] Unknown Layer Properties count " +
                      std::to_string(remain_props.size());
    throw exception::not_supported(msg);
  }
}

void UnsqueezeLayer::exportTo(Exporter &exporter,
                              const ml::train::ExportMethods &method) const {
  exporter.saveResult(unsqueeze_props, method, this);
}

} /* namespace nntrainer */
