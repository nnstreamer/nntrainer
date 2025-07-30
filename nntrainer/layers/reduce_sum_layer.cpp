// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Sumon Nath <sumon.nath@samsung.com>
 *
 * @file   reduce_sum_layer.cpp
 * @date   24 July 2025
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Sumon Nath <sumon.nath@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is Reduce Sum Layer Class for Neural Network
 */

#include <layer_context.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <reduce_sum_layer.h>

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

void ReduceSumLayer::finalize(InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "Reduce sum only supports 1 input for now";

  const TensorDim &in_dim = context.getInputDimensions()[0];
  TensorDim out_dim = in_dim;

  /**
   * if reduce axis is not provided, reduction is performed across all the
   * dimensions except the batch
   */
  auto &reduce_axis = std::get<props::ReduceDimension>(reduce_sum_props);
  out_dim.setTensorDim(reduce_axis.get(), 1);
  context.setOutputDimensions({out_dim});
}

void ReduceSumLayer::forwarding(RunLayerContext &context, bool training) {
  auto &reduce_axis = std::get<props::ReduceDimension>(reduce_sum_props);
  context.getInput(SINGLE_INOUT_IDX)
    .sum(reduce_axis, context.getOutput(SINGLE_INOUT_IDX));
}

void ReduceSumLayer::calcDerivative(RunLayerContext &context) {
  auto &deriv = context.getIncomingDerivative(SINGLE_INOUT_IDX);
  auto &ret_deriv = context.getOutgoingDerivative(SINGLE_INOUT_IDX);

  ret_deriv.setZero();
  ret_deriv.add_i(deriv);
}

void ReduceSumLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, reduce_sum_props);
  if (!remain_props.empty()) {
    std::string msg = "[ReduceSumLayer] Unknown Layer Properties count " +
                      std::to_string(remain_props.size());
    throw exception::not_supported(msg);
  }
}

void ReduceSumLayer::exportTo(Exporter &exporter,
                              const ml::train::ExportMethods &method) const {
  exporter.saveResult(reduce_sum_props, method, this);
}

} /* namespace nntrainer */
