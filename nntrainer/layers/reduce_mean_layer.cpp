// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   reduce_mean_layer.cpp
 * @date   25 Nov 2021
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is Reduce Mean Layer Class for Neural Network
 */

#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <reduce_mean_layer.h>

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

void ReduceMeanLayer::finalize(InitLayerContext &context) {
  if (context.getNumInputs() != 1) {
    throw std::invalid_argument("Reduce mean only supports 1 input for now");
  }

  const TensorDim &in_dim = context.getInputDimensions()[0];
  TensorDim out_dim = in_dim;

  /** if reduce axis is not provided, reduction is performed across all the
   * dimensions */
  auto &reduce_axis = std::get<props::Axis>(reduce_mean_props);
  if (reduce_axis.empty()) {
    out_dim = TensorDim({1, 1, 1, 1});
  }

  out_dim.setTensorDim(reduce_axis.get(), 1);
  context.setOutputDimensions({out_dim});
}

void ReduceMeanLayer::forwarding(RunLayerContext &context, bool training) {
  auto &reduce_axis = std::get<props::Axis>(reduce_mean_props);
  if (reduce_axis.empty()) {
    context.getInput(SINGLE_INOUT_IDX)
      .average(context.getOutput(SINGLE_INOUT_IDX));
  } else {
    context.getInput(SINGLE_INOUT_IDX)
      .average(reduce_axis, context.getOutput(SINGLE_INOUT_IDX));
  }
}

void ReduceMeanLayer::calcDerivative(RunLayerContext &context) {
  auto &deriv = context.getOutgoingDerivative(SINGLE_INOUT_IDX);
  auto &ret_deriv = context.getIncomingDerivative(SINGLE_INOUT_IDX);

  unsigned int div = ret_deriv.size() / deriv.size();
  auto &reduce_axis = std::get<props::Axis>(reduce_mean_props);

  if (reduce_axis.empty()) {
    ret_deriv.setValue(deriv.getValue(0));
  } else {
    /** TODO: optimize this by supporting broadcast in copy */
    ret_deriv.setZero();
    ret_deriv.add_i(deriv);
  }

  ret_deriv.divide_i(div);
}

void ReduceMeanLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, reduce_mean_props);
  if (!remain_props.empty()) {
    std::string msg = "[ReduceMeanLayer] Unknown Layer Properties count " +
                      std::to_string(remain_props.size());
    throw exception::not_supported(msg);
  }
}

void ReduceMeanLayer::exportTo(Exporter &exporter,
                               const ExportMethods &method) const {
  exporter.saveResult(reduce_mean_props, method, this);
}

} /* namespace nntrainer */
