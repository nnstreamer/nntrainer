// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Sachin Singh <sachin.3@samsung.com>
 *
 * @file   topk_layer.cpp
 * @date   28 July 2025
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Sachin Singh <sachin.3@samsung.com>
 * @bug	   No known bugs except for NYI items
 * @brief  This is Topk Layer Class for Neural Network
 *
 */

#include <layer_context.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <topk_layer.h>
namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

void TopkLayer::finalize(InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "Topk only supports 1 input";

  auto &KCount = std::get<props::K>(topk_props);

  NNTR_THROW_IF(KCount.empty(), std::invalid_argument)
    << "k value not set in Topk layer";

  unsigned int k = KCount.get();

  NNTR_THROW_IF(k == 0 || k > context.getInputDimensions()[0].width(),
                std::invalid_argument)
    << "k value is invalid in Topk layer. k is " << k
    << ". It should be in range [1," << context.getInputDimensions()[0].width()
    << "]";

  TensorDim out_dim = context.getInputDimensions()[0];
  TensorDim idx_dim = context.getInputDimensions()[0];

  out_dim.width(k);
  idx_dim.width(k);

  out_dim.setDataType(context.getActivationDataType());
  context.setOutputDimensions({out_dim, idx_dim});
}

void TopkLayer::forwarding(RunLayerContext &context, bool training) {

  unsigned int k = std::get<props::K>(topk_props).get();

  auto [output, indices] = context.getInput(0).topK(k);

  context.getOutput(0).copy(output);
  context.getOutput(1).copy(indices);
}

void TopkLayer::calcDerivative(RunLayerContext &context) {

  auto output = context.getIncomingDerivative(0);
  auto indices = context.getOutput(1);

  for (unsigned int b = 0; b < output.batch(); ++b) {
    for (unsigned int c = 0; c < output.channel(); ++c) {
      for (unsigned int h = 0; h < output.height(); ++h) {
        for (unsigned int w = 0; w < output.width(); ++w) {

          auto u = indices.getValue<uint32_t>(b, c, h, w);
          auto val = output.getValue(b, c, h, w);
          context.getOutgoingDerivative(0).setValue(b, c, h, u, val);
        }
      }
    }
  }
}

void TopkLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, topk_props);
  if (!remain_props.empty()) {
    std::string msg = "[TopkLayer] Unknown Layer Properties count " +
                      std::to_string(remain_props.size());
    throw exception::not_supported(msg);
  }
}

void TopkLayer::exportTo(Exporter &exporter,
                         const ml::train::ExportMethods &method) const {
  exporter.saveResult(topk_props, method, this);
}

} /* namespace nntrainer */
