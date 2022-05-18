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

#include <flatten_layer.h>
#include <layer_context.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

void FlattenLayer::finalize(InitLayerContext &context) {
  const TensorDim &in_dim = context.getInputDimensions()[0];

  std::string target_shape =
    "target_shape=1:1:" + std::to_string(in_dim.getFeatureLen());
  ReshapeLayer::setProperty({target_shape});

  /** @note the output dimension is in invalid state till finalize of
   * reshape_layer is finished */
  ReshapeLayer::finalize(context);

  if (in_dim.channel() == 1 && in_dim.height() == 1) {
    ml_logw("Warning: the flatten layer is redundant");
  }
}

void FlattenLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, reshape_props);
  if (!remain_props.empty()) {
    std::string msg = "[FlattenLayer] Unknown Layer Properties count " +
                      std::to_string(values.size());
    throw exception::not_supported(msg);
  }
}

void FlattenLayer::exportTo(Exporter &exporter,
                            const ml::train::ExportMethods &method) const {
  exporter.saveResult(reshape_props, method, this);
}

} /* namespace nntrainer */
