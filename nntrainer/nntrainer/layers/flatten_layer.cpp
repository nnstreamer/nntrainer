// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   flatten_layer.cpp
 * @date   16 June 2020
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @author hyeonseok Lee <hs89.lee@samsung.com>
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

  std::string target_shape;

  const unsigned int start_dimension =
    std::get<props::StartDimension>(flatten_props).get();
  const unsigned int end_dimension =
    std::get<props::EndDimension>(flatten_props).get();

  if (in_dim.getFormat() == ml::train::TensorDim::Format::NHWC) {

    NNTR_THROW_IF((start_dimension != 1) &&
                    (end_dimension != ml::train::TensorDim::MAXDIM - 1),
                  std::invalid_argument)
      << "NHWC format does not support start and end dimension property of "
         "flatten layer";

    target_shape =
      "target_shape=" + std::to_string(in_dim.getFeatureLen()) + ":1:1";
  } else {

    NNTR_THROW_IF(start_dimension > end_dimension, std::invalid_argument)
      << "start_dimension is bigger than end_dimension";

    TensorDim target_dim = in_dim;

    unsigned int flattened_size = 1;
    for (unsigned int i = start_dimension; i <= end_dimension; ++i) {
      flattened_size *= in_dim[i];
      target_dim[i] = 1;
    }
    target_dim[end_dimension] = flattened_size;

    target_shape = "target_shape=" + std::to_string(target_dim[1]) + ":" +
                   std::to_string(target_dim[2]) + ":" +
                   std::to_string(target_dim[3]);
  }

  ReshapeLayer::setProperty({target_shape});

  /** @note the output dimension is in invalid state till finalize of
   * reshape_layer is finished */
  ReshapeLayer::finalize(context);

  if (in_dim.channel() == 1 && in_dim.height() == 1) {
    ml_logw("Warning: the flatten layer is redundant");
  }
}

void FlattenLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, flatten_props);
  remain_props = loadProperties(remain_props, reshape_props);
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
