// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   transpose_layer.cpp
 * @date   21 August 2023
 * @brief  Implementation of transpose layer
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#include <cmath>
#include <complex>
#include <iostream>
#include <vector>

#include "transpose_layer.h"

namespace custom {

static constexpr size_t SINGLE_INOUT_IDX = 0;

void TransposeLayer::finalize(nntrainer::InitLayerContext &context) {
  std::vector<nntrainer::TensorDim> dim = context.getInputDimensions();

  for (unsigned int i = 0; i < dim.size(); ++i) {
    if (dim[i].getDataLen() == 0) {
      throw std::invalid_argument("Input dimension is not set");
    } else {
      dim[i].channel(dim[i].channel());
      dim[i].height(dim[i].height());
      dim[i].width(dim[i].width());
    }
  }

  context.setOutputDimensions(dim);
}

void TransposeLayer::forwarding(nntrainer::RunLayerContext &context,
                                bool training) {
  nntrainer::Tensor &in = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &out = context.getOutput(SINGLE_INOUT_IDX);

  in.transpose("1:0:2", out);
}

void TransposeLayer::calcDerivative(nntrainer::RunLayerContext &context) {
  // std::throw_with_nested(std::runtime_error("Training is not supported
  // yet."));
}

#ifdef PLUGGABLE

nntrainer::Layer *create_transpose_layer() {
  auto layer = new TransposeLayer();
  std::cout << "transpose layer created\n";
  return layer;
}

void destroy_transpose_layer(nntrainer::Layer *layer) {
  std::cout << "transpose layer deleted\n";
  delete layer;
}

extern "C" {
nntrainer::LayerPluggable ml_train_layer_pluggable{create_transpose_layer,
                                                   destroy_transpose_layer};
}

#endif

} // namespace custom
