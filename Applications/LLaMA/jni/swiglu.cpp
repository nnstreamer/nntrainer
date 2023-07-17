// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   swiglu.h
 * @date   14 July 2023
 * @brief  Implementation of SwiGLU activation function
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#include <iostream>

#include "swiglu.h"

namespace custom {

static constexpr size_t SINGLE_INOUT_IDX = 0;

namespace ActivationOp {
/**
 * @brief activation function swish
 * @param x input
 * @retval swish(x)
 */
float swish(float x) { return x / (1 + nntrainer::exp_util(-x)); }
} // namespace ActivationOp

void SwiGLULayer::finalize(nntrainer::InitLayerContext &context) {
  std::vector<nntrainer::TensorDim> dim = context.getInputDimensions();

  for (unsigned int i = 0; i < dim.size(); ++i) {
    if (dim[i].getDataLen() == 0) {
      throw std::invalid_argument("Input dimension is not set");
    } else {
      dim[i].channel(dim[i].channel());
      dim[i].height(dim[i].height());
      dim[i].width(dim[i].width() / 2);
    }
  }

  context.setOutputDimensions(dim);
}

void SwiGLULayer::forwarding(nntrainer::RunLayerContext &context,
                             bool training) {
  nntrainer::Tensor &in = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &out = context.getOutput(SINGLE_INOUT_IDX);

  for (int b = 0; b < (int)in.batch(); b++) {
    for (int c = 0; c < (int)in.channel(); c++) {
      for (int h = 0; h < (int)in.height(); h++) {
        for (int w = 0; w < (int)in.width(); w++) {
          int in_idx = in.batch() * b + in.channel() * c + in.height() * h + w;
          int out_idx =
            in.batch() * b + in.channel() * c + in.height() * h + w / 2;
          out.getData()[out_idx] =
            in.getValue(in_idx) * ActivationOp::swish(in.getValue(in_idx + 1));
        }
      }
    }
  }
}

void SwiGLULayer::calcDerivative(nntrainer::RunLayerContext &context) {
  std::throw_with_nested(std::runtime_error("Training is not supported yet."));
}

#ifdef PLUGGABLE

nntrainer::Layer *create_swiglu_layer() {
  auto layer = new SwiGLULayer();
  std::cout << "swiglu created\n";
  return layer;
}

void destroy_swiglu_layer(nntrainer::Layer *layer) {
  std::cout << "swiglu deleted\n";
  delete layer;
}

extern "C" {
nntrainer::LayerPluggable ml_train_layer_pluggable{create_swiglu_layer,
                                                   destroy_swiglu_layer};
}

#endif

} // namespace custom
