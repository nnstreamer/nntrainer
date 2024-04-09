// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   reorganization.cpp
 * @date   06 April 2023
 * @todo support in-place operation. we can get channel, height, width
 * coordinate from index of buffer memory. then we can use reorganizePos and
 * restorePos func
 * @brief  This file contains the mean absolute error loss as a sample layer
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <iostream>

#include "reorg_layer.h"

namespace custom {

static constexpr size_t SINGLE_INOUT_IDX = 0;

namespace ReorgOp {

/**
 * @brief re-organize tensor
 * @return output coordinate of reorganized tensor
 */
int reorg(int b, int c, int h, int w, int batch, int channel, int height,
          int width) {
  int out_c = channel / 4;
  int c2 = c % out_c;
  int offset = c / out_c;
  int w2 = w * 2 + offset % 2;
  int h2 = h * 2 + offset / 2;
  int out_index = w2 + width * 2 * (h2 + height * 2 * (c2 + out_c * b));
  return out_index;
}
} // namespace ReorgOp

void ReorgLayer::finalize(nntrainer::InitLayerContext &context) {
  std::vector<nntrainer::TensorDim> dim = context.getInputDimensions();

  for (unsigned int i = 0; i < dim.size(); ++i) {
    if (dim[i].getDataLen() == 0) {
      throw std::invalid_argument("Input dimension is not set");
    } else {
      dim[i].channel(dim[i].channel() * 4);
      dim[i].height(dim[i].height() / 2);
      dim[i].width(dim[i].width() / 2);
    }
  }

  context.setOutputDimensions(dim);
}

void ReorgLayer::forwarding(nntrainer::RunLayerContext &context,
                            bool training) {
  nntrainer::Tensor &in = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &out = context.getOutput(SINGLE_INOUT_IDX);

  for (int b = 0; b < (int)in.batch(); b++) {
    for (int c = 0; c < (int)in.channel(); c++) {
      for (int h = 0; h < (int)in.height(); h++) {
        for (int w = 0; w < (int)in.width(); w++) {
          int out_idx =
            w + in.width() * (h + in.height() * (c + in.channel() * b));
          int in_idx = ReorgOp::reorg(b, c, h, w, in.batch(), in.channel(),
                                      in.height(), in.width());
          out.getData()[out_idx] = in.getValue(in_idx);
        }
      }
    }
  }
}

void ReorgLayer::calcDerivative(nntrainer::RunLayerContext &context) {
  const nntrainer::Tensor &derivative_ =
    context.getIncomingDerivative(SINGLE_INOUT_IDX);

  nntrainer::Tensor &dx = context.getOutgoingDerivative(SINGLE_INOUT_IDX);

  for (int b = 0; b < (int)derivative_.batch(); b++) {
    for (int c = 0; c < (int)derivative_.channel(); c++) {
      for (int h = 0; h < (int)derivative_.height(); h++) {
        for (int w = 0; w < (int)derivative_.width(); w++) {
          int in_idx =
            w + derivative_.width() *
                  (h + derivative_.height() * (c + derivative_.channel() * b));
          int out_idx = ReorgOp::reorg(
            b, c, h, w, derivative_.batch(), derivative_.channel(),
            derivative_.height(), derivative_.width());
          dx.getData()[out_idx] = derivative_.getValue(in_idx);
        }
      }
    }
  }
}

#ifdef PLUGGABLE

nntrainer::Layer *create_reorg_layer() {
  auto layer = new ReorgLayer();
  std::cout << "reorg created\n";
  return layer;
}

void destroy_reorg_layer(nntrainer::Layer *layer) {
  std::cout << "reorg deleted\n";
  delete layer;
}

extern "C" {
nntrainer::LayerPluggable ml_train_layer_pluggable{create_reorg_layer,
                                                   destroy_reorg_layer};
}

#endif

} // namespace custom
