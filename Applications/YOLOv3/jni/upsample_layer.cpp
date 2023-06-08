// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   upsample_layer.h
 * @date   8 June 2023
 * @brief  It is a implementation of upsample layer for 2x upsample.
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <iostream>

#include <upsample_layer.h>

namespace custom {

static constexpr size_t SINGLE_INOUT_IDX = 0;

void UpsampleLayer::finalize(nntrainer::InitLayerContext &context) {
  std::vector<nntrainer::TensorDim> dim = context.getInputDimensions();

  for (unsigned int i = 0; i < dim.size(); ++i) {
    if (dim[i].getDataLen() == 0) {
      throw std::invalid_argument("Input dimension is not set");
    } else {
      dim[i].channel(dim[i].channel());
      dim[i].height(dim[i].height() * 2);
      dim[i].width(dim[i].width() * 2);
    }
  }

  context.setOutputDimensions(dim);
}

void UpsampleLayer::forwarding(nntrainer::RunLayerContext &context,
                               bool training) {
  nntrainer::Tensor &in = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &out = context.getOutput(SINGLE_INOUT_IDX);

  for (int b = 0; b < (int)out.batch(); b++) {
    for (int c = 0; c < (int)out.channel(); c++) {
      for (int h = 0; h < (int)out.height(); h++) {
        for (int w = 0; w < (int)out.width(); w++) {
          out.setValue(b, c, h, w, in.getValue(b, c, h / 2, w / 2));
        }
      }
    }
  }
}

void UpsampleLayer::calcDerivative(nntrainer::RunLayerContext &context) {
  const nntrainer::Tensor &derivative_ =
    context.getIncomingDerivative(SINGLE_INOUT_IDX);

  nntrainer::Tensor &dx = context.getOutgoingDerivative(SINGLE_INOUT_IDX);

  float val = 0;
  for (int b = 0; b < (int)derivative_.batch(); b++) {
    for (int c = 0; c < (int)derivative_.channel(); c++) {
      for (int h = 0; h < (int)derivative_.height(); h++) {
        for (int w = 0; w < (int)derivative_.width(); w++) {
          if (h % 2 == 0 && w % 2 == 0)
            dx.setValue(b, c, h / 2, w / 2, 0);

          val =
            dx.getValue(b, c, h / 2, w / 2) + derivative_.getValue(b, c, h, w);
          dx.setValue(b, c, h / 2, w / 2, val);
        }
      }
    }
  }
}

#ifdef PLUGGABLE

nntrainer::Layer *create_upsample_layer() {
  auto layer = new UpsampleLayer();
  std::cout << "upsample created\n";
  return layer;
}

void destroy_upsample_layer(nntrainer::Layer *layer) {
  std::cout << "upsample deleted\n";
  delete layer;
}

extern "C" {
nntrainer::LayerPluggable ml_train_layer_pluggable{create_upsample_layer,
                                                   destroy_upsample_layer};
}

#endif

} // namespace custom
