// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   swiglu.cpp
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

static constexpr size_t OUT_IDX = 0;
static constexpr size_t INPUT_IDX_1 = 0;
static constexpr size_t INPUT_IDX_2 = 1;

namespace ActivationOp {
/**
 * @brief activation function swish
 * @param x input
 * @retval swish(x)
 */
float swish(float x) { return x / (1 + nntrainer::exp_util(-x)); }
// namespace ActivationOp
} // namespace ActivationOp

void SwiGLULayer::finalize(nntrainer::InitLayerContext &context) {
  context.setOutputDimensions({context.getInputDimensions()[0]});
}

void SwiGLULayer::forwarding(nntrainer::RunLayerContext &context,
                             bool training) {
  nntrainer::Tensor &in1 = context.getInput(INPUT_IDX_1);
  nntrainer::Tensor &in2 = context.getInput(INPUT_IDX_2);
  nntrainer::Tensor &out = context.getOutput(OUT_IDX);

  if (in1.getDataType() == ml::train::TensorDim::DataType::FP32) {
    for (int b = 0; b < (int)in1.batch(); b++) {
      for (int c = 0; c < (int)in1.channel(); c++) {
        for (int h = 0; h < (int)in1.height(); h++) {
          for (int w = 0; w < (int)in1.width(); w++) {
            out.setValue(b, c, h, w,
                         ActivationOp::swish(in1.getValue(b, c, h, w)) *
                           in2.getValue(b, c, h, w));
          }
        }
      }
    }
  } else if (in1.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    for (int b = 0; b < (int)in1.batch(); b++) {
      for (int c = 0; c < (int)in1.channel(); c++) {
        for (int h = 0; h < (int)in1.height(); h++) {
          for (int w = 0; w < (int)in1.width(); w++) {
            out.setValue(
              b, c, h, w,
              static_cast<float>(
                ActivationOp::swish(
                  static_cast<float>(in1.getValue<_FP16>(b, c, h, w))) *
                static_cast<float>(in2.getValue<_FP16>(b, c, h, w))));
          }
        }
      }
    }
#else
    NNTR_THROW_IF(true, std::invalid_argument) << "enable-fp16 is not set!";
#endif
  }
}

void SwiGLULayer::calcDerivative(nntrainer::RunLayerContext &context) {
  // std::throw_with_nested(std::runtime_error("Training is not supported
  // yet."));
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
