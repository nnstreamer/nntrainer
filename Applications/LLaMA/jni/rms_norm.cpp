// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   rms_norm.cpp
 * @date   19 July 2023
 * @brief  Implementation of RMS normalization function
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#include <cmath>
#include <iostream>

#include "rms_norm.h"

namespace custom {

static constexpr size_t SINGLE_INOUT_IDX = 0;

void RMSNormLayer::finalize(nntrainer::InitLayerContext &context) {
  std::vector<nntrainer::TensorDim> dim = context.getInputDimensions();

  context.setOutputDimensions(dim);

  auto &rmsparams_gamma = std::get<props::RMS_NORM_GAMMA_INIT>(rms_props);

  nntrainer::TensorDim gamma_dim(
    1, 1, 1, dim[0].width(),
    nntrainer::TensorDim::TensorType(context.getFormat(),
                                     context.getWeightDataType()));
  wt_idx[RMSParams::gamma] = context.requestWeight(
    gamma_dim, rmsparams_gamma, nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f,
    "gamma", false);
}

void RMSNormLayer::forwarding(nntrainer::RunLayerContext &context,
                              bool training) {
  nntrainer::Tensor &in = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &out = context.getOutput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &gamma = context.getWeight(wt_idx[RMSParams::gamma]);

  auto &epsilon = std::get<nntrainer::props::Epsilon>(rms_props).get();

#ifndef ENABLE_FP16
  std::function<float(float)> f = [](float x) { return 1 / std::sqrt(x); };
  auto t = in.multiply(in).average(3).add(epsilon);
  t.apply_i(f);
#else
  ml::train::TensorDim d = in.getDim();
  d.width(1);
  nntrainer::Tensor t(d, true);
  unsigned int axis_dim = in.getDim()[3];
  for (unsigned int i = 0; i < in.getDim()[2]; ++i) {
    float sum = 0.0;
    _FP16 *data = in.getAddress<_FP16>(0, 0, i, 0);
    for (unsigned int j = 0; j < axis_dim; ++j) {
      sum += powf(static_cast<float>(data[j]), 2.0f);
    }
    t.setValue(0, 0, i, 0, 1.0 / sqrt(sum / axis_dim + epsilon));
  }
#endif

  in.multiply(t, out);
  out.multiply_i(gamma);
}

void RMSNormLayer::calcDerivative(nntrainer::RunLayerContext &context) {
  // std::throw_with_nested(std::runtime_error("Training is not supported
  // yet."));
}

#ifdef PLUGGABLE

nntrainer::Layer *create_rms_norm_layer() {
  auto layer = new RMSNormLayer();
  std::cout << "rms_norm created\n";
  return layer;
}

void destroy_rms_norm_layer(nntrainer::Layer *layer) {
  std::cout << "rms_norm deleted\n";
  delete layer;
}

extern "C" {
nntrainer::LayerPluggable ml_train_layer_pluggable{create_rms_norm_layer,
                                                   destroy_rms_norm_layer};
}

#endif

} // namespace custom
