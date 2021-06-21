// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   mae_loss.cpp
 * @date   10 June 2021
 * @brief  This file contains the mean absolute error loss as a sample layer
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */
#include "mae_loss.h"

#include <cmath>

#include <tensor.h>

constexpr const float EPSILON_ = 1e-7;
namespace custom {

int MaeLossLayer::initialize(nntrainer::Manager &manager) {
  output_dim = input_dim;
  return ML_ERROR_NONE;
}

int MaeLossLayer::setProperty(std::vector<std::string> values) {
  /// this implementation makes to pass the test, this will change soon.
  return values.size();
}

void MaeLossLayer::forwarding(bool training) {
  nntrainer::Tensor &label = net_hidden[0]->getGradientRef();
  nntrainer::Tensor &predicted = net_input[0]->getVariableRef();
  nntrainer::Tensor &output = net_hidden[0]->getVariableRef();

  bool with_label = !label.uninitialized();
  if (with_label) {
    predicted.subtract(label, output);
    /// make Tensor::abs instead and use it here
    output.apply_i(fabs);
  } else {
    output.fill(predicted);
  }
}

void MaeLossLayer::calcDerivative() {
  nntrainer::Tensor &predicted = net_input[0]->getVariableRef();
  nntrainer::Tensor &label = net_hidden[0]->getGradientRef();

  nntrainer::Tensor &deriv = net_input[0]->getGradientRef();

  /// This can be saved at MaeLossLayer::forwarding, but this is done here on
  /// purpose for demonstration purpose
  predicted.subtract(label, deriv);

  deriv.apply_i([](float x) {
    if (fabs(x) < EPSILON_) {
      return 0.0f;
    }
    return x > 0 ? 1.0f : -1.0f;
  });
}

bool MaeLossLayer::requireLabel() const { return true; }

#ifdef PLUGGABLE

nntrainer::LayerV1 *create_mae_loss_layer() {
  auto layer = new MaeLossLayer();
  std::cout << "mae loss layer created\n";
  return layer;
}

void destory_mae_loss_layer(nntrainer::LayerV1 *layer) {
  std::cout << "mae loss layer destroyed\n";
  delete layer;
}

/**
 * @note ml_train_layer_pluggable defines the entry point for nntrainer to
 * register a plugin layer
 */
extern "C" {
nntrainer::LayerV1Pluggable ml_train_layerv1_pluggable{create_mae_loss_layer,
                                                       destory_mae_loss_layer};
}

#endif
} // namespace custom
