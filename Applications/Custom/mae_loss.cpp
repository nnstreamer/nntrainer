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

namespace custom {
const std::string MaeLossLayer::type = "mae_loss";

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
nntrainer::LayerPluggable ml_train_layer_pluggable{create_mae_loss_layer,
                                                   destory_mae_loss_layer};
}

#endif
} // namespace custom
