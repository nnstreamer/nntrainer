// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   time_dist.cpp
 * @date   01 April 2021
 * @brief  This is Time Distributed Layer Class of Neural Network
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#include <layer_internal.h>
#include <lazy_tensor.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <parse_util.h>
#include <time_dist.h>
#include <util_func.h>

namespace nntrainer {

const std::string TimeDistLayer::type = "time_dist";

int TimeDistLayer::initialize(Manager &manager) {
  int status = ML_ERROR_NONE;

  if (getNumInputs() != 1) {
    throw std::invalid_argument("Time distributed layer takes only one input");
  }

  if (!dist_layer) {
    throw std::invalid_argument("distributed layer is not set properly");
  }

  if (input_dim[0].channel() != 1) {
    throw std::invalid_argument(
      "only 1 channel is allow for time distributed layer");
  }

  TensorDim dist_dim = input_dim[0];
  dist_dim.height(1);

  dist_layer->setInputDimension({dist_dim});

  // Set the weight of dist_layer
  // Input & Output Buffer is set by manager of model.
  // During forwarding and backwarding, it set the input and output buffer of
  // dist_layer properly
  // dist_layer will use forwarding_with_val and backwarding_with_val
  dist_layer->initialize(manager);

  output_dim[0] = dist_layer->getOutputDimension()[0];

  // input_dim[0].height is number of time iteration
  output_dim[0].height(input_dim[0].height());

  return status;
}

void TimeDistLayer::forwarding(bool training) {
  // NYI
}

void TimeDistLayer::copy(std::shared_ptr<Layer> l) {
  Layer::copy(l);

  std::shared_ptr<TimeDistLayer> from =
    std::static_pointer_cast<TimeDistLayer>(l);
  this->dist_layer = from->dist_layer;
}

void TimeDistLayer::setDistLayer(std::shared_ptr<Layer> l) {
  dist_layer = l;
  Layer::setActivation(l->getActivationType());
};

void TimeDistLayer::calcDerivative() {
  // NYI
}

void TimeDistLayer::calcGradient() {
  // NYI
}

} /* namespace nntrainer */
