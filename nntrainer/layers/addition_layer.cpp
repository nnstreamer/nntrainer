// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file	addition_layer.cpp
 * @date	30 July 2020
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug		No known bugs except for NYI items
 * @brief	This is Addition Layer Class for Neural Network
 *
 */

#include <addition_layer.h>
#include <layer_internal.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <parse_util.h>
#include <util_func.h>

namespace nntrainer {

int AdditionLayer::initialize() {
  int status = ML_ERROR_NONE;
  if (num_inputs == 0) {
    ml_loge("Error: number of inputs are not initialized");
    return ML_ERROR_INVALID_PARAMETER;
  }

  for (unsigned int idx = 0; idx < num_inputs; ++idx) {
    if (input_dim[idx].getDataLen() == 1) {
      ml_logw("Warning: the length of previous layer dimension is one");
    }
  }

  /** input dimension indicates the dimension for all the inputs to follow */
  output_dim[0] = input_dim[0];

  return status;
}

sharedConstTensors AdditionLayer::forwarding(sharedConstTensors in) {
  hidden = Tensor(input_dim[0]);
  hidden.setZero();

  TensorDim &in_dim = input_dim[0];

  for (unsigned int idx = 0; idx < num_inputs; ++idx) {
    if (in_dim != in[idx]->getDim())
      throw std::runtime_error("Error: addition layer requires same "
                               "shape from all input layers");
    hidden.add_i(*in[idx]);
  }

  return {MAKE_SHARED_TENSOR(hidden)};
}

sharedConstTensors AdditionLayer::backwarding(sharedConstTensors derivative,
                                              int iteration) {

  sharedConstTensors ret;
  for (unsigned int i = 0; i < num_inputs; ++i) {
    sharedTensor t =
      std::shared_ptr<Tensor>(new Tensor(), std::default_delete<Tensor[]>());
    *t = *derivative[0];
    ret.push_back(t);
  }

  return ret;
}

void AdditionLayer::setProperty(const PropertyType type,
                                const std::string &value) {
  int status = ML_ERROR_NONE;

  switch (type) {
  case PropertyType::num_inputs: {
    if (!value.empty()) {
      status = setUint(num_inputs, value);
      throw_status(status);
      if (num_inputs < 1)
        throw std::invalid_argument("Minimum number of inputs must be 1");
    }
  } break;
  default:
    Layer::setProperty(type, value);
    break;
  }
}

} /* namespace nntrainer */
