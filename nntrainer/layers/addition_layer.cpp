// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   addition_layer.cpp
 * @date   30 July 2020
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is Addition Layer Class for Neural Network
 *
 */

#include <addition_layer.h>
#include <layer_internal.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <parse_util.h>
#include <util_func.h>

namespace nntrainer {

const std::string AdditionLayer::type = "addition";

int AdditionLayer::initialize(Manager &manager) {
  int status = ML_ERROR_NONE;
  if (getNumInputs() == 0) {
    ml_loge("Error: number of inputs are not initialized");
    return ML_ERROR_INVALID_PARAMETER;
  }

  for (unsigned int idx = 0; idx < getNumInputs(); ++idx) {
    if (input_dim[idx].getDataLen() == 1) {
      ml_logw("Warning: the length of previous layer dimension is one");
    }
  }

  /** input dimension indicates the dimension for all the inputs to follow */
  output_dim[0] = input_dim[0];

  return status;
}

void AdditionLayer::forwarding(bool training) {
  Tensor &hidden_ = net_hidden[0]->getVariableRef();
  TensorDim &in_dim = input_dim[0];

  /** @todo check possibility for in-place of addition layer */
  for (unsigned int idx = 0; idx < getNumInputs(); ++idx) {
    if (in_dim != net_input[idx]->getDim())
      throw std::invalid_argument("Error: addition layer requires same "
                                  "shape from all input layers");
    hidden_.add_i(net_input[idx]->getVariableRef());
  }
}

void AdditionLayer::calcDerivative() {

  for (unsigned int i = 0; i < getNumInputs(); ++i) {
    net_input[i]->getGradientRef() = net_hidden[0]->getGradientRef();
  }
}

void AdditionLayer::setProperty(const PropertyType type,
                                const std::string &value) {
  int status = ML_ERROR_NONE;

  switch (type) {
  case PropertyType::num_inputs: {
    if (!value.empty()) {
      unsigned int num_inputs;
      status = setUint(num_inputs, value);
      throw_status(status);
      setNumInputs(num_inputs);
    }
  } break;
  default:
    Layer::setProperty(type, value);
    break;
  }
}

} /* namespace nntrainer */
