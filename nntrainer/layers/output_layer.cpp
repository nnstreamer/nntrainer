// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file        output_layer.cpp
 * @date        05 Nov 2020
 * @see         https://github.com/nnstreamer/nntrainer
 * @author      Jijoong Moon <jijoong.moon@samsung.com>
 * @bug         No known bugs except for NYI items
 * @brief       This is Multi Output Layer Class for Neural Network
 *
 */

#include <cstring>
#include <layer_internal.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <output_layer.h>
#include <parse_util.h>
#include <util_func.h>

namespace nntrainer {

const std::string OutputLayer::type = "output";

int OutputLayer::initialize(Manager &manager) {
  int status = ML_ERROR_NONE;

  if (getNumInputs() == 0) {
    ml_loge("Error: number of inputs are not initialized");
    return ML_ERROR_INVALID_PARAMETER;
  }

  // TODO : get output dimensions and set accordingly.
  //        for now, it's just copy of input dim.

  for (unsigned int idx = 0; idx < getNumOutputs(); ++idx) {
    output_dim[idx] = input_dim[0];
  }

  return status;
}

void OutputLayer::forwarding(bool training) {
  Tensor &input_ = net_input[0]->getVariableRef();
  for (unsigned int idx = 0; idx < getNumOutputs(); ++idx) {
    net_hidden[idx]->getVariableRef() = input_;
  }
}

void OutputLayer::calcDerivative() {

  Tensor &ret = net_input[0]->getGradientRef();

  for (unsigned int idx = 0; idx < getNumOutputs(); ++idx) {
    if (idx == 0) {
      ret.fill(net_hidden[idx]->getGradientRef(), false);
    } else {
      ret.add_i(net_hidden[idx]->getGradientRef());
    }
  }
}

void OutputLayer::setProperty(const PropertyType type,
                              const std::string &value) {
  int status = ML_ERROR_NONE;

  switch (type) {
  case PropertyType::num_outputs: {
    if (!value.empty()) {
      unsigned int num_outputs;
      status = setUint(num_outputs, value);
      throw_status(status);
      setNumOutputs(num_outputs);
    }
  } break;
  default:
    LayerV1::setProperty(type, value);
    break;
  }
}

} /* namespace nntrainer */
