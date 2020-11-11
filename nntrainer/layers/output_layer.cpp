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

const std::string OutputLayer::type = "multi_output";

int OutputLayer::initialize() {
  int status = ML_ERROR_NONE;

  if (num_inputs == 0) {
    ml_loge("Error: number of inputs are not initialized");
    return ML_ERROR_INVALID_PARAMETER;
  }

  output_dim.clear();

  // TODO : get output dimensions and set accordingly.
  //        for now, it's just copy of input dim.

  for (unsigned int idx = 0; idx < num_outputs; ++idx) {
    output_dim.push_back(input_dim[0]);
  }

  return status;
}

sharedConstTensors OutputLayer::forwarding(sharedConstTensors in) {

  sharedConstTensors ret;
  for (unsigned int idx = 0; idx < num_outputs; ++idx) {
    Tensor out = Tensor(output_dim[idx]);
    out = *in[0];
    ret.push_back(MAKE_SHARED_TENSOR(out));
  }

  return ret;
}

sharedConstTensors OutputLayer::backwarding(sharedConstTensors derivative,
                                            int iteration) {

  Tensor ret = Tensor(input_dim[0]);

  for (unsigned int idx = 0; idx < num_outputs; ++idx) {
    ret.add_i(*derivative[idx]);
  }

  return {MAKE_SHARED_TENSOR(ret)};
}

void OutputLayer::setProperty(const PropertyType type,
                              const std::string &value) {
  int status = ML_ERROR_NONE;

  switch (type) {
  case PropertyType::num_outputs: {
    if (!value.empty()) {
      status = setUint(num_outputs, value);
      throw_status(status);
      if (num_outputs < 1)
        throw std::invalid_argument("Minimum number of outputs must be 1");
    }
  } break;
  default:
    Layer::setProperty(type, value);
    break;
  }
}

} /* namespace nntrainer */
