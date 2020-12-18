// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file	concat_layer.cpp
 * @date	27 Oct 2020
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 * @brief	This is Concat Layer Class for Neural Network
 *
 */

#include <concat_layer.h>
#include <cstring>
#include <layer_internal.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <parse_util.h>
#include <util_func.h>

namespace nntrainer {

const std::string ConcatLayer::type = "concat";

int ConcatLayer::initialize(Manager &manager) {
  int status = ML_ERROR_NONE;
  unsigned int channel = 0;

  if (num_inputs == 0) {
    ml_loge("Error: number of inputs are not initialized");
    return ML_ERROR_INVALID_PARAMETER;
  }

  const TensorDim &d = input_dim[0];
  channel += d.channel();
  for (unsigned int idx = 1; idx < num_inputs; ++idx) {
    const TensorDim &dim = input_dim[idx];

    for (unsigned int i = 2; i < d.rank(); ++i) {
      if (d[i] != dim[i])
        throw std::runtime_error("Error: concat layer requires same "
                                 "shape from  all input layers");
    }
    channel += input_dim[idx].channel();
  }

  output_dim[0] = input_dim[0];
  output_dim[0].channel(channel);

  return status;
}

void ConcatLayer::forwarding(sharedConstTensors in) {
  Tensor &hidden_ = net_hidden[0]->getVariableRef();

#ifdef DEBUG
  unsigned int channel = 0;
  const TensorDim &d = net_input[0]->getDim();
  channel += d.channel();
  for (unsigned int idx = 1; idx < num_inputs; ++idx) {
    const TensorDim &dim = net_input[idx]->getDim();

    for (unsigned int i = 2; i < d.rank(); ++i) {
      if (d[i] != dim[i])
        throw std::runtime_error("Error: concat layer requires same "
                                 "shape from  all input layers");
    }
    channel += input_dim[idx].channel();
  }

  if (channel != output_dim[0].channel())
    throw std::runtime_error(
      "Error: Sum of channel of input layers is not same with output channel");
#endif

  unsigned int f_size = output_dim[0].getFeatureLen();

  for (unsigned int b = 0; b < input_dim[0].batch(); ++b) {
    unsigned int position = 0;
    for (unsigned int idx = 0; idx < num_inputs; ++idx) {
      TensorDim in_dim = net_input[idx]->getDim();
      memcpy(
        hidden_.getAddress(b * f_size + position),
        net_input[idx]->getVariable().getAddress(b * in_dim.getFeatureLen()),
        in_dim.getFeatureLen() * sizeof(float));
      position += in_dim.getFeatureLen();
    }
  }
}

void ConcatLayer::calcDerivative(sharedConstTensors derivative) {
  TensorDim d = net_hidden[0]->getDim();

  unsigned int position = 0;
  for (unsigned int idx = 0; idx < num_inputs; ++idx) {
    TensorDim in_dim = input_dim[idx];

    for (unsigned int b = 0; b < in_dim.batch(); ++b) {
      memcpy(
        net_input[idx]->getGradient().getAddress(b * in_dim.getFeatureLen()),
        net_hidden[0]->getGradient().getAddress(b * d.getFeatureLen() +
                                                position),
        in_dim.getFeatureLen() * sizeof(float));
    }
    position += in_dim.getFeatureLen();
  }
}

void ConcatLayer::setProperty(const PropertyType type,
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
