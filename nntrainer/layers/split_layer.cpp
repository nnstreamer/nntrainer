// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   split_layer.cpp
 * @date   21 May 2021
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is Split Layer Class for Neural Network
 *
 */

#include <cstring>
#include <layer_internal.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <parse_util.h>
#include <split_layer.h>
#include <util_func.h>

namespace nntrainer {

const std::string SplitLayer::type = "split";

int SplitLayer::initialize(Manager &manager) {
  int status = ML_ERROR_NONE;

  if (split_dimension < 1) {
    ml_loge("Error: cannot split along the batch dimension");
    return ML_ERROR_INVALID_PARAMETER;
  }

  if (split_dimension >= MAXDIM) {
    ml_loge("Error: split dimension exceeding the total number of dimensions");
    return ML_ERROR_INVALID_PARAMETER;
  }

  if (getNumInputs() == 0) {
    ml_loge("Error: number of inputs are not initialized");
    return ML_ERROR_INVALID_PARAMETER;
  }

  if (getNumInputs() > 1) {
    ml_loge("Error: only a single input is supported with split layer");
    return ML_ERROR_INVALID_PARAMETER;
  }

  const TensorDim &in_dim = input_dim[0];
  setNumOutputs(in_dim.getTensorDim(split_dimension));

  TensorDim d = in_dim;
  d.setTensorDim(split_dimension, 1);

  for (unsigned int idx = 0; idx < getNumOutputs(); ++idx) {
    output_dim[idx] = d;
  }

  /**
   * Setup input_reshape_helper to which input will be reshaped in forwarding
   * to facilitate easier processing.
   */
  input_reshape_helper = {1, 1, 1, 1};
  for (unsigned int idx = 0; idx < split_dimension; ++idx) {
    input_reshape_helper.batch(input_reshape_helper.batch() *
                               in_dim.getTensorDim(idx));
  }

  input_reshape_helper.height(in_dim.getTensorDim(split_dimension));

  for (unsigned int idx = split_dimension + 1; idx < MAXDIM; ++idx) {
    input_reshape_helper.width(input_reshape_helper.width() *
                               in_dim.getTensorDim(idx));
  }

  /**
   * Setup output_reshape_helper to which input will be reshaped in forwarding
   * to facilitate easier processing.
   */
  output_reshape_helper = input_reshape_helper;
  output_reshape_helper.height(1);

  return status;
}

void SplitLayer::forwarding(bool training) {
  /**
   * NYI
   */
}

void SplitLayer::calcDerivative() {
  /**
   * NYI
   */
}

void SplitLayer::setProperty(const PropertyType type,
                             const std::string &value) {
  int status = ML_ERROR_NONE;

  switch (type) {
  case PropertyType::split_dimension: {
    if (!split_dimension.empty()) {
      status = setUint(split_dimension, value);
      throw_status(status);
    }
  } break;
  default:
    Layer::setProperty(type, value);
    break;
  }
}

} /* namespace nntrainer */
