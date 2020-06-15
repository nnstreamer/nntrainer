/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * SPDX-License-Identifier: Apache-2.0-only
 *
 * @file	pooling2d_layer.cpp
 * @date	12 June 2020
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 * @brief	This is 2 Dimensional Pooling Layer Class for Neural Network
 *
 */

#include <layer.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <parse_util.h>
#include <pooling2d_layer.h>
#include <util_func.h>

namespace nntrainer {

int Pooling2DLayer::initialize(bool last) {
  int status = ML_ERROR_NONE;
  if (input_dim.getDataLen() == 1) {
    ml_logw("Warning: the length of previous layer dimension is one");
  }
  if (input_dim.batch() <= 0 || input_dim.height() <= 0 ||
      input_dim.width() <= 0 || input_dim.channel() <= 0) {
    ml_loge("Error: Dimension must be greater than 0");
    return ML_ERROR_INVALID_PARAMETER;
  }

  this->last_layer = last;
  output_dim.batch(input_dim.batch());
  output_dim.channel(input_dim.channel());
  output_dim.height(
    (input_dim.height() - pooling_size[0] + 2 * padding[0]) / stride[0] + 1);
  output_dim.width(
    (input_dim.width() - pooling_size[1] + 2 * padding[1]) / stride[1] + 1);

  hidden = Tensor(output_dim);

  return status;
}

Tensor Pooling2DLayer::forwarding(Tensor in, int &status) {
  // NYI
  return in;
}

Tensor Pooling2DLayer::forwarding(Tensor in, Tensor output, int &status) {
  return forwarding(in, status);
}

Tensor Pooling2DLayer::backwarding(Tensor in, int iteration) {
  // NYI
  return in;
}

void Pooling2DLayer::copy(std::shared_ptr<Layer> l) {
  // NYI
}

int Pooling2DLayer::setSize(int *size,
                            nntrainer::Pooling2DLayer::PropertyType type) {
  int status = ML_ERROR_NONE;
  // NYI
  return status;
}

int Pooling2DLayer::setProperty(std::vector<std::string> values) {
  int status = ML_ERROR_NONE;

  for (unsigned int i = 0; i < values.size(); ++i) {
    std::string key;
    std::string value;

    status = getKeyValue(values[i], key, value);
    NN_RETURN_STATUS();
    unsigned int t = parseLayerProperty(key);
    switch (static_cast<PropertyType>(t)) {
    case PropertyType::pooling:
      pooling_type = (PoolingType)parseType(value, TOKEN_POOLING);
      if (pooling_type == PoolingType::unknown) {
        ml_loge("Error: Unknown pooling type");
        return ML_ERROR_INVALID_PARAMETER;
      }
      break;
    case PropertyType::pooling_size:
      status = getValues(POOLING2D_DIM, value, (int *)(pooling_size));
      NN_RETURN_STATUS();      
      if (pooling_size[0] == 0 || pooling_size[1] == 0) {
        ml_loge("Error: pooling_size must be greater than 0");
        return ML_ERROR_INVALID_PARAMETER;
      }
      break;
    case PropertyType::stride:
      status = getValues(POOLING2D_DIM, value, (int *)(stride));
      NN_RETURN_STATUS();      
      if (stride[0] == 0 || stride[1] == 0) {
        ml_loge("Error: stride must be greater than 0");
        return ML_ERROR_INVALID_PARAMETER;
      }
      break;
    case PropertyType::padding:
      status = getValues(POOLING2D_DIM, value, (int *)(padding));
      NN_RETURN_STATUS();      
      if (padding[0] < 0 || padding[1] < 0) {
        ml_loge("Error: padding must be greater than 0");
        return ML_ERROR_INVALID_PARAMETER;
      }
      break;
    default:
      ml_loge("Error: Unknown Layer Property Key : %s", key.c_str());
      status = ML_ERROR_INVALID_PARAMETER;
      break;
    }
  }
  return status;
}

Tensor Pooling2DLayer::zero_pad(int batch, Tensor in,
                                unsigned int const *padding) { // NYI
  return in;
}

Tensor Pooling2DLayer::pooling2d(Tensor in, int &status) {
  // NYI
  return in;
}

} /* namespace nntrainer */
