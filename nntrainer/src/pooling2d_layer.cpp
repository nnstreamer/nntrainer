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

  if (pooling_type == PoolingType::max) {
    max_idx.resize(output_dim.getDataLen());
  }

  return status;
}

Tensor Pooling2DLayer::forwarding(Tensor in, int &status) {
  for (unsigned int b = 0; b < in.batch(); ++b) {
    Tensor in_padded = zero_pad(b, in, padding);
    Tensor result = pooling2d(b, in_padded, status);
    memcpy(hidden.getAddress(b * hidden.getDim().getFeatureLen()),
           result.getData(), result.getDim().getDataLen() * sizeof(float));
  }
  return hidden;
}

Tensor Pooling2DLayer::forwarding(Tensor in, Tensor output, int &status) {
  return forwarding(in, status);
}

Tensor Pooling2DLayer::backwarding(Tensor derivative, int iteration) {
  unsigned int batch = input_dim.batch();
  unsigned int channel = input_dim.channel();
  unsigned int height = input_dim.height();
  unsigned int width = input_dim.width();
  unsigned int p_height = pooling_size[0];
  unsigned int p_width = pooling_size[1];
  unsigned int p_size = p_height * p_width;

  unsigned int J, K;
  Tensor result = Tensor(input_dim);
  float *out = result.getData();
  switch (pooling_type) {
  case PoolingType::max: {
    for (unsigned int i = 0; i < derivative.getDim().getDataLen(); ++i) {
      out[max_idx[i]] += derivative.getData()[i];
    }
  } break;
  case PoolingType::average: {
    for (unsigned int b = 0; b < batch; ++b) {
      for (unsigned int i = 0; i < channel; ++i) {
        J = 0;
        for (unsigned int j = 0; j <= height - p_height; j += stride[0]) {
          K = 0;
          for (unsigned int k = 0; k <= width - p_width; k += stride[1]) {
            float del = derivative.getValue(b, i, J, K) / (p_size);
            for (unsigned int pi = 0; pi < p_height; ++pi) {
              for (unsigned int pj = 0; pj < p_width; ++pj) {
                result.setValue(b, i, j + pi, k + pj,
                                result.getValue(b, i, j + pi, k + pj) + del);
              }
            }
            K++;
          }
          J++;
        }
      }
    }
  } break;
  case PoolingType::global_max:
  case PoolingType::global_average:
  default:
    ml_loge("Error: Unknown Pooling Type");
    break;
  }
  return result;
}

void Pooling2DLayer::copy(std::shared_ptr<Layer> l) {
  std::shared_ptr<Pooling2DLayer> from =
    std::static_pointer_cast<Pooling2DLayer>(l);

  this->pooling_type = from->pooling_type;

  for (unsigned int i = 0; i < POOLING2D_DIM; ++i) {
    this->pooling_size[i] = from->pooling_size[i];
    this->stride[i] = from->stride[i];
    this->padding[i] = from->padding[i];
  }

  this->input.copy(from->input);
  this->hidden.copy(from->hidden);
  this->dim = from->dim;
  this->input_dim = from->input_dim;
  this->output_dim = from->output_dim;
  this->last_layer = from->last_layer;
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
      if ((int) padding[0] < 0 || (int) padding[1] < 0) {
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

Tensor Pooling2DLayer::pooling2d(unsigned int batch, Tensor in, int &status) {
  unsigned int channel = in.channel();
  unsigned int height = in.height();
  unsigned int width = in.width();
  unsigned int p_height = pooling_size[0];
  unsigned int p_width = pooling_size[1];
  unsigned int base_idx = batch * output_dim.getFeatureLen();

  Tensor output(output_dim.channel(), output_dim.height(), output_dim.width());

  unsigned int J, K;
  switch (pooling_type) {
  case PoolingType::max: {
    for (unsigned int i = 0; i < channel; ++i) {
      J = 0;
      for (unsigned int j = 0; j <= height - p_height; j += stride[0]) {
        K = 0;
        for (unsigned int k = 0; k <= width - p_width; k += stride[1]) {
          float max = std::numeric_limits<float>::min();
          for (unsigned int pi = 0; pi < p_height; ++pi) {
            for (unsigned int pj = 0; pj < p_width; ++pj) {
              float val = in.getValue(0, i, j + pi, k + pj);
              if (max < val) {
                max_idx[base_idx +
                        i * output_dim.height() * output_dim.width() +
                        J * output_dim.width() + K] =
                  batch * input_dim.getFeatureLen() + i * height * width +
                  (j + pi) * width + (k + pj);
                max = val;
              }
            }
          }
          output.setValue(0, i, J, K, max);
          K++;
        }
        J++;
      }
    }
  } break;
  case PoolingType::average: {
    unsigned int p_size = p_height*p_width;
    for (unsigned int i = 0; i < channel; ++i) {
      J = 0;
      for (unsigned int j = 0; j <= height - p_height; j += stride[0]) {
        K = 0;
        for (unsigned int k = 0; k <= width - p_width; k += stride[1]) {
          float sum = 0.0;
          for (unsigned int pi = 0; pi < p_height; ++pi) {
            for (unsigned int pj = 0; pj < p_width; ++pj) {
              sum += in.getValue(0, i, j + pi, k + pj);
            }
          }
          sum = sum / (p_size);
          output.setValue(0, i, J, K, sum);
          K++;
        }
        J++;
      }
    }
  } break;
  case PoolingType::global_max: {
  } break;
  case PoolingType::global_average: {
  } break;
  default:
    ml_loge("Error: Unknown Pooling Type");
    status = ML_ERROR_INVALID_PARAMETER;
    break;
  }

  return output;
}

} /* namespace nntrainer */
