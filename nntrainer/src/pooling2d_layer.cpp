// SPDX-License-Identifier: Apache-2.0-only
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
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
#include <lazy_tensor.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <parse_util.h>
#include <pooling2d_layer.h>
#include <util_func.h>

namespace nntrainer {

int Pooling2DLayer::initialize() {
  int status = ML_ERROR_NONE;
  if (input_dim.getDataLen() == 1) {
    ml_logw("Warning: the length of previous layer dimension is one");
  }

  output_dim.batch(input_dim.batch());
  output_dim.channel(input_dim.channel());

  if (pooling_type == PoolingType::max ||
      pooling_type == PoolingType::average) {
    output_dim.height(
      (input_dim.height() - pool_size[0] + 2 * padding[0]) / stride[0] + 1);
    output_dim.width(
      (input_dim.width() - pool_size[1] + 2 * padding[1]) / stride[1] + 1);
  } else {
    output_dim.height(1);
    output_dim.width(1);
  }

  if (pooling_type == PoolingType::max ||
      pooling_type == PoolingType::global_max) {
    max_idx.resize(output_dim.getDataLen());
  }

  return status;
}

sharedConstTensor Pooling2DLayer::forwarding(sharedConstTensor in) {
  input = *in;

  TensorDim hidden_dim = output_dim;
  hidden_dim.batch(in->batch());
  hidden = Tensor(hidden_dim);
  hidden.setZero();

  for (unsigned int b = 0; b < input.batch(); ++b) {
    Tensor in_padded = zero_pad(b, input, padding);
    Tensor result = pooling2d(b, in_padded);
    memcpy(hidden.getAddress(b * hidden.getDim().getFeatureLen()),
           result.getData(), result.getDim().getDataLen() * sizeof(float));
  }

  return MAKE_SHARED_TENSOR(hidden);
}

sharedConstTensor Pooling2DLayer::backwarding(sharedConstTensor derivative,
                                              int iteration) {
  unsigned int batch = input_dim.batch();
  unsigned int channel = input_dim.channel();
  unsigned int height = input_dim.height();
  unsigned int width = input_dim.width();
  unsigned int p_height = pool_size[0];
  unsigned int p_width = pool_size[1];
  unsigned int p_size = p_height * p_width;

  unsigned int J, K;
  Tensor result = Tensor(input_dim);
  result.setZero();
  float *out = result.getData();
  switch (pooling_type) {
  case PoolingType::max: {
    for (unsigned int i = 0; i < derivative->getDim().getDataLen(); ++i) {
      out[max_idx[i]] += derivative->getData()[i];
    }
  } break;
  case PoolingType::average: {
    for (unsigned int b = 0; b < batch; ++b) {
      for (unsigned int i = 0; i < channel; ++i) {
        J = 0;
        for (unsigned int j = 0; j <= height - p_height; j += stride[0]) {
          K = 0;
          for (unsigned int k = 0; k <= width - p_width; k += stride[1]) {
            float del =
              derivative->getValue(b, i, J, K) / static_cast<float>(p_size);
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
  case PoolingType::global_max: {
    for (unsigned int i = 0; i < derivative->getDim().getDataLen(); ++i) {
      out[max_idx[i]] += derivative->getData()[i];
    }
  } break;
  case PoolingType::global_average: {
    unsigned int p_size = width * height;
    for (unsigned int b = 0; b < batch; ++b) {
      for (unsigned int i = 0; i < channel; ++i) {
        float del = derivative->getValue(b, i, 0, 0) / (p_size);
        for (unsigned int j = 0; j < height; ++j) {
          for (unsigned int k = 0; k < width; ++k) {
            result.setValue(b, i, j, k, del);
          }
        }
      }
    }

  } break;
  default:
    throw std::runtime_error("Error: Unknown Pooling Type");
  }
  return MAKE_SHARED_TENSOR(std::move(result));
}

int Pooling2DLayer::setSize(int *size, PropertyType type) {
  int status = ML_ERROR_NONE;
  switch (type) {
  case PropertyType::pool_size:
    for (int i = 0; i < POOLING2D_DIM; ++i) {
      pool_size[i] = size[i];
    }
    break;
  case PropertyType::stride:
    for (int i = 0; i < POOLING2D_DIM; ++i) {
      stride[i] = size[i];
    }
    break;
  case PropertyType::padding:
    for (int i = 0; i < POOLING2D_DIM; ++i) {
      padding[i] = size[i];
    }
    break;
  default:
    ml_loge("Error: Unknown Layer Property type");
    status = ML_ERROR_INVALID_PARAMETER;
    break;
  }
  return status;
}

void Pooling2DLayer::copy(std::shared_ptr<Layer> l) {
  std::shared_ptr<Pooling2DLayer> from =
    std::static_pointer_cast<Pooling2DLayer>(l);

  this->pooling_type = from->pooling_type;

  for (unsigned int i = 0; i < POOLING2D_DIM; ++i) {
    this->pool_size[i] = from->pool_size[i];
    this->stride[i] = from->stride[i];
    this->padding[i] = from->padding[i];
  }

  this->input.copy(from->input);
  this->hidden.copy(from->hidden);
  this->input_dim = from->input_dim;
  this->output_dim = from->output_dim;
}

void Pooling2DLayer::setProperty(const PropertyType type,
                                 const std::string &value) {
  int status = ML_ERROR_NONE;

  switch (type) {
  case PropertyType::pooling:
    if (!value.empty()) {
      pooling_type = (PoolingType)parseType(value, TOKEN_POOLING);
      if (pooling_type == PoolingType::unknown) {
        throw std::invalid_argument("[Pooling2d_layer]: Unknown pooling type");
      }
      break;
    }
  case PropertyType::pool_size:
    if (!value.empty()) {
      status = getValues(POOLING2D_DIM, value, (int *)(pool_size));
      throw_status(status);
      if (pool_size[0] == 0 || pool_size[1] == 0) {
        throw std::invalid_argument(
          "[Pooling2d_layer] pool_size must be greater than 0");
      }
    }
    break;
  case PropertyType::stride:
    if (!value.empty()) {
      status = getValues(POOLING2D_DIM, value, (int *)(stride));
      throw_status(status);
      if (stride[0] == 0 || stride[1] == 0) {
        throw std::invalid_argument(
          "[Pooling2d_layer] stride must be greater than 0");
      }
    }
    break;
  case PropertyType::padding:
    if (!value.empty()) {
      status = getValues(POOLING2D_DIM, value, (int *)(padding));
      throw_status(status);
      if ((int)padding[0] < 0 || (int)padding[1] < 0) {
        throw std::invalid_argument(
          "[Pooling2d_layer] padding must be greater than 0");
      }
    }
    break;
  default:
    Layer::setProperty(type, value);
    break;
  }
}

Tensor Pooling2DLayer::pooling2d(unsigned int batch, Tensor &in) {
  unsigned int channel = in.channel();
  unsigned int height = in.height();
  unsigned int width = in.width();
  unsigned int p_height = pool_size[0];
  unsigned int p_width = pool_size[1];
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
    unsigned int p_size = p_height * p_width;
    for (unsigned int i = 0; i < channel; ++i) {
      J = 0;
      for (unsigned int j = 0; j <= height - p_height; j += stride[0]) {
        K = 0;
        for (unsigned int k = 0; k <= width - p_width; k += stride[1]) {
          float sum = 0.0f;
          for (unsigned int pi = 0; pi < p_height; ++pi) {
            for (unsigned int pj = 0; pj < p_width; ++pj) {
              sum += in.getValue(0, i, j + pi, k + pj);
            }
          }
          sum = sum / static_cast<float>(p_size);
          output.setValue(0, i, J, K, sum);
          K++;
        }
        J++;
      }
    }
  } break;
  case PoolingType::global_max: {
    output.setZero();
    for (unsigned int i = 0; i < channel; ++i) {
      unsigned int idx = batch * input_dim.getFeatureLen() + i * height * width;
      float max = std::numeric_limits<float>::min();
      for (unsigned int j = 0; j < height; ++j) {
        for (unsigned int k = 0; k < width; ++k) {
          float val = in.getValue(0, i, j, k);
          if (max < val) {
            max_idx[base_idx + i] = idx + j * width + k;
            max = val;
          }
        }
      }
      output.setValue(0, i, 0, 0, max);
    }
  } break;
  case PoolingType::global_average: {
    output.setZero();
    Tensor sum_wh = in.chain().sum(3).sum(2).run();
    for (unsigned int i = 0; i < channel; ++i) {
      output.setValue(0, i, 0, 0,
                      sum_wh.getValue(0, i, 0, 0) / (in.width() * in.height()));
    }
  } break;
  default:
    ml_loge("Error: Unknown Pooling Type");
    throw std::runtime_error("Error: Unknown Pooling Type");
    break;
  }

  return output;
}

} /* namespace nntrainer */
