// SPDX-License-Identifier: Apache-2.0
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

#include <cstring>
#include <limits>

#include <layer_internal.h>
#include <lazy_tensor.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <parse_util.h>
#include <pooling2d_layer.h>
#include <util_func.h>

namespace nntrainer {

const std::string Pooling2DLayer::type = "pooling2d";

int Pooling2DLayer::initialize(Manager &manager) {
  int status = ML_ERROR_NONE;

  if (input_dim.size() != 1 || output_dim.size() != 1) {
    throw std::invalid_argument("Convolution layer only takes one input");
  }

  TensorDim &in_dim = input_dim[0];
  TensorDim &out_dim = output_dim[0];

  if (in_dim.getDataLen() == 1) {
    ml_logw("Warning: the length of previous layer dimension is one");
  }

  out_dim.batch(in_dim.batch());
  out_dim.channel(in_dim.channel());

  if (pooling_type == PoolingType::max ||
      pooling_type == PoolingType::average) {
    out_dim.height(
      (in_dim.height() - pool_size[0] + 2 * padding[0]) / stride[0] + 1);
    out_dim.width((in_dim.width() - pool_size[1] + 2 * padding[1]) / stride[1] +
                  1);
  } else {
    out_dim.height(1);
    out_dim.channel(1);
    out_dim.width(in_dim.channel());
  }

  if (pooling_type == PoolingType::max) {
    max_idx.resize(out_dim.getDataLen());
  }

  if (pooling_type == PoolingType::global_max) {
    max_idx_global.resize(out_dim.getDataLen());
  }

  return status;
}

void Pooling2DLayer::forwarding(sharedConstTensors in) {
  Tensor &input_ = net_input[0]->getVariableRef();
  Tensor &hidden_ = net_hidden[0]->getVariableRef();

  TensorDim &hidden_dim = output_dim[0];
  TensorDim &in_dim = input_dim[0];

  if (hidden_.uninitialized()) {
    hidden_ = Tensor(hidden_dim);
  }

  for (unsigned int b = 0; b < in_dim.batch(); ++b) {
    Tensor in_padded;
    zero_pad(b, input_, padding.data(), in_padded);
    Tensor result = hidden_.getBatchSlice(b, 1);
    result = pooling2d(b, in_padded, result);
  }
}

void Pooling2DLayer::calcDerivative(sharedConstTensors derivative) {
  unsigned int batch = input_dim[0].batch();
  unsigned int channel = input_dim[0].channel();
  unsigned int height = input_dim[0].height();
  unsigned int width = input_dim[0].width();
  unsigned int p_height = pool_size[0];
  unsigned int p_width = pool_size[1];
  unsigned int p_size = p_height * p_width;

  unsigned int J, K;

  Tensor &deriv = net_hidden[0]->getGradientRef();
  Tensor &result = net_input[0]->getGradientRef();

  result.setZero();
  float *out = result.getData();

  switch (pooling_type) {
  case PoolingType::max: {
    for (unsigned int i = 0; i < deriv.getDim().getDataLen(); ++i) {
      out[max_idx[i]] += deriv.getData()[i];
    }
  } break;
  case PoolingType::average: {
    for (unsigned int b = 0; b < batch; ++b) {
      for (unsigned int i = 0; i < channel; ++i) {
        J = 0;
        for (unsigned int j = 0; j <= height - p_height; j += stride[0]) {
          K = 0;
          for (unsigned int k = 0; k <= width - p_width; k += stride[1]) {
            float del = deriv.getValue(b, i, J, K) / static_cast<float>(p_size);
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
    for (unsigned int i = 0; i < deriv.getDim().getDataLen(); ++i) {
      float der = deriv.getData()[i] / max_idx_global[i].size();
      for (unsigned int m = 0; m < max_idx_global[i].size(); m++) {
        out[max_idx_global[i][m]] += der;
      }
    }
  } break;
  case PoolingType::global_average: {
    unsigned int p_size = width * height;
    for (unsigned int b = 0; b < batch; ++b) {
      for (unsigned int i = 0; i < channel; ++i) {
        float del = deriv.getValue(b, 0, 0, i) / (p_size);
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

void Pooling2DLayer::setBatch(unsigned int batch) {
  Layer::setBatch(batch);

  if (pooling_type == PoolingType::max) {
    max_idx.resize(output_dim[0].getDataLen());
  } else if (pooling_type == PoolingType::global_max) {
    max_idx_global.resize(output_dim[0].getDataLen());
  }
}

void Pooling2DLayer::copy(std::shared_ptr<Layer> l) {
  Layer::copy(l);

  std::shared_ptr<Pooling2DLayer> from =
    std::static_pointer_cast<Pooling2DLayer>(l);

  this->pooling_type = from->pooling_type;

  for (unsigned int i = 0; i < POOLING2D_DIM; ++i) {
    this->pool_size[i] = from->pool_size[i];
    this->stride[i] = from->stride[i];
    this->padding[i] = from->padding[i];
  }
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
      status = getValues(POOLING2D_DIM, value, (int *)(pool_size.data()));
      throw_status(status);
      if (pool_size[0] == 0 || pool_size[1] == 0) {
        throw std::invalid_argument(
          "[Pooling2d_layer] pool_size must be greater than 0");
      }
    }
    break;
  case PropertyType::stride:
    if (!value.empty()) {
      status = getValues(POOLING2D_DIM, value, (int *)(stride.data()));
      throw_status(status);
      if (stride[0] == 0 || stride[1] == 0) {
        throw std::invalid_argument(
          "[Pooling2d_layer] stride must be greater than 0");
      }
    }
    break;
  case PropertyType::padding:
    if (!value.empty()) {
      status = getValues(POOLING2D_DIM, value, (int *)(padding.data()));
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

Tensor Pooling2DLayer::pooling2d(unsigned int batch, Tensor &in,
                                 Tensor &output) {
  unsigned int channel = in.channel();
  unsigned int height = in.height();
  unsigned int width = in.width();
  unsigned int p_height = pool_size[0];
  unsigned int p_width = pool_size[1];
  TensorDim &out_dim = output_dim[0];
  unsigned int base_idx = batch * out_dim.getFeatureLen();

  if (output.uninitialized())
    output = Tensor(1, out_dim.channel(), out_dim.height(), out_dim.width());

  unsigned int J, K;
  switch (pooling_type) {
  case PoolingType::max: {
    for (unsigned int i = 0; i < channel; ++i) {
      J = 0;
      for (unsigned int j = 0; j <= height - p_height; j += stride[0]) {
        K = 0;
        for (unsigned int k = 0; k <= width - p_width; k += stride[1]) {
          float max = std::numeric_limits<float>::lowest();
          for (unsigned int pi = 0; pi < p_height; ++pi) {
            for (unsigned int pj = 0; pj < p_width; ++pj) {
              float val = in.getValue(0, i, j + pi, k + pj);
              if (max < val) {
                max_idx[base_idx + i * out_dim.height() * out_dim.width() +
                        J * out_dim.width() + K] =
                  batch * input_dim[0].getFeatureLen() + i * height * width +
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
    for (unsigned int i = 0; i < channel; ++i) {
      unsigned int idx =
        batch * input_dim[0].getFeatureLen() + i * height * width;
      float max = std::numeric_limits<float>::lowest();
      max_idx_global[base_idx + i].clear();
      for (unsigned int j = 0; j < height; ++j) {
        for (unsigned int k = 0; k < width; ++k) {
          float val = in.getValue(0, i, j, k);
          if (max <= val) {
            if (max < val)
              max_idx_global[base_idx + i].clear();
            max_idx_global[base_idx + i].push_back(idx + j * width + k);
            max = val;
          }
        }
      }
      output.setValue(0, 0, 0, i, max);
    }
  } break;
  case PoolingType::global_average: {
    Tensor sum_wh = in.chain().sum(3).sum(2).run();
    for (unsigned int i = 0; i < channel; ++i) {
      output.setValue(0, 0, 0, i,
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
