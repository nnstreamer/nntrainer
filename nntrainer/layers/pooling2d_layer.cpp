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
    max_idx.reserve(in_dim.batch() * output_dim[0].getFeatureLen());
  } else if (pooling_type == PoolingType::global_max) {
    max_idx_global.resize(out_dim.batch() * output_dim[0].getFeatureLen());
  }

  return status;
}

void Pooling2DLayer::setBatch(unsigned int batch) {
  Layer::setBatch(batch);
  if (pooling_type == PoolingType::max) {
    max_idx.reserve(batch * output_dim[0].getFeatureLen());
  } else if (pooling_type == PoolingType::global_max) {
    max_idx_global.resize(batch * output_dim[0].getFeatureLen());
  }
}

void Pooling2DLayer::forwarding(bool training) {
  Tensor &input_ = net_input[0]->getVariableRef();
  Tensor &hidden_ = net_hidden[0]->getVariableRef();

  TensorDim &hidden_dim = output_dim[0];
  TensorDim &in_dim = input_dim[0];

  if (hidden_.uninitialized()) {
    hidden_ = Tensor(hidden_dim);
  }

  if (training) {
    if (pooling_type == PoolingType::global_max) {
      max_idx_global.clear();
    } else {
      max_idx.clear();
    }
  }

  for (unsigned int b = 0; b < in_dim.batch(); ++b) {
    Tensor in_sub = input_.getBatchSlice(b, 1);
    Tensor result = hidden_.getBatchSlice(b, 1);
    // zero_pad(b, input_, padding.data(), in_padded);
    pooling2d(in_sub, training, result);
  }
}

void Pooling2DLayer::calcDerivative() {
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
  float *result_data = result.getData();

  unsigned int out_map_size = deriv.height() * deriv.width();
  unsigned int in_map_size = height * width;

  switch (pooling_type) {
  case PoolingType::max: {
    auto iter = max_idx.begin();
    const float *deriv_data = deriv.getData();
    for (unsigned int b = 0; b < batch; ++b) {
      for (unsigned int c = 0; c < channel; ++c) {
        for (unsigned int i = 0; i < out_map_size; ++i) {
          /// max_idx = -1 means the max idx was at the padding, so no need to
          /// update
          if (*iter != -1) {
            result_data[*iter] += *deriv_data;
          }
          iter++;
          deriv_data++;
        }
        result_data += in_map_size;
      }
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
        result_data[max_idx_global[i][m]] += der;
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

Tensor Pooling2DLayer::pooling2d(Tensor &in, bool training, Tensor &output) {

  unsigned int channel = in.channel();
  unsigned int pad_height = padding[0];
  unsigned int pad_width = padding[1];
  int in_height = in.height();
  int in_width = in.width();
  unsigned int height = in_height + pad_height * 2;
  unsigned int width = in_width + pad_width * 2;
  unsigned int patch_height = pool_size[0];
  unsigned int patch_width = pool_size[1];

  NNTR_THROW_IF(output.uninitialized(), std::invalid_argument)
    << "[Pooling2D] output is uninitialized, this is not supported";

  /**
   * @brief pooling function
   * @param in_c channel sliced data
   * @param start_h (height index pointing the start of the patch)
   * @param start_w (width index pointing the start of the patch)
   * @return result value of pooling
   */
  std::function<float(const float *, int, int)> pool_fn;

  switch (pooling_type) {
  case PoolingType::max: {
    pool_fn = [&, this](const float *in_data, int start_h, int start_w) {
      int end_h = start_h + patch_height;
      int end_w = start_w + patch_width;

      float max_val = std::numeric_limits<float>::lowest();

      int cur_max_idx = -1;
      int eff_end_h = std::min(end_h, in_height);
      for (int h = std::max(0, start_h); h < eff_end_h; ++h) {
        int eff_end_w = std::min(end_w, in_width);
        for (int w = std::max(0, start_w); w < eff_end_w; ++w) {
          int cur_idx = h * in_width + w;
          float val = in_data[cur_idx];
          if (max_val < val) {
            max_val = val;
            if (training) {
              cur_max_idx = cur_idx;
            }
          }
        }
      }

      if (training) {
        max_idx.push_back(cur_max_idx);
      }

      return max_val;
    };
    break;
  }
  case PoolingType::average:
  case PoolingType::global_max:
  case PoolingType::global_average:
  case PoolingType::unknown:
  default:
    /// NYI;
    break;
  }

  unsigned int J, K;
  switch (pooling_type) {
  case PoolingType::max: {
    const float *in_data = in.getData();
    float *out_data = output.getData();
    unsigned int map_size = in_height * in_width;
    int heigth_stride_end = height - patch_height - pad_height;
    int width_stride_end = width - patch_width - pad_width;
    for (unsigned int i = 0; i < channel; ++i) {
      const float *in_data_channel_sliced = in_data + i * map_size;
      for (int j = -pad_height; j <= heigth_stride_end; j += stride[0]) {
        for (int k = -pad_width; k <= width_stride_end; k += stride[1]) {
          float pool_value = pool_fn(in_data_channel_sliced, j, k);
          *out_data = pool_value;
          out_data++;
        }
      }
    }
  } break;
  case PoolingType::average: {
    unsigned int p_size = patch_height * patch_width;
    for (unsigned int i = 0; i < channel; ++i) {
      J = 0;
      for (unsigned int j = 0; j <= height - patch_height; j += stride[0]) {
        K = 0;
        for (unsigned int k = 0; k <= width - patch_width; k += stride[1]) {
          float sum = 0.0f;
          for (unsigned int pi = 0; pi < patch_height; ++pi) {
            for (unsigned int pj = 0; pj < patch_width; ++pj) {
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
      unsigned int idx = i * height * width;
      float max = std::numeric_limits<float>::lowest();
      max_idx_global[i].clear();
      for (unsigned int j = 0; j < height; ++j) {
        for (unsigned int k = 0; k < width; ++k) {
          float val = in.getValue(0, i, j, k);
          if (max <= val) {
            if (max < val)
              max_idx_global[i].clear();
            max_idx_global[i].push_back(idx + j * width + k);
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
