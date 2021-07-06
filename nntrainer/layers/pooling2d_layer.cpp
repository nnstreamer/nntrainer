// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   pooling2d_layer.cpp
 * @date   12 June 2020
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is 2 Dimensional Pooling Layer Class for Neural Network
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

static constexpr size_t SINGLE_INOUT_IDX = 0;

void Pooling2DLayer::finalize(InitLayerContext &context) {
  if (context.getNumInputs() != 1) {
    throw std::invalid_argument(
      "[Pooling2D] pooling layer only takes one input");
  }

  const TensorDim &in_dim = context.getInputDimensions()[SINGLE_INOUT_IDX];
  TensorDim out_dim;

  if (pooling_type == PoolingType::global_max ||
      pooling_type == PoolingType::global_average) {
    if (pool_size[0] != 0 || pool_size[1] != 0) {
      throw std::invalid_argument(
        "[Pooling2D] global_max, global_average does not accept pool size");
    }
    pool_size[0] = in_dim.height();
    pool_size[1] = in_dim.width();
  }

  padding = std::get<props::Padding2D>(pool2d_props)
              .compute(in_dim, {pool_size[0], pool_size[1]});

  auto [pt, pb, pl, pr] = padding;

  if (pooling_type == PoolingType::global_max ||
      pooling_type == PoolingType::global_average) {
    if (pt + pb + pl + pr != 0) {
      throw std::invalid_argument(
        "[Pooling2D] global_max, global_average does not accept padding");
    }

    if (stride[0] != 1 || stride[1] != 1) {
      throw std::invalid_argument(
        "[Pooling2D] global_max, global_average does not accept stride");
    }
  }

  unsigned int eff_in_height = in_dim.height() + pt + pb;
  unsigned int eff_in_width = in_dim.width() + pl + pr;

  if (eff_in_height < pool_size[0] || eff_in_width < pool_size[1]) {
    throw std::invalid_argument(
      "[Pooling2D] Failed to initialize: in size + padding is smaller "
      "than effective kernel");
  }

  unsigned int IM = std::numeric_limits<int>::max();

  if (eff_in_height - pb - pool_size[0] > IM ||
      eff_in_width - pr - pool_size[1] > IM) {
    throw std::invalid_argument(
      "[Pooling2D] Failed to initialize: Calculated patch end is over int max");
  }

  out_dim.batch(in_dim.batch());
  out_dim.channel(in_dim.channel());
  out_dim.height((eff_in_height - pool_size[0]) / stride[0] + 1);
  out_dim.width((eff_in_width - pool_size[1]) / stride[1] + 1);
  context.setOutputDimensions({out_dim});

  if (pooling_type == PoolingType::global_max) {
    max_idx_global.reserve(out_dim.batch() * out_dim.getFeatureLen());
  } else {
    max_idx.reserve(in_dim.batch() * out_dim.getFeatureLen());
  }
}

void Pooling2DLayer::setBatch(const TensorDim &output_dim, unsigned int batch) {
  if (pooling_type == PoolingType::max) {
    max_idx.reserve(batch * output_dim.getFeatureLen());
  } else if (pooling_type == PoolingType::global_max) {
    max_idx_global.resize(batch * output_dim.getFeatureLen());
  }
}

void Pooling2DLayer::forwarding(RunLayerContext &context, bool training) {
  Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);
  Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);

  const TensorDim &in_dim = input_.getDim();

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
    pooling2d(in_sub, training, result);
  }
}

void Pooling2DLayer::calcDerivative(RunLayerContext &context) {
  Tensor &deriv = context.getIncomingDerivative(SINGLE_INOUT_IDX);
  Tensor &result = context.getOutgoingDerivative(SINGLE_INOUT_IDX);

  const TensorDim &in_dim = result.getDim();
  unsigned int batch = in_dim.batch();
  unsigned int channel = in_dim.channel();
  int height = in_dim.height();
  int width = in_dim.width();

  auto [pt, pb, pl, pr] = padding;
  unsigned int p_height = pool_size[0];
  unsigned int p_width = pool_size[1];

  unsigned int J, K;

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
  case PoolingType::global_average:
  case PoolingType::average: {
    int heigth_stride_end = height - p_height + pb;
    int width_stride_end = width - p_width + pr;
    auto iter = max_idx.begin();
    for (unsigned int b = 0; b < batch; ++b) {
      for (unsigned int i = 0; i < channel; ++i) {
        J = 0;
        for (int j = -pt; j <= heigth_stride_end; j += stride[0]) {
          K = 0;
          for (int k = -pl; k <= width_stride_end; k += stride[1]) {
            float del = deriv.getValue(b, i, J, K) / *iter;
            int patch_height_end =
              std::min(static_cast<int>(j + p_height), height);
            int patch_width_end =
              std::min(static_cast<int>(k + p_width), width);
            int start_h = std::max(0, j);
            int start_w = std::max(0, k);
            for (int h = start_h; h < patch_height_end; ++h) {
              for (int w = start_w; w < patch_width_end; ++w) {
                result.setValue(b, i, h, w, result.getValue(b, i, h, w) + del);
              }
            }
            iter++;
            K++;
          }
          J++;
        }
      }
    }
  } break;
  case PoolingType::global_max: {
    auto iter = max_idx_global.begin();
    float *deriv_data = deriv.getData();
    unsigned int outer_loop = batch * channel;
    for (unsigned int _ = 0; _ < outer_loop; _++) {
      for (unsigned int i = 0; i < out_map_size; ++i) {
        float der = *deriv_data / iter->size();

        for (auto &idx : *iter) {
          result_data[idx] += der;
        }

        iter++;
        deriv_data++;
      }
      result_data += in_map_size;
    }
  } break;
  default:
    throw std::runtime_error("Error: Unknown Pooling Type");
  }
}

void Pooling2DLayer::setProperty(const std::vector<std::string> &values) {
  /// @todo: deprecate this in favor of loadProperties
  for (unsigned int i = 0; i < values.size(); ++i) {
    std::string key;
    std::string value;
    std::stringstream ss;

    if (getKeyValue(values[i], key, value) != ML_ERROR_NONE) {
      throw std::invalid_argument("Error parsing the property: " + values[i]);
    }

    if (value.empty()) {
      ss << "value is empty: key: " << key << ", value: " << value;
      throw std::invalid_argument(ss.str());
    }

    /// @note this calls derived setProperty if available
    setProperty(key, value);
  }
}

void Pooling2DLayer::setProperty(const std::string &type_str,
                                 const std::string &value) {
  using PropertyType = LayerV1::PropertyType;
  int status = ML_ERROR_NONE;
  LayerV1::PropertyType type =
    static_cast<LayerV1::PropertyType>(parseLayerProperty(type_str));

  switch (type) {
  case PropertyType::pooling:
    pooling_type = (PoolingType)parseType(value, TOKEN_POOLING);
    if (pooling_type == PoolingType::unknown) {
      throw std::invalid_argument("[Pooling2d_layer]: Unknown pooling type");
    }
    break;
  case PropertyType::pool_size:
    status = getValues(POOLING2D_DIM, value, (int *)(pool_size.data()));
    throw_status(status);
    if (pool_size[0] == 0 || pool_size[1] == 0) {
      throw std::invalid_argument(
        "[Pooling2d_layer] pool_size must be greater than 0");
    }
    break;
  case PropertyType::stride:
    status = getValues(POOLING2D_DIM, value, (int *)(stride.data()));
    throw_status(status);
    if (stride[0] == 0 || stride[1] == 0) {
      throw std::invalid_argument(
        "[Pooling2d_layer] stride must be greater than 0");
    }
    break;
  case PropertyType::padding:
    from_string(value, std::get<props::Padding2D>(pool2d_props));
    break;
  default:
    std::string msg = "[Pooling2DLayer] Unknown Layer Property Key for value " +
                      std::string(value);
    throw exception::not_supported(msg);
  }
}

void Pooling2DLayer::pooling2d(Tensor &in, bool training, Tensor &output) {

  unsigned int channel = in.channel();
  auto [pt, pb, pl, pr] = padding;

  int in_height = in.height();
  int in_width = in.width();
  unsigned int height = in_height + pt + pb;
  unsigned int width = in_width + pl + pr;
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
      int eff_end_w = std::min(end_w, in_width);
      start_w = std::max(0, start_w);
      for (int h = std::max(0, start_h); h < eff_end_h; ++h) {
        for (int w = start_w; w < eff_end_w; ++w) {
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
  case PoolingType::global_max: {
    pool_fn = [&, this](const float *in_data, int start_h, int start_w) {
      int end_h = start_h + patch_height;
      int end_w = start_w + patch_width;

      float max_val = std::numeric_limits<float>::lowest();

      decltype(max_idx_global)::iterator indexor;
      if (training) {
        max_idx_global.emplace_back();
        indexor = max_idx_global.end() - 1;
      }

      for (int h = start_h; h < end_h; ++h) {
        for (int w = start_w; w < end_w; ++w) {
          int cur_idx = h * in_width + w;
          float val = in_data[cur_idx];
          if (max_val < val) {
            max_val = val;
            if (training) {
              indexor->clear();
              indexor->push_back(cur_idx);
            }
          } else if (max_val == val) {
            if (training) {
              indexor->push_back(cur_idx);
            }
          }
        }
      }

      return max_val;
    };
    break;
  }
  case PoolingType::global_average:
  case PoolingType::average: {
    pool_fn = [&, this](const float *in_data, int start_h, int start_w) {
      int end_h = start_h + patch_height;
      int end_w = start_w + patch_width;
      float total = 0.0f;

      int eff_end_h = std::min(end_h, in_height);
      int eff_end_w = std::min(end_w, in_width);
      int eff_start_h = std::max(0, start_h);
      int eff_start_w = std::max(0, start_w);

      int cnt = (eff_end_h - eff_start_h) * (eff_end_w - eff_start_w);
      for (int h = eff_start_h; h < eff_end_h; ++h) {
        for (int w = eff_start_w; w < eff_end_w; ++w) {
          float val = in_data[h * in_width + w];
          total += val;
        }
      }

      if (training) {
        max_idx.push_back(cnt);
      }
      return total / cnt;
    };
    break;
  }
  case PoolingType::unknown:
  default:
    throw std::invalid_argument("unknown pooling type given");
    break;
  }

  const float *in_data = in.getData();
  float *out_data = output.getData();

  unsigned int map_size = in_height * in_width;

  int heigth_stride_end = height - patch_height - pb;
  int width_stride_end = width - patch_width - pr;
  for (unsigned int i = 0; i < channel; ++i) {
    const float *in_data_channel_sliced = in_data + i * map_size;
    for (int j = -pt; j <= heigth_stride_end; j += stride[0]) {
      for (int k = -pl; k <= width_stride_end; k += stride[1]) {
        float pool_value = pool_fn(in_data_channel_sliced, j, k);
        *out_data = pool_value;
        out_data++;
      }
    }
  }
}

} /* namespace nntrainer */
