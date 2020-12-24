// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file	conv2d_layer.h
 * @date	02 June 2020
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @author	Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug		No known bugs except for NYI items
 * @brief	This is Convolution Layer Class for Neural Network
 *
 */
#include <algorithm>
#include <cstring>
#include <string>

#include <blas_interface.h>
#include <conv2d_layer.h>
#include <layer_internal.h>
#include <lazy_tensor.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <parse_util.h>
#include <profiler.h>
#include <util_func.h>

namespace nntrainer {

/// @note this will be deleted after conv2d optimization is done!
#ifdef PROFILE
namespace {

int pad_profile_key;
int conv_gemm_profile_key;
int im2col_key;
int add_bias_key;
int clean_up;
int temp_key;

void register_event() {
  pad_profile_key = profile::Profiler::Global().registerEvent("zero_pad");
  im2col_key = profile::Profiler::Global().registerEvent("im2col");
  conv_gemm_profile_key =
    profile::Profiler::Global().registerEvent("conv_gemm");

  add_bias_key = profile::Profiler::Global().registerEvent("add_bias_key");
  clean_up = profile::Profiler::Global().registerEvent("clean_up");
  temp_key = profile::Profiler::Global().registerEvent("temp_key");
}
} // namespace
#endif
const std::string Conv2DLayer::type = "conv2d";

enum ConvParams { weight, bias };

int Conv2DLayer::initialize(Manager &manager) {
  int status = ML_ERROR_NONE;

  if (input_dim.size() != 1 || output_dim.size() != 1) {
    throw std::invalid_argument("Convolution layer only takes one input");
  }

  TensorDim &in_dim = input_dim[0];
  TensorDim &out_dim = output_dim[0];

  if (in_dim.getDataLen() == 1) {
    ml_logw("Warning: the length of previous layer dimension is one");
  }

  std::string kernelPrefix = "Conv2d:filter";
  std::string biasPrefix = "Conv2d:bias";

  TensorDim dim =
    TensorDim(filter_size, in_dim.channel(), kernel_size[0], kernel_size[1]);
  TensorDim bias_dim = TensorDim(1, filter_size, 1, 1);

  if (weights.empty()) {
    weights.reserve(2);
    weights.emplace_back(dim, weight_initializer, true, kernelPrefix);
    weights.emplace_back(bias_dim, bias_initializer, true, biasPrefix);
    manager.trackWeights(weights);
  } else {
    weights[ConvParams::weight].reset(dim, weight_initializer, true);
    weights[ConvParams::bias].reset(bias_dim, bias_initializer, true);
  }

  // this output_dim should be the same with dimension of hidden
  out_dim.batch(in_dim.batch());
  out_dim.channel(filter_size);
  out_dim.height(
    (in_dim.height() - kernel_size[0] + 2 * padding[0]) / stride[0] + 1);
  out_dim.width((in_dim.width() - kernel_size[1] + 2 * padding[1]) / stride[1] +
                1);

#ifdef PROFILE
  register_event();
#endif

  return status;
}

void Conv2DLayer::forwarding(sharedConstTensors in) {
  int status = ML_ERROR_NONE;

  if (num_inputs != 1)
    throw std::invalid_argument("Convolution layer only takes one input");

  Tensor &input_ = net_input[0]->getVariableRef();

  TensorDim &in_dim = input_dim[0];
  TensorDim &out_dim = output_dim[0];

  Tensor &hidden_ = net_hidden[0]->getVariableRef();

  Tensor &filter_kernel = weightAt(ConvParams::weight).getVariableRef();
  Tensor &bias_kernel = weightAt(ConvParams::bias).getVariableRef();

  /** Calculate Convolution 2D
   *
   * This is the 2D Matrix Shape [ height ] x [ width ]
   *   . Height : filter_size
   *   . Width  : Input Channel * Kernel_size[0] * Kernel_size[1]
   *
   *                              imKernel
   *                        +------|------|------+
   *                        |------|------|------|
   * [filter_size (height)] |------|------|------|
   *                        |------|------|------|
   *                        +------|------|------+
   *                     [Input Channel * Kernel_size[0]
   *                       * Kernel_size[1] (width)]
   *
   *
   * After im2Col with channel_mode true (in : input)
   *
   * This is the 2D Matrix Shape [ height ] x [ width ]
   *   . Height : Input Channel * Kernel_size[0] * Kernel_size[1]
   *   . Width  : output_dim.height * output_dim.width
   *
   *                      +-|-|-|-|      |-|-|-|-+
   *   [Input Channel     | | | | |      | | | | |
   *   * Kernel_size[0]   |_|_|_|_|      |_|_|_|_|
   *  * Kenel_size[1]     | | | | | .... | | | | |
   *    (height)]         |_|_|_|_|      |_|_|_|_|
   *                      | | | | |      | | | | |
   *                      +_|_|_|_|      |_|_|_|_+
   *                     [ output_dim.height
   *                      * output_dim.width (width) ]
   *
   * Output Dimention
   *   -> [Channel ( = filter_size = output_dim.channel )]
   *       x [output_dim.height x output_dim.width]
   */
  for (unsigned int b = 0; b < in_dim.batch(); ++b) {
    Tensor out = hidden_.getBatchSlice(b, 1);
    Tensor in_sub = input_.getBatchSlice(b, 1);

    START_PROFILE(im2col_key);
    /// @todo allocate before batch and reuse the allocated tensor
    /// pass a memory block and use Tensor::Map
    Tensor im2col_result =
      im2col(in_sub, filter_kernel.getDim(), padding, stride, true);
    END_PROFILE(im2col_key);

    START_PROFILE(conv_gemm_profile_key);
    status = conv2d_gemm(filter_kernel.getData(), filter_kernel.getDim(),
                         im2col_result.getData(), out_dim, out.getData(), true);
    END_PROFILE(conv_gemm_profile_key);
    if (status != ML_ERROR_NONE)
      throw std::runtime_error("Forwarding Convolution failed.");
  }
  START_PROFILE(add_bias_key);
  hidden_.add_i(bias_kernel);
  END_PROFILE(add_bias_key);

  loss = 0.0f;
  if (weight_regularizer == WeightRegularizerType::l2norm) {
    loss += weight_regularizer_constant * 0.5f * (filter_kernel.l2norm());
  }
}

void Conv2DLayer::calcDerivative(sharedConstTensors derivatives) {

  int status = ML_ERROR_NONE;
  TensorDim &in_dim = input_dim[0];

  Tensor &derivative = net_hidden[0]->getGradientRef();
  Tensor &filter_kernel = weightAt(ConvParams::weight).getVariableRef();

  std::array<unsigned int, CONV2D_DIM> same_pad;
  same_pad[0] = kernel_size[0] - 1;
  same_pad[1] = kernel_size[1] - 1;

  /** Calculate return derivative
   *
   * This is the 2D Matrix Shape [ height ] x [ width ]
   *   . Height : filter.channel = input_dim.channel
   *   . Width  : filter_size * kernel_size[0] * kernel_size[1]
   *
   *                                kernel
   *                             f0      fn-1          fn
   *                            k..0     k..0         k..0
   *                          +---|---|---|---|...|---|---+
   *                          |---|---|---|---|...|---|---|
   * [filter.channel(height)] |---|---|---|---|...|---|---|
   *   (=input_dim.channel)   |---|---|---|---|...|---|---|
   *                          +---|---|---|---|...|---|---+
   *                                 [ filter_size
   *                               * kernel_size[0]
   *                            * kernel_size[1] (width) ]
   *
   *
   * After im2Col with channel_mode true ( in : derivative with full padding )
   *
   * This is the 2D Matrix Shape [ height ] x [ width ]
   *   . Height : filter_size * kernel_size[0] * kernel_size[1]
   *   . Width  : (input_dim.height + padding[0]*2) x (input_dim.width +
   * padding[1]*2)
   *
   *                      +-|-|-|-|      |-|-|-|-+
   *                      | | | | |      | | | | |
   *  [ filter_size       |_|_|_|_|      |_|_|_|_|
   *  * kernel_size[0]    | | | | | .... | | | | |
   *  * kernel_size[1]    |_|_|_|_|      |_|_|_|_|
   *    (height) ]        | | | | |      | | | | |
   *                      +_|_|_|_|      |_|_|_|_+
   *                     [(input_dim.height() + padding[0] *2)
   *                      * (input_dim.width() + padding[1] *2)]
   *
   * Output Dimension
   *
   *   -> [ input_dim.channel (height) ]
   *       x [(input_dim.height() + padding[0]*2)
   *           *(input_dim.width() + padding[1]*2) (width)]
   */
  using uint = unsigned int;

  TensorDim kdim(in_dim.channel(), filter_size, kernel_size[0], kernel_size[1]);

  uint kernel_total_size = kernel_size[0] * kernel_size[1];

  Tensor imKernel(1, 1, in_dim.channel(), filter_size * kernel_total_size);
  float *imKernel_raw = imKernel.getData();

  for (uint channel_idx = 0; channel_idx < in_dim.channel(); ++channel_idx) {
    /// each row contains all kernel element in particular channel.
    uint row_size = kernel_total_size * filter_size;
    for (uint filter_idx = 0; filter_idx < filter_size; ++filter_idx) {

      /// starting index of each kernel in imKernel
      float *start =
        imKernel_raw + channel_idx * row_size + filter_idx * kernel_total_size;
      /// starting index of each channel in filter
      float *filter_start =
        filter_kernel.getAddress(filter_idx, channel_idx, 0, 0);

      std::reverse_copy(filter_start, filter_start + kernel_total_size, start);
    }
  }

  Tensor ret;
  ret = Tensor(1, in_dim.channel(), in_dim.height() + padding[0] * 2,
               in_dim.width() + padding[1] * 2);

  TensorDim input_dim_padded = ret.getDim();

  for (unsigned int b = 0; b < in_dim.batch(); ++b) {
    Tensor inSub = derivative.getBatchSlice(b, 1);

    Tensor im2col_result = im2col(inSub, kdim, same_pad, stride, true);

    status = conv2d_gemm(imKernel_raw, kdim, im2col_result.getData(),
                         input_dim_padded, ret.getData(), true);
    if (status != ML_ERROR_NONE)
      throw std::runtime_error("calcDerivative Convolution failed.");

    strip_pad(ret, padding.data(), net_input[0]->getGradientRef(), b);
  }
}

void Conv2DLayer::calcGradient(sharedConstTensors derivatives) {
  TensorDim &in_dim = input_dim[0];

  Tensor &filter_kernel = weightAt(ConvParams::weight).getVariableRef();
  Tensor &derivative = net_hidden[0]->getGradientRef();
  Tensor &input_ = net_input[0]->getVariableRef();

  Tensor &delK = weightAt(ConvParams::weight).getGradientRef();
  Tensor &delBias = weightAt(ConvParams::bias).getGradientRef();
  delK.setZero();

  /** Calculate DelK
   *
   * This is the 2D Matrix Shape [ height ] x [ width ]
   *   . Height : filter_size
   *   . Width  : derivative.height * derivative.width
   *
   *                          derivative
   *                        +------|------+
   *                        |------|------|
   *  [filter_size (height) |------|------|
   * (=derivative->channel) |------|------|
   *                        +------|------+
   *                     [derivative->height
   *                       * derivative->width (width)]
   *
   *
   * After im2Col with channel_mode false ( in : input )
   *
   * This is the 2D Matrix Shape [ height ] x [ width ]
   *   . Height : derivative.height * derivative.width
   *   . Width  : input_dim.channel * Kernel_size[0] * Kernel_size[1]
   *
   *                      +-|-|-|-|      |-|-|-|-+
   *                      | | | | |      | | | | |
   *  [derivative->width  |_|_|_|_|      |_|_|_|_|
   * * derivative->height | | | | | .... | | | | |
   *   (height)]          +_|_|_|_|      |_|_|_|_+
   *                     [ input_dim.channel(filter_channel)  * kernel_size[0]
   *                      * kernel_size[1] (width) ]
   *
   * Output Dimension
   *   -> [ derivative->channel = filter_size (height) ]
   *       x [input_dim.channel * kernel_size[0] * kernel_size[1] (width) ]
   */

  int status = ML_ERROR_NONE;

  TensorDim kdim{
    {derivative.channel(), derivative.height(), derivative.width()}};

  TensorDim out_dim{{filter_size, in_dim.channel() * filter_kernel.height() *
                                    filter_kernel.width()}};

  for (unsigned int b = 0; b < in_dim.batch(); ++b) {
    Tensor in_sub = input_.getBatchSlice(b, 1);
    Tensor im2col_result =
      im2col(in_sub, derivative.getDim(), padding, stride, false);

    status = conv2d_gemm(
      derivative.getAddress(b * derivative.getDim().getFeatureLen()), kdim,
      im2col_result.getData(), out_dim, delK.getData(), false, 1.0);

    if (status != ML_ERROR_NONE)
      throw std::runtime_error("Backwarding Convolution failed.");
  }
  delBias = derivative.sum({0, 2, 3});

  //  Update K / bias
  if (isWeightRegularizerL2Norm()) {
    status = delK.add_i(filter_kernel, weight_regularizer_constant);
    if (status != ML_ERROR_NONE)
      throw std::runtime_error("Weight regularization failed");
  }
}

void Conv2DLayer::copy(std::shared_ptr<Layer> l) {
  Layer::copy(l);

  std::shared_ptr<Conv2DLayer> from = std::static_pointer_cast<Conv2DLayer>(l);
  this->filter_size = from->filter_size;
  for (unsigned int i = 0; i < CONV2D_DIM; ++i) {
    this->kernel_size[i] = from->kernel_size[i];
    this->stride[i] = from->stride[i];
    this->padding[i] = from->padding[i];
  }
}

int Conv2DLayer::setSize(int *size, PropertyType type) {
  int status = ML_ERROR_NONE;
  switch (type) {
  case PropertyType::kernel_size:
    for (unsigned int i = 0; i < CONV2D_DIM; ++i) {
      kernel_size[i] = size[i];
    }
    break;
  case PropertyType::stride:
    for (unsigned int i = 0; i < CONV2D_DIM; ++i) {
      stride[i] = size[i];
    }
    break;
  case PropertyType::padding:
    for (unsigned int i = 0; i < CONV2D_DIM; ++i) {
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

int Conv2DLayer::setFilter(int f) {
  int status = ML_ERROR_NONE;
  if (f <= 0) {
    ml_loge("Error: number of filters must be greater than 0");
    status = ML_ERROR_INVALID_PARAMETER;
  }
  filter_size = f;
  return status;
}

void Conv2DLayer::setProperty(const PropertyType type,
                              const std::string &value) {
  int status = ML_ERROR_NONE;

  switch (type) {
  case PropertyType::filters: {
    if (!value.empty()) {
      status = setUint(filter_size, value);
      throw_status(status);
    }
  } break;
  case PropertyType::kernel_size:
    if (!value.empty()) {
      status = getValues(CONV2D_DIM, value, (int *)(kernel_size.data()));
      throw_status(status);
      if (kernel_size[0] == 0 || kernel_size[1] == 0) {
        throw std::invalid_argument(
          "[Conv2DLayer] kernel_size must be greater than 0");
      }
    }
    break;
  case PropertyType::stride:
    if (!value.empty()) {
      status = getValues(CONV2D_DIM, value, (int *)(stride.data()));
      throw_status(status);
      if (stride[0] == 0 || stride[1] == 0) {
        throw std::invalid_argument(
          "[Conv2DLayer] stride must be greater than 0");
      }
    }
    break;
  case PropertyType::padding:
    if (!value.empty()) {
      status = getValues(CONV2D_DIM, value, (int *)(padding.data()));
      throw_status(status);
    }
    break;
  default:
    Layer::setProperty(type, value);
    break;
  }
}

int Conv2DLayer::conv2d_gemm(const float *mkernel, TensorDim kdim,
                             const float *in, TensorDim outdim, float *out,
                             bool channel_mode, float beta_dgemm) {
  int status = ML_ERROR_NONE;

  float alpha_dgemm = 1.0f;
  const float *data = mkernel;
  float *rdata = out;

  unsigned int kh, kw, w;

  if (channel_mode) {
    kh = kdim.batch();
    kw = kdim.getFeatureLen();
    w = outdim.width() * outdim.height();
  } else {
    kh = outdim.height();
    kw = kdim.width() * kdim.height();
    w = outdim.width();
  }

  sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, kh, w, kw, alpha_dgemm, data,
        kw, in, w, beta_dgemm, rdata, w);

  return status;
}

Tensor Conv2DLayer::im2col(const Tensor &in, const TensorDim &kdim,
                           const std::array<unsigned int, CONV2D_DIM> &padding,
                           const std::array<unsigned int, CONV2D_DIM> &mstride,
                           bool channel_mode) {
  /// @todo: add dimension validation here
  const int pad_value = 0;
  unsigned int ph = padding[0];
  unsigned int pw = padding[1];

  unsigned int channel = in.channel();
  unsigned int height = in.height() + ph * 2;
  unsigned int width = in.width() + pw * 2;
  unsigned int k_height = kdim.height();
  unsigned int k_width = kdim.width();
  unsigned int out_height = (height - k_height) / mstride[0] + 1;
  unsigned int out_width = (width - k_width) / mstride[1] + 1;

  Tensor im2col_array;
  if (channel_mode) {
    im2col_array = Tensor(kdim.getFeatureLen(), out_height * out_width);
  } else {
    im2col_array =
      Tensor(k_height * k_width, in.channel() * out_height * out_width);
  }

  if (pad_value == 0) {
    im2col_array.setZero();
  } else {
    /// not reaching here, just preparing for non-zero pad_value
    im2col_array.setValue(pad_value);
  }

  auto in_range = [](unsigned int virtual_pos, unsigned int pad,
                     unsigned int actual_len) -> bool {
    return pad <= virtual_pos && virtual_pos < (pad + actual_len);
  };

  if (channel_mode) {
    unsigned int h_stride_end = height - k_height;
    unsigned int w_stride_end = width - k_width;
    /// get a patch, size of kernel
    /// hs is height_strided, ws is width_strided
    unsigned int im_w = 0;
    for (unsigned int hs = 0; hs <= h_stride_end; hs += mstride[0]) {
      for (unsigned int ws = 0; ws <= w_stride_end; ws += mstride[1]) {
        unsigned int im_h = 0;
        unsigned int patch_height_end = k_height + hs;
        unsigned int patch_width_end = k_width + ws;

        for (unsigned int c = 0; c < channel; ++c) {
          /// map the patch to a single line loopting through channel
          for (unsigned int h = hs; h < patch_height_end; ++h) {
            if (!in_range(h, ph, in.height())) {
              im_h += k_width;
              continue;
            }

            for (unsigned int w = ws; w < patch_width_end; ++w) {
              if (!in_range(w, pw, in.width())) {
                im_h++;
                continue;
              }
              float val = in.getValue(0, c, h - ph, w - pw);
              im2col_array.setValue(0, 0, im_h, im_w, val);
              im_h++;
            }
          }
        }
        im_w++;
      }
    }
  } else {
    unsigned int im_w = 0;
    for (unsigned int c = 0; c < channel; ++c) {
      for (unsigned int hs = 0; hs <= height - k_height; hs += mstride[0]) {
        for (unsigned int ws = 0; ws <= width - k_width; ws += mstride[1]) {
          unsigned int im_h = 0;
          unsigned int patch_height_end = k_height + hs;
          unsigned int patch_width_end = k_width + ws;

          for (unsigned int h = hs; h < patch_height_end; ++h) {
            if (!in_range(h, ph, in.height())) {
              im_h += k_width;
              continue;
            }

            for (unsigned int w = ws; w < patch_width_end; ++w) {
              if (!in_range(w, pw, in.width())) {
                im_h++;
                continue;
              }
              float val = in.getValue(0, c, h - ph, w - pw);
              im2col_array.setValue(0, 0, im_h, im_w, val);
              im_h++;
            }
          }
          im_w++;
        }
      }
    }
  }

  return im2col_array;
}

void Conv2DLayer::scaleSize(float scalesize) noexcept {
  filter_size = (unsigned int)(scalesize * (float)filter_size);
  filter_size = std::max(filter_size, 1u);
}

} /* namespace nntrainer */
