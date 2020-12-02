// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file	conv2d_layer.h
 * @date	02 June 2020
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
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
#include <util_func.h>

namespace nntrainer {

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

  return status;
}

void Conv2DLayer::forwarding(sharedConstTensors in) {
  int status = ML_ERROR_NONE;

  if (num_inputs != 1)
    throw std::invalid_argument("Convolution layer only takes one input");

  Tensor &input_ = net_input[0]->var;

  TensorDim &in_dim = input_dim[0];
  TensorDim &out_dim = output_dim[0];

  Tensor &hidden_ = net_hidden[0]->var;
  /** @todo This check is redundant, remove it later */
  if (hidden_.uninitialized()) {
    hidden_ = Tensor(out_dim);
  }

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
    Tensor inSub = input_.getBatchSlice(b, 1);

    status =
      conv2d_gemm(filter_kernel.getData(), filter_kernel.getDim(), inSub,
                  out_dim, stride, padding, out.getData(), out.length(), true);
    if (status != ML_ERROR_NONE)
      throw std::runtime_error("Forwarding Convolution failed.");
  }
  hidden_.add_i(bias_kernel);

  loss = 0.0f;
  if (weight_regularizer == WeightRegularizerType::l2norm) {
    loss += weight_regularizer_constant * 0.5f * (filter_kernel.l2norm());
  }
}

void Conv2DLayer::calcDerivative(sharedConstTensors derivatives) {

  int status = ML_ERROR_NONE;
  TensorDim &in_dim = input_dim[0];

  Tensor &derivative = net_hidden[0]->var;
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
   *                             f0      f1          fn
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
   *   -> [ input_dim.chennel (height) ]
   *       x [(input_dim.height() + padding[0]*2)
   *           *(input_dim.width() + padding[1]*2) (width)]
   */
  using uint = unsigned int;
  bool no_padding = padding[0] == 0 && padding[1] == 0;

  if (net_input[0]->var.uninitialized())
    net_input[0]->var = Tensor(input.getDim());

  Tensor ret;
  if (no_padding)
    ret = net_input[0]->var;
  else
    ret =
      Tensor(in_dim.batch(), in_dim.channel(), in_dim.height() + padding[0] * 2,
             in_dim.width() + padding[1] * 2);

  TensorDim kdim(ret.channel(), filter_size, kernel_size[0], kernel_size[1]);

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

  TensorDim input_dim_padded = TensorDim(ret.getDim());
  input_dim_padded.batch(1);

  for (unsigned int b = 0; b < in_dim.batch(); ++b) {
    Tensor inSub = derivative.getBatchSlice(b, 1);

    status =
      conv2d_gemm(imKernel_raw, kdim, inSub, input_dim_padded, stride, same_pad,
                  ret.getAddress(b * ret.getDim().getFeatureLen()),
                  input_dim_padded.getFeatureLen(), true);
    if (status != ML_ERROR_NONE)
      throw std::runtime_error("calcDerivative Convolution failed.");
  }

  if (!no_padding)
    strip_pad(ret, padding.data(), net_input[0]->var);
}

void Conv2DLayer::calcGradient(sharedConstTensors derivatives) {
  TensorDim &in_dim = input_dim[0];

  Tensor &filter_kernel = weightAt(ConvParams::weight).getVariableRef();
  Tensor &derivative = net_hidden[0]->var;
  Tensor &input_ = net_input[0]->var;

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
   *                     [ input_dim.channel * kernel_size[0]
   *                      * kernel_size[1] (width) ]
   *
   * Output Dimension
   *   -> [ derivative->channel = filter_size (height) ]
   *       x [input_dim.channel * kernel_size[0] * kernel_size[1] (width) ]
   */

  int status = ML_ERROR_NONE;
  for (unsigned int b = 0; b < in_dim.batch(); ++b) {
    Tensor inSub = input_.getBatchSlice(b, 1);

    status = conv2d_gemm(
      derivative.getAddress(b * derivative.getDim().getFeatureLen()),
      TensorDim(1, derivative.channel(), derivative.height(),
                derivative.width()),
      inSub,
      TensorDim(1, 1, filter_size,
                kernel_size[0] * kernel_size[1] * in_dim.channel()),
      stride, padding, delK.getData(), delK.length(), false, 1.0);
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

int Conv2DLayer::conv2d_gemm(
  const float *mkernel, TensorDim kdim, Tensor const &in, TensorDim outdim,
  const std::array<unsigned int, CONV2D_DIM> &mstride,
  const std::array<unsigned int, CONV2D_DIM> &pad, float *out,
  unsigned int osize, bool channel_mode, float beta_dgemm) {
  int status = ML_ERROR_NONE;
  std::vector<float> in_col;

  if (channel_mode) {
    in_col.resize(kdim.getFeatureLen() * outdim.width() * outdim.height());
  } else {
    in_col.resize(kdim.width() * kdim.height() * outdim.width());
  }

  Tensor in_padded = zero_pad(0, in, pad.data());
  status =
    im2col(in_padded, kdim, in_col.data(), outdim, mstride, channel_mode);
  if (status != ML_ERROR_NONE)
    throw std::runtime_error("Forwarding Convolution failed.");

  float alpha_dgemm = 1.0f;
  const float *data = mkernel;
  const float *mdata = in_col.data();
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
        kw, mdata, w, beta_dgemm, rdata, w);

  return status;
}

int Conv2DLayer::im2col(Tensor in_padded, TensorDim kdim, float *in_col,
                        TensorDim outdim,
                        const std::array<unsigned int, CONV2D_DIM> &mstride,
                        bool channel_mode) {

  int status = ML_ERROR_NONE;
  unsigned int count;
  unsigned int channel = in_padded.channel();
  unsigned int height = in_padded.height();
  unsigned int width = in_padded.width();
  unsigned int k_width = kdim.width();
  unsigned int k_height = kdim.height();

  unsigned int J = 0;
  if (channel_mode) {
    for (unsigned int j = 0; j <= height - k_height; j += mstride[0]) {
      for (unsigned int k = 0; k <= width - k_width; k += mstride[1]) {
        count = 0;
        for (unsigned int i = 0; i < channel; ++i) {
          for (unsigned int ki = 0; ki < k_height; ++ki) {
            for (unsigned int kj = 0; kj < k_width; ++kj) {
              in_col[count * (outdim.width() * outdim.height()) + J] =
                in_padded
                  .getData()[i * height * width + (j + ki) * width + (k + kj)];
              count++;
            }
          }
        }
        J++;
      }
    }
    if (J != outdim.width() * outdim.height())
      status = ML_ERROR_INVALID_PARAMETER;
  } else {
    for (unsigned int i = 0; i < channel; ++i) {
      for (unsigned int j = 0; j <= height - k_height; j += mstride[0]) {
        for (unsigned int k = 0; k <= width - k_width; k += mstride[1]) {
          count = 0;
          for (unsigned int ki = 0; ki < k_height; ++ki) {
            for (unsigned int kj = 0; kj < k_width; ++kj) {
              in_col[count * (outdim.width()) + J] =
                in_padded
                  .getData()[i * height * width + (j + ki) * width + (k + kj)];
              count++;
            }
          }
          J++;
        }
      }
    }
    if (J != outdim.width())
      status = ML_ERROR_INVALID_PARAMETER;
  }

  return status;
}

void Conv2DLayer::scaleSize(float scalesize) noexcept {
  filter_size = (unsigned int)(scalesize * (float)filter_size);
  filter_size = std::max(filter_size, 1u);
}

} /* namespace nntrainer */
