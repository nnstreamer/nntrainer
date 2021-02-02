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
namespace {

struct ChannelToCol {};
struct ChannelToRow {};

/**
 * @brief convolution operation abstracted to
 * in \x\ filter = out, where \x\ denotes convolution operation
 *
 * @tparam T either ChannelToRow, ChannelToCol.
 * ChannelToCol is channel mode (channel goes to column)
 * ChannelToRow is non-channel mode (channle goes to row)
 * @param in input
 * @param filter filter to convolution to
 * @param out target output, it should be already allocated with end dimension
 * @param padding {height, width} padding
 * @param stride {height, width} stride
 * @param dilation {height, width} dilation
 */
template <typename T>
void conv2d_op(const Tensor &in, const Tensor &filter, Tensor &out,
               const std::array<unsigned, CONV2D_DIM> &padding = {0, 0},
               const std::array<unsigned, CONV2D_DIM> &stride = {1, 1},
               const std::array<unsigned, CONV2D_DIM> &dilation = {1, 1}) {
  throw std::invalid_argument("conv2d_op need to be called with a tag");
}

/**
 * @brief implementation of convolution ops needed when channel need to go to
 * column, the example describes what to expect, internally the im2col array
 * is transposed to maximize cache locality and minimize memcopy
 * eg)    x     \*\     filter      =     y, where \*\ denotes conv operation
 * (N, C, H, W) \*\ (FN, C, FH, FW) = (N, FN, OH, OW)
 * ==> for each batch, slice imx
 * [1] imx <- im2col(x)
 * [2]   filter  x             imx              =       y
 * (FN, FH * FW * C) x (FH * FW * C, OH * OW)   = (FN, OH * OW)
 * [3] y <- reshape(y, (FN, OH, OW))
 *
 * @copydoc template<typename T>conv2d_op(const Tensor &in, const Tensor
 * &filter, Tensor &out, const std::array<unsigned, CONV2D_DIM> &padding, const
 * std::array<unsigned, CONV2D_DIM> &stride, const std::array<unsgined,
 * CONV2D_DIM> &dilation)
 */
template <>
void conv2d_op<ChannelToCol>(const Tensor &in, const Tensor &filter,
                             Tensor &out,
                             const std::array<unsigned, CONV2D_DIM> &padding,
                             const std::array<unsigned, CONV2D_DIM> &stride,
                             const std::array<unsigned, CONV2D_DIM> &dilation) {

  NNTR_THROW_IF(in.channel() != filter.channel(), std::invalid_argument)
    << "channel mismatch, in_dim: " << in.getDim()
    << "out_dim: " << out.getDim();

  TensorDim filter_dim = filter.getDim();
  Tensor f = filter;

  f.reshape({filter.batch(), filter.getDim().getFeatureLen()});

  // todo: allocate im2col_result beforehand to save time
  Tensor im2col_result;
  for (unsigned int b = 0; b < in.batch(); ++b) {
    Tensor out_sub = out.getBatchSlice(b, 1);
    Tensor in_sub = in.getBatchSlice(b, 1);

    out_sub.reshape({out.channel(), out.width() * out.height()});
    Conv2DLayer::im2col(in_sub, filter_dim, padding, stride, true,
                        im2col_result);
    f.dot(im2col_result, out_sub, false, true);
  }
}

/**
 * @brief implementation of convolution ops needed when channel need to go to
 * row, the example describes what to expect
 * eg)    x     \*\       dy        =       dw, where \*\ denotes conv operation
 * (N, C, H, W) \*\ (N, FN, OH, OW) = (FN, C, FH, FW)
 * ==> for each batch, slice dy, x
 * [1] imx <- im2col(x)
 * [2]   dy      x        imx             =         dw
 * (FN, OH * OW) x (OH * OW, C * FH * FW) = (FN, C * FH * FW)
 * [3] dw <- reshape(dw, (FN, C, FH, FW))
 *
 * @copydoc template<typename T>conv2d_op(const Tensor &in, const Tensor
 * &filter, Tensor &out, const std::array<unsigned, CONV2D_DIM> &padding, const
 * std::array<unsigned, CONV2D_DIM> &stride, const std::array<unsgined,
 * CONV2D_DIM> &dilation)
 */
template <>
void conv2d_op<ChannelToRow>(const Tensor &in, const Tensor &filter,
                             Tensor &out,
                             const std::array<unsigned, CONV2D_DIM> &padding,
                             const std::array<unsigned, CONV2D_DIM> &stride,
                             const std::array<unsigned, CONV2D_DIM> &dilation) {
  Tensor f = filter;
  f.reshape(
    {filter.batch(), 1, filter.channel(), filter.height() * filter.width()});

  Tensor o = out;
  o.reshape({out.batch(), out.channel() * out.height() * out.width()});
  o.setZero(); // initialize to 0 as it accumulates

  // todo: allocate im2col_result beforehand to save time
  Tensor im2col_result;
  for (unsigned int b = 0; b < in.batch(); ++b) {
    Tensor in_sub = in.getBatchSlice(b, 1);
    Tensor filter_sub = f.getBatchSlice(b, 1);

    Conv2DLayer::im2col(in_sub, filter.getDim(), padding, stride, false,
                        im2col_result);

    filter_sub.dot(im2col_result, o, false, false, 1.0f);
  }
}
} // namespace

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

void Conv2DLayer::forwarding(bool training) {
  int status = ML_ERROR_NONE;

  if (num_inputs != 1)
    throw std::invalid_argument("Convolution layer only takes one input");

  Tensor &input_ = net_input[0]->getVariableRef();
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
  conv2d_op<ChannelToCol>(input_, filter_kernel, hidden_, padding, stride);

  status = hidden_.add_i(bias_kernel);
  if (status != ML_ERROR_NONE) {
    throw std::invalid_argument("[Conv2D] adding bias failed");
  }
  loss = 0.0f;
  if (weight_regularizer == WeightRegularizerType::l2norm) {
    loss += weight_regularizer_constant * 0.5f * (filter_kernel.l2norm());
  }
}

void Conv2DLayer::calcDerivative() {
  Tensor &derivative = net_hidden[0]->getGradientRef();
  Tensor &filter_kernel = weightAt(ConvParams::weight).getVariableRef();

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
  Tensor flipped = filter_kernel.convFlip();

  std::array<unsigned int, CONV2D_DIM> same_pad;
  same_pad[0] = kernel_size[0] - 1;
  same_pad[1] = kernel_size[1] - 1;

  conv2d_op<ChannelToCol>(derivative, flipped, net_input[0]->getGradientRef(),
                          same_pad, {1, 1}, {1, 1});
}

void Conv2DLayer::calcGradient() {
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

  conv2d_op<ChannelToRow>(input_, derivative, delK, padding, stride);

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

void Conv2DLayer::im2col(const Tensor &in, const TensorDim &kdim,
                         const std::array<unsigned int, CONV2D_DIM> &padding,
                         const std::array<unsigned int, CONV2D_DIM> &mstride,
                         bool channel_mode, Tensor &out) {
  const int pad_value = 0;
  unsigned int ph = padding[0];
  unsigned int pw = padding[1];

  unsigned int channel = in.channel();
  int in_height = in.height();
  int in_width = in.width();
  unsigned int height = in_height + ph * 2;
  unsigned int width = in_width + pw * 2;
  unsigned int k_height = kdim.height();
  unsigned int k_width = kdim.width();
  unsigned int out_height = (height - k_height) / mstride[0] + 1;
  unsigned int out_width = (width - k_width) / mstride[1] + 1;

  NNTR_THROW_IF((height - k_height) % mstride[0] != 0, std::invalid_argument)
    << "calculated height is not integer. "
    << "padded height: " << height << " kernel height: " << k_height
    << " stride: " << mstride[0];

  NNTR_THROW_IF((width - k_width) % mstride[1] != 0, std::invalid_argument)
    << "calculated width is not integer. "
    << "padded width: " << width << " kernel width: " << k_width
    << " stride: " << mstride[1];

  if (out.uninitialized()) {
    if (channel_mode) {
      out = Tensor(out_height * out_width, in.channel() * k_height * k_width);
    } else {
      out = Tensor(k_height * k_width, in.channel() * out_height * out_width);
    }

    if (pad_value == 0) {
      out.setZero();
    } else {
      /// not reaching here, just preparing for non-zero pad_value
      out.setValue(pad_value);
    }
  }

  float *out_data = out.getData();

  if (channel_mode) {
    int h_stride_end = height - k_height - ph;
    int w_stride_end = width - k_width - pw;

    /// get a patch, size of kernel
    /// hs is height_strided, ws is width_strided
    unsigned int owidth = out.width();
    unsigned int base_im_w = 0;
    for (int hs = -ph; hs <= h_stride_end; hs += mstride[0]) {
      unsigned int base_im_h = 0;
      int patch_height_end = k_height + hs;
      /// map the patch to a single line looping through channel
      for (unsigned int c = 0; c < channel; ++c) {
        for (int h = hs; h < patch_height_end; ++h) {
          if (h < 0 || in_height <= h) {
            base_im_h += k_width;
            continue;
          }

          unsigned int im_w = base_im_w;
          for (int ws = -pw; ws <= w_stride_end; ws += mstride[1]) {
            unsigned int im_h = base_im_h;
            int patch_width_end = k_width + ws;

            for (int w = ws; w < patch_width_end; ++w) {
              if (w < 0 || in_width <= w) {
                im_h++;
                continue;
              }
              out_data[im_w * owidth + im_h] = in.getValue(0, c, h, w);
              im_h++;
            }
            im_w++;
          }
          base_im_h += k_width;
        }
      }
      base_im_w += out_width;
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
            if (h < ph || in_height + ph <= h) {
              im_h += k_width;
              continue;
            }

            for (unsigned int w = ws; w < patch_width_end; ++w) {
              if (w < pw || in_width + pw <= w) {
                im_h++;
                continue;
              }

              float val = in.getValue(0, c, h - ph, w - pw);
              out.setValue(0, 0, im_h, im_w, val);
              im_h++;
            }
          }
          im_w++;
        }
      }
    }
  }
}

void Conv2DLayer::scaleSize(float scalesize) noexcept {
  filter_size = (unsigned int)(scalesize * (float)filter_size);
  filter_size = std::max(filter_size, 1u);
}

} /* namespace nntrainer */
