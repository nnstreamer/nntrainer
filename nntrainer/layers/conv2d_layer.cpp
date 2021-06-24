// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   conv2d_layer.h
 * @date   02 June 2020
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is Convolution Layer Class for Neural Network
 *
 */
#include <algorithm>
#include <cstring>
#include <limits>
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

/**
 * @brief     reconstruct image data from 2d column matrix
 *
 * @param[in] in input data
 * @param[in] kdim kernel dimesion for define number of row
 * @param[in] padding padding information
 * @param[in] mstride stride value : x, y direction
 * @param[in] dilation kernel dilation factor : x, y each
 * @param[out] image image tensor to put
 */
static void col2im(const Tensor &col_matrix, const TensorDim &kdim,
                   const std::array<unsigned, 4> &padding,
                   const std::array<unsigned, CONV2D_DIM> &mstride,
                   const std::array<unsigned, CONV2D_DIM> &dilation,
                   Tensor &image) {
  unsigned pt = padding[0];
  unsigned pb = padding[1];
  unsigned pl = padding[2];
  unsigned pr = padding[3];

  unsigned k_height = kdim.height();
  unsigned k_width = kdim.width();

  /// effective kernel height considering dilation
  unsigned eff_k_height = (k_height - 1) * dilation[0] + 1;
  /// effective kernel width considering dilation
  unsigned eff_k_width = (k_width - 1) * dilation[1] + 1;

  unsigned im_channel = image.channel();
  int im_height = image.height();
  int im_width = image.width();

  unsigned hstride = mstride[0];
  unsigned wstride = mstride[1];

  unsigned hdilation = dilation[0];
  unsigned wdilation = dilation[1];

  /// image considering padding
  unsigned im_eff_height = im_height + pt + pb;
  unsigned im_eff_width = im_width + pl + pr;
  image.setZero();

  int h_stride_end = im_eff_height - eff_k_height - pb;
  int w_stride_end = im_eff_width - eff_k_width - pr;

  unsigned col_w = 0;
  for (int hs = -pt; hs <= h_stride_end; hs += hstride) {
    for (int ws = -pl; ws <= w_stride_end; ws += wstride) {
      unsigned col_h = 0;
      int patch_height_end = hs + eff_k_height;
      int patch_width_end = ws + eff_k_width;
      for (unsigned c = 0; c < im_channel; c++) {
        for (int h = hs; h < patch_height_end; h += hdilation) {
          if (h < 0 || im_height <= h) {
            col_h += k_width;
            continue;
          }
          for (int w = ws; w < patch_width_end; w += wdilation) {
            if (w < 0 || im_width <= w) {
              col_h++;
              continue;
            }

            float *val = image.getAddress(0, c, h, w);
            *val += col_matrix.getValue(0, 0, col_h, col_w);
            col_h++;
          }
        }
      }
      col_w++;
    }
  }
}

/**
 * @brief     reform the data to 2d matrix
 * a region is sampled considering @a padding, @a mstride of unit @a kdim
 * Each region is mapped to one column,
 * if channel mode, kernel channel is considered part of kernel feature
 * if not, kernel channel is consider part of output dimension
 *
 * @param[in] in input data
 * @param[in] kdim kernel dimesion for define number of row
 * @param[in] padding padding information
 * @param[in] mstride stride value : x, y direction
 * @param[in] dilation kernel dilation factor : x, y each
 * @param[out] out out tensor to put, if uninitialized, allocate a new tensor
 * and set padding
 * @note if out is initialized tensor, setting padding is skipped.
 */
static void im2col(const Tensor &in, const TensorDim &kdim,
                   const std::array<unsigned int, 4> &padding,
                   const std::array<unsigned int, CONV2D_DIM> &mstride,
                   const std::array<unsigned int, CONV2D_DIM> &dilation,
                   Tensor &out) {
  /// for channel last mode, this is deprecated for now, leaving here on
  /// purpose.
  /** @code
  //   ================ initialize part ====================
  //   out_height -= 2;
  //   out =
  //     Tensor(k_height * k_width, in.channel() * (out_height) *
  //     (out_width));
  //   unsigned int im_w = 0;
  //   ================ loop part ====================
  //   if (eff_k_height > height || eff_k_width > width)
  //     throw std::runtime_error("Kernel shape bigger than input shape");

  //   for (unsigned int c = 0; c < channel; ++c) {
  //     for (unsigned int hs = 0; hs <= height - eff_k_height; hs +=
  //     mstride[0]) {
  //       for (unsigned int ws = 0; ws <= width - eff_k_width; ws +=
  //       mstride[1]) {
  //         unsigned int im_h = 0;
  //         unsigned int patch_height_end = eff_k_height + hs;
  //         unsigned int patch_width_end = eff_k_width + ws;

  //         for (unsigned int h = hs; h < patch_height_end; h += dilation[0]) {
  //           if (h < ph || in_height + ph <= h) {
  //             im_h += k_width;
  //             continue;
  //           }

  //           for (unsigned int w = ws; w < patch_width_end; w += dilation[1])
  //           {
  //             if (w < pw || in_width + pw <= w) {
  //               im_h++;
  //               continue;
  //             }

  //             float val = in.getValue(0, c, h - ph, w - pw);
  //             out.setValue(0, 0, im_h, im_w, val);
  //             im_h++;
  //           }
  //         }
  //         im_w++;
  //       }
  //     }
  //   }
  */

  const int pad_value = 0;
  unsigned pt = padding[0];
  unsigned pb = padding[1];
  unsigned pl = padding[2];
  unsigned pr = padding[3];

  unsigned int channel = in.channel();
  int in_height = in.height();
  int in_width = in.width();
  unsigned int height = in_height + pt + pb;
  unsigned int width = in_width + pl + pr;
  unsigned int k_height = kdim.height();
  unsigned int k_width = kdim.width();

  /// effective kernel height considering dilation
  unsigned int eff_k_height = (k_height - 1) * dilation[0] + 1;
  /// effective kernel width considering dilation
  unsigned int eff_k_width = (k_width - 1) * dilation[1] + 1;

  unsigned int out_height = (height - eff_k_height) / mstride[0] + 1;
  unsigned int out_width = (width - eff_k_width) / mstride[1] + 1;

  if (out.uninitialized()) {
    out = Tensor(out_height * out_width, in.channel() * k_height * k_width);

    if (pad_value == 0) {
      out.setZero();
    } else {
      /// not reaching here, just preparing for non-zero pad_value
      out.setValue(pad_value);
    }
  }

  float *out_data = out.getData();

  int h_stride_end = height - eff_k_height - pb;
  int w_stride_end = width - eff_k_width - pr;

  /// get a patch, size of kernel
  /// hs is height_strided, ws is width_strided
  unsigned int owidth = out.width();
  unsigned int base_im_w = 0;
  for (int hs = -pt; hs <= h_stride_end; hs += mstride[0]) {
    unsigned int base_im_h = 0;
    int patch_height_end = eff_k_height + hs;
    /// map the patch to a single line looping through channel
    for (unsigned int c = 0; c < channel; ++c) {
      for (int h = hs; h < patch_height_end; h += dilation[0]) {
        if (h < 0 || in_height <= h) {
          base_im_h += k_width;
          continue;
        }

        unsigned int im_w = base_im_w;
        for (int ws = -pl; ws <= w_stride_end; ws += mstride[1]) {
          unsigned int im_h = base_im_h;
          int patch_width_end = eff_k_width + ws;

          for (int w = ws; w < patch_width_end; w += dilation[1]) {
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
}

} // namespace

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

  TensorDim dim =
    TensorDim(filter_size, in_dim.channel(), kernel_size[0], kernel_size[1]);
  TensorDim bias_dim = TensorDim(1, filter_size, 1, 1);

  padding = std::get<props::Padding2D>(conv_props).compute(in_dim, dim);

  if (weights.empty()) {
    weights.reserve(2);
    weights.emplace_back(dim, weight_initializer, weight_regularizer,
                         weight_regularizer_constant, true, false,
                         "Conv2d:filter");
    weights.emplace_back(bias_dim, bias_initializer, WeightRegularizer::NONE,
                         1.0f, true, false, "Conv2d:bias");
    manager.trackWeights(weights);
  } else {
    weights[ConvParams::weight].reset(dim, weight_initializer,
                                      weight_regularizer,
                                      weight_regularizer_constant, true);
    weights[ConvParams::bias].reset(bias_dim, bias_initializer,
                                    WeightRegularizer::NONE, 1.0f, true);
  }

  /// we don't have same padding for now but later, same padding don't apply
  /// when kernel size is even in current implementation (we need to handle
  /// assymetric padding)

  // this output_dim must be the same with dimension of hidden
  unsigned int eff_in_height = in_dim.height() + padding[0] + padding[1];
  unsigned int eff_in_width = in_dim.width() + padding[2] + padding[3];

  out_dim.batch(in_dim.batch());
  out_dim.channel(filter_size);
  out_dim.height((eff_in_height - kernel_size[0]) / stride[0] + 1);
  out_dim.width((eff_in_width - kernel_size[1]) / stride[1] + 1);

  if (eff_in_height < kernel_size[0] || eff_in_width < kernel_size[1]) {
    ml_loge("Failed to initialize: in size + padding is smaller than effective "
            "kernel");
    return ML_ERROR_INVALID_PARAMETER;
  }

  unsigned int IM = std::numeric_limits<int>::max();

  if (eff_in_height - padding[0] - kernel_size[0] > IM ||
      eff_in_width - padding[2] - kernel_size[1] > IM) {
    ml_loge("Failed to initialize: Calculated patch end is over int max");
    return ML_ERROR_INVALID_PARAMETER;
  }

  return status;
}

void Conv2DLayer::forwarding(bool training) {
  int status = ML_ERROR_NONE;

  if (getNumInputs() != 1)
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
  TensorDim &in_dim = input_dim[0];
  TensorDim &out_dim = output_dim[0];
  TensorDim filter_dim = filter_kernel.getDim();
  TensorDim filter_dim_squeezed{filter_kernel.batch(),
                                filter_kernel.getDim().getFeatureLen()};

  filter_kernel.reshape(filter_dim_squeezed);

  /// @note allocating this at initialize phase will save initialization time
  /// with extra memory overhead
  Tensor im2col_result;
  for (unsigned int b = 0; b < in_dim.batch(); ++b) {
    Tensor out = hidden_.getBatchSlice(b, 1);
    out.reshape({filter_size, out_dim.width() * out_dim.height()});

    Tensor in_sub = input_.getBatchSlice(b, 1);

    im2col(in_sub, filter_dim, padding, stride, {1, 1}, im2col_result);

    filter_kernel.dot(im2col_result, out, false, true);
  }

  filter_kernel.reshape(filter_dim);
  status = hidden_.add_i(bias_kernel);
  if (status != ML_ERROR_NONE) {
    throw std::invalid_argument("[Conv2D] adding bias failed");
  }

  loss = weightAt(ConvParams::weight).getRegularizationLoss();
}

void Conv2DLayer::calcDerivative() {
  Tensor &derivative = net_hidden[0]->getGradientRef();
  Tensor &input_derivative = net_input[0]->getGradientRef();
  Tensor &filter_kernel = weightAt(ConvParams::weight).getVariableRef();

  TensorDim filter_dim = filter_kernel.getDim();
  TensorDim filter_dim_squeezed{filter_kernel.batch(),
                                filter_kernel.getDim().getFeatureLen()};

  filter_kernel.reshape(filter_dim_squeezed);

  /// for each batch
  /// filter_kernel^T X derivaitive  -> column matrix
  /// col2im(column matrix) to reconstruct the original image
  Tensor col2im_result;
  for (unsigned int b = 0; b < derivative.batch(); ++b) {
    Tensor deriv_sub = derivative.getBatchSlice(b, 1);
    Tensor in_deriv_sub = input_derivative.getBatchSlice(b, 1);
    deriv_sub.reshape({filter_size, derivative.width() * derivative.height()});

    filter_kernel.dot(deriv_sub, col2im_result, true, false);
    col2im(col2im_result, filter_dim, padding, stride, {1, 1}, in_deriv_sub);
  }

  filter_kernel.reshape(filter_dim);
}

void Conv2DLayer::calcGradient() {
  Tensor &derivative = net_hidden[0]->getGradientRef();
  Tensor &input_ = net_input[0]->getVariableRef();

  Tensor &delK = weightAt(ConvParams::weight).getGradientRef();
  Tensor &delBias = weightAt(ConvParams::bias).getGradientRef();
  delK.setZero();

  TensorDim filter_dim = delK.getDim();
  TensorDim filter_dim_squeezed{filter_dim.batch(), filter_dim.getFeatureLen()};

  delK.reshape(filter_dim_squeezed);

  Tensor im2col_result;
  TensorDim out_dim_squeezed{filter_size,
                             derivative.width() * derivative.height()};

  /// input -(im2col)-> column_matrix -> filter x (column_matrix) = output
  /// so delK = dy x column_matrix ^ T;
  for (unsigned int b = 0; b < input_.batch(); ++b) {
    Tensor deriv_sub = derivative.getBatchSlice(b, 1);
    deriv_sub.reshape(out_dim_squeezed);

    Tensor in_sub = input_.getBatchSlice(b, 1);

    im2col(in_sub, filter_dim, padding, stride, {1, 1}, im2col_result);
    deriv_sub.dot(im2col_result, delK, false, false, b == 0 ? 0 : 1);
  }

  delK.reshape(filter_dim);
  delBias = derivative.sum({0, 2, 3});
}

void Conv2DLayer::copy(std::shared_ptr<LayerV1> l) {
  LayerV1::copy(l);

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
      from_string(value, std::get<props::Padding2D>(conv_props));
    }
    break;
  default:
    LayerV1::setProperty(type, value);
    break;
  }
}

} /* namespace nntrainer */
