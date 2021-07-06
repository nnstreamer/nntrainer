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

static constexpr size_t SINGLE_INOUT_IDX = 0;

namespace {

static TensorDim calcCol2ImOutputDim(const TensorDim &out,
                                     const TensorDim &kdim) {

  return TensorDim({kdim.getFeatureLen(), out.width() * out.height()});
}

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
                   const std::array<unsigned, CONV2D_DIM> &padding,
                   const std::array<unsigned, CONV2D_DIM> &mstride,
                   const std::array<unsigned, CONV2D_DIM> &dilation,
                   Tensor &image) {
  unsigned ph = padding[0];
  unsigned pw = padding[1];

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
  unsigned im_eff_height = im_height + 2 * ph;
  unsigned im_eff_width = im_width + 2 * pw;
  image.setZero();

  int h_stride_end = im_eff_height - eff_k_height - ph;
  int w_stride_end = im_eff_width - eff_k_width - pw;

  unsigned col_w = 0;
  for (int hs = -ph; hs <= h_stride_end; hs += hstride) {
    for (int ws = -pw; ws <= w_stride_end; ws += wstride) {
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

static TensorDim
calcIm2ColOutputDim(const TensorDim &in, const TensorDim &kdim,
                    const std::array<unsigned int, CONV2D_DIM> &padding,
                    const std::array<unsigned int, CONV2D_DIM> &mstride,
                    const std::array<unsigned int, CONV2D_DIM> &dilation) {

  unsigned int ph = padding[0];
  unsigned int pw = padding[1];

  int in_height = in.height();
  int in_width = in.width();
  unsigned int height = in_height + ph * 2;
  unsigned int width = in_width + pw * 2;
  unsigned int k_height = kdim.height();
  unsigned int k_width = kdim.width();

  /// effective kernel height considering dilation
  unsigned int eff_k_height = (k_height - 1) * dilation[0] + 1;
  /// effective kernel width considering dilation
  unsigned int eff_k_width = (k_width - 1) * dilation[1] + 1;

  unsigned int out_height = (height - eff_k_height) / mstride[0] + 1;
  unsigned int out_width = (width - eff_k_width) / mstride[1] + 1;

  return TensorDim({out_height * out_width, in.channel() * k_height * k_width});
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
 * @param[out] out out tensor, padding set each time for now
 * @note if out is initialized tensor, setting padding is skipped.
 */
static void im2col(const Tensor &in, const TensorDim &kdim,
                   const std::array<unsigned int, CONV2D_DIM> &padding,
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

  unsigned int ph = padding[0];
  unsigned int pw = padding[1];

  unsigned int channel = in.channel();
  int in_height = in.height();
  int in_width = in.width();
  unsigned int height = in_height + ph * 2;
  unsigned int width = in_width + pw * 2;
  unsigned int k_height = kdim.height();
  unsigned int k_width = kdim.width();

  /// effective kernel height considering dilation
  unsigned int eff_k_height = (k_height - 1) * dilation[0] + 1;
  /// effective kernel width considering dilation
  unsigned int eff_k_width = (k_width - 1) * dilation[1] + 1;

  [[maybe_unused]] unsigned int out_height =
    (height - eff_k_height) / mstride[0] + 1;
  unsigned int out_width = (width - eff_k_width) / mstride[1] + 1;

  float *out_data = out.getData();

  int h_stride_end = height - eff_k_height - ph;
  int w_stride_end = width - eff_k_width - pw;

  /// get a patch, size of kernel
  /// hs is height_strided, ws is width_strided
  unsigned int owidth = out.width();
  unsigned int base_im_w = 0;
  for (int hs = -ph; hs <= h_stride_end; hs += mstride[0]) {
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
        for (int ws = -pw; ws <= w_stride_end; ws += mstride[1]) {
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

enum ConvParams { weight, bias, im2col_result, col2im_result };

void Conv2DLayer::finalize(InitLayerContext &context) {
  if (context.getNumInputs() != 1) {
    throw std::invalid_argument("Convolution layer takes only one input");
  }

  const TensorDim &in_dim = context.getInputDimensions()[0];

  TensorDim dim =
    TensorDim(filter_size, in_dim.channel(), kernel_size[0], kernel_size[1]);
  TensorDim bias_dim = TensorDim(1, filter_size, 1, 1);

  wt_idx[ConvParams::weight] =
    context.requestWeight(dim, weight_initializer, weight_regularizer,
                          weight_regularizer_constant, "Conv2d:filter", true);
  wt_idx[ConvParams::bias] =
    context.requestWeight(bias_dim, bias_initializer, WeightRegularizer::NONE,
                          1.0f, "Conv2d:bias", true);

  /// we don't have same padding for now but later, same padding don't apply
  /// when kernel size is even in current implementation (we need to handle
  /// assymetric padding)

  // this output_dim must be the same with dimension of hidden
  TensorDim out_dim;
  out_dim.batch(in_dim.batch());
  out_dim.channel(filter_size);
  out_dim.height(
    (in_dim.height() - kernel_size[0] + 2 * padding[0]) / stride[0] + 1);
  out_dim.width((in_dim.width() - kernel_size[1] + 2 * padding[1]) / stride[1] +
                1);
  context.setOutputDimensions({out_dim});

  unsigned int eff_in_height = in_dim.height() + padding[0] * 2;
  unsigned int eff_in_width = in_dim.width() + padding[1] * 2;

  if (eff_in_height < kernel_size[0] || eff_in_width < kernel_size[1]) {
    throw std::invalid_argument(
      "Failed to initialize: in size + padding is smaller than effective "
      "kernel");
  }

  unsigned int IM = std::numeric_limits<int>::max();

  if (eff_in_height - padding[0] - kernel_size[0] > IM ||
      eff_in_width - padding[1] - kernel_size[1] > IM) {
    throw std::invalid_argument(
      "Failed to initialize: Calculated patch end is over int max");
  }

  wt_idx[ConvParams::im2col_result] = context.requestTensor(
    calcIm2ColOutputDim(in_dim, dim, padding, stride, {1, 1}), "Conv2d:im2col",
    false, ITERATION_LIFESPAN);
  wt_idx[ConvParams::col2im_result] =
    context.requestTensor(calcCol2ImOutputDim(out_dim, dim), "Conv2d:col2im",
                          false, BACKWARD_FUNC_LIFESPAN);
}

void Conv2DLayer::forwarding(RunLayerContext &context, bool training) {
  int status = ML_ERROR_NONE;

  Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);
  Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);

  Tensor &filter_kernel = context.getWeight(wt_idx[ConvParams::weight]);
  Tensor &bias_kernel = context.getWeight(wt_idx[ConvParams::bias]);

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
  const TensorDim &in_dim = input_.getDim();
  const TensorDim &out_dim = hidden_.getDim();
  const TensorDim &filter_dim = filter_kernel.getDim();
  TensorDim filter_dim_squeezed{filter_kernel.batch(),
                                filter_kernel.getDim().getFeatureLen()};

  filter_kernel.reshape(filter_dim_squeezed);

  Tensor &im2col_result = context.getTensor(wt_idx[ConvParams::im2col_result]);
  /**
   * @todo im2col_result lifespan can be epoch and then setZero can be done
   * just once at the start of training then every iteration
   *
   * @todo even better, allocate in_sub with pad, and set stride for in_sub
   * appropriately
   */
  /**
   * Below sets the pad area values to zero
   * it is faster to do this way than seting selective area to zero
   */
  im2col_result.setZero();
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
}

void Conv2DLayer::calcDerivative(RunLayerContext &context) {
  Tensor &derivative = context.getIncomingDerivative(SINGLE_INOUT_IDX);
  Tensor &input_derivative = context.getOutgoingDerivative(SINGLE_INOUT_IDX);
  Tensor &filter_kernel = context.getWeight(wt_idx[ConvParams::weight]);

  TensorDim filter_dim = filter_kernel.getDim();
  TensorDim filter_dim_squeezed{filter_kernel.batch(),
                                filter_kernel.getDim().getFeatureLen()};

  filter_kernel.reshape(filter_dim_squeezed);

  /// for each batch
  /// filter_kernel^T X derivaitive  -> column matrix
  /// col2im(column matrix) to reconstruct the original image
  Tensor &col2im_result = context.getTensor(wt_idx[ConvParams::col2im_result]);
  for (unsigned int b = 0; b < derivative.batch(); ++b) {
    Tensor deriv_sub = derivative.getBatchSlice(b, 1);
    Tensor in_deriv_sub = input_derivative.getBatchSlice(b, 1);
    deriv_sub.reshape({filter_size, derivative.width() * derivative.height()});

    filter_kernel.dot(deriv_sub, col2im_result, true, false);
    col2im(col2im_result, filter_dim, padding, stride, {1, 1}, in_deriv_sub);
  }

  filter_kernel.reshape(filter_dim);
}

void Conv2DLayer::calcGradient(RunLayerContext &context) {
  Tensor &derivative = context.getIncomingDerivative(SINGLE_INOUT_IDX);
  Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);

  Tensor &delK = context.getWeightGrad(wt_idx[ConvParams::weight]);
  Tensor &delBias = context.getWeightGrad(wt_idx[ConvParams::bias]);
  delK.setZero();

  TensorDim filter_dim = delK.getDim();
  TensorDim filter_dim_squeezed{filter_dim.batch(), filter_dim.getFeatureLen()};

  delK.reshape(filter_dim_squeezed);

  /**
   * no need to set zero for im2col_result, as its lifespan is ITERATION,
   * so its zero padded values will still be zero
   */
  Tensor &im2col_result = context.getTensor(wt_idx[ConvParams::im2col_result]);
  TensorDim out_dim_squeezed{filter_size,
                             derivative.width() * derivative.height()};

  /// input -(im2col)-> column_matrix -> filter x (column_matrix) = output
  /// so delK = dy x column_matrix ^ T;
  for (unsigned int b = 0; b < input_.batch(); ++b) {
    Tensor deriv_sub = derivative.getBatchSlice(b, 1);
    deriv_sub.reshape(out_dim_squeezed);

    Tensor in_sub = input_.getBatchSlice(b, 1);

    /**
     * @todo this result can be cached from the forward iteration at the
     * expense of memory. In this case, memory of im2col_result must be saved
     * for the whole batch. try this while benchmarking.
     */
    im2col(in_sub, filter_dim, padding, stride, {1, 1}, im2col_result);
    deriv_sub.dot(im2col_result, delK, false, false, b == 0 ? 0 : 1);
  }

  delK.reshape(filter_dim);
  delBias = derivative.sum({0, 2, 3});
}

void Conv2DLayer::setProperty(const std::vector<std::string> &values) {
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

void Conv2DLayer::setProperty(const std::string &type_str,
                              const std::string &value) {
  using PropertyType = LayerV1::PropertyType;
  int status = ML_ERROR_NONE;
  LayerV1::PropertyType type =
    static_cast<LayerV1::PropertyType>(parseLayerProperty(type_str));

  switch (type) {
  case PropertyType::filters: {
    status = setUint(filter_size, value);
    throw_status(status);
  } break;
  case PropertyType::kernel_size:
    status = getValues(CONV2D_DIM, value, (int *)(kernel_size.data()));
    throw_status(status);
    if (kernel_size[0] == 0 || kernel_size[1] == 0) {
      throw std::invalid_argument(
        "[Conv2DLayer] kernel_size must be greater than 0");
    }
    break;
  case PropertyType::stride:
    status = getValues(CONV2D_DIM, value, (int *)(stride.data()));
    throw_status(status);
    if (stride[0] == 0 || stride[1] == 0) {
      throw std::invalid_argument(
        "[Conv2DLayer] stride must be greater than 0");
    }
    break;
  case PropertyType::padding:
    status = getValues(CONV2D_DIM, value, (int *)(padding.data()));
    throw_status(status);
    break;
  default:
    LayerImpl::setProperty(type_str, value);
    break;
  }
}

} /* namespace nntrainer */
