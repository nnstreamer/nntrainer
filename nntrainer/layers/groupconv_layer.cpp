// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Sehyeon Park <shlion@snu.ac.kr>
 *
 * @file   groupconv_layer.cpp
 * @date   28 April 2025
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Sehyeon Park <shlion@snu.ac.kr>
 * @bug    No known bugs except for NYI items
 * @brief  This is Group Convolution Layer Class for Neural Network, Based on
 * conv2d layer.
 *
 */
#include <algorithm>
#include <cstring>
#include <limits>
#include <string>

#include <cpu_backend.h>
#include <groupconv_layer.h>
#include <layer_context.h>
#include <lazy_tensor.h>
#include <nntr_threads.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <profiler.h>
#include <tensor_dim.h>
#include <thread>
#include <util_func.h>

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

namespace {

static TensorDim calcCol2ImOutputDim(const TensorDim &out,
                                     const TensorDim &kdim) {

  return TensorDim({kdim.getFeatureLen(), out.width() * out.height()},
                   out.getTensorType());
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
                   const std::array<unsigned, GROUPCONV_DIM * 2> &padding,
                   const std::array<props::Stride, GROUPCONV_DIM> &mstride,
                   const std::array<props::Dilation, GROUPCONV_DIM> &dilation,
                   Tensor &image) {

  auto pt = padding[0];
  auto pb = padding[1];
  auto pl = padding[2];
  auto pr = padding[3];

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

  int h_stride_end = im_eff_height - eff_k_height - pt;
  int w_stride_end = im_eff_width - eff_k_width - pl;

  /** @todo We need to implement way to use this kind of function to work inside
   * of Tensor. Then we could remove to access the getData or getValue which has
   * dependecy of data type. This is from conv2d layer, and work in a same way
   * that of reconstructing image data.
   */
  auto apply_data = [&]<typename T>(T *val) {
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

              val = image.getAddress<T>(0, c, h, w);
              *val += col_matrix.getValue<T>(0, 0, col_h, col_w);
              col_h++;
            }
          }
        }
        col_w++;
      }
    }
  };

  if (image.getDataType() == nntrainer::Tdatatype::FP32) {
    float val;
    apply_data(&val);
  }
#ifdef ENABLE_FP16
  else if (image.getDataType() == nntrainer::Tdatatype::FP16) {
    _FP16 val;
    apply_data(&val);
  }
#endif
  else {
    throw std::runtime_error("Not supported datatype");
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
 * @param[out] out out tensor, padding set each time for now
 * @note if out is initialized tensor, setting padding is skipped.
 */

static void im2col(const Tensor &in, const TensorDim &kdim,
                   const std::array<unsigned int, GROUPCONV_DIM * 2> &padding,
                   const std::array<props::Stride, GROUPCONV_DIM> &mstride,
                   const std::array<props::Dilation, GROUPCONV_DIM> &dilation,
                   Tensor &out) {

  auto pt = padding[0];
  auto pb = padding[1];
  auto pl = padding[2];
  auto pr = padding[3];

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

  out.reshape(
    TensorDim({out_height * out_width, in.channel() * k_height * k_width},
              in.getTensorType()));
  // float *out_data = out.getData();

  auto apply_data = [&]<typename T>(T *out_data) {
    int h_stride_end = height - eff_k_height - pt;
    int w_stride_end = width - eff_k_width - pl;

    /// get a patch, size of kernel
    /// hs is height_strided, ws is width_strided
    unsigned int owidth = out.width();
    unsigned int base_im_w = 0;
    for (int hs = -pt; hs <= h_stride_end; hs += mstride[0]) {
      unsigned int base_im_h = 0;
      int patch_height_end = eff_k_height + hs;
      /// map the patch to a single line looping through channel
      // We need to optimize this padding & copy. May be use multi threads, or
      // SIMD
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
              out_data[im_w * owidth + im_h] = in.getValue<T>(0, c, h, w);
              im_h++;
            }
            im_w++;
          }
          base_im_h += k_width;
        }
      }
      base_im_w += out_width;
    }
  };

  if (out.getDataType() == nntrainer::Tdatatype::FP32) {
    float *out_data = out.getData<float>();
    apply_data(out_data);
  }
#ifdef ENABLE_FP16
  else if (out.getDataType() == nntrainer::Tdatatype::FP16) {
    _FP16 *out_data = out.getData<_FP16>();
    apply_data(out_data);
  }
#endif
  else {
    throw std::runtime_error("Not supported datatype");
  }
}
} // unnamed namespace

enum ConvParams { weight, bias };

GroupConvLayer::GroupConvLayer(
  const std::array<unsigned int, GROUPCONV_DIM * 2> &padding_) :
  LayerImpl(),
  padding(padding_),
  conv_props(
    props::FilterSize(), std::array<props::KernelSize, GROUPCONV_DIM>(),
    std::array<props::Stride, GROUPCONV_DIM>(), props::Padding2D(),
    std::array<props::Dilation, GROUPCONV_DIM>(), props::SplitNumber()) {
  wt_idx.fill(std::numeric_limits<unsigned>::max());
}

void GroupConvLayer::finalize(InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "Convolution layer takes only one input";

  const TensorDim &in_dim = context.getInputDimensions()[0];

  // If SplitNumber is empty, default is the number of channel.
  if (std::get<props::SplitNumber>(conv_props).empty()) {
    std::get<props::SplitNumber>(conv_props).set(in_dim.channel());
  }

  auto &weight_regularizer =
    std::get<props::WeightRegularizer>(*layer_impl_props);
  auto &weight_regularizer_constant =
    std::get<props::WeightRegularizerConstant>(*layer_impl_props);
  auto &weight_initializer =
    std::get<props::WeightInitializer>(*layer_impl_props);
  auto &weight_decay = std::get<props::WeightDecay>(*layer_impl_props);
  auto &bias_decay = std::get<props::BiasDecay>(*layer_impl_props);
  auto &bias_initializer = std::get<props::BiasInitializer>(*layer_impl_props);
  auto &disable_bias = std::get<props::DisableBias>(*layer_impl_props);

  unsigned int filter_size = std::get<props::FilterSize>(conv_props);
  auto &kernel_size =
    std::get<std::array<props::KernelSize, GROUPCONV_DIM>>(conv_props);
  auto &stride = std::get<std::array<props::Stride, GROUPCONV_DIM>>(conv_props);
  auto &dilation =
    std::get<std::array<props::Dilation, GROUPCONV_DIM>>(conv_props);
  unsigned int &group_n = std::get<props::SplitNumber>(conv_props);

  auto in_t_type = in_dim.getTensorType();
  in_t_type.data_type = context.getWeightDataType();

  // Assert the number of input channels is divisible by the number of groups.
  NNTR_THROW_IF(in_dim.channel() % group_n != 0, std::invalid_argument)
    << "Failed to initialize: Input channels must be divisible by number of "
       "groups.";

  TensorDim kernel_dim = TensorDim(filter_size, in_dim.channel() / group_n,
                                   kernel_size[0], kernel_size[1], in_t_type);

  TensorDim bias_dim = TensorDim(1, filter_size, 1, 1, in_t_type);

  padding = std::get<props::Padding2D>(conv_props)
              .compute(in_dim, kernel_dim, {stride[0], stride[1]},
                       {dilation[0], dilation[1]});

  wt_idx[ConvParams::weight] = context.requestWeight(
    kernel_dim, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, "filter", true, 0);

  if (disable_bias.empty() || disable_bias.get() == false) {
    wt_idx[ConvParams::bias] =
      context.requestWeight(bias_dim, bias_initializer, WeightRegularizer::NONE,
                            1.0f, bias_decay, "bias", true, 0);
  }

  // this output_dim must be the same with dimension of hidden
  unsigned int eff_in_height = in_dim.height() + padding[0] + padding[1];
  unsigned int eff_in_width = in_dim.width() + padding[2] + padding[3];

  unsigned int eff_k_height = (kernel_size[0] - 1) * dilation[0] + 1;
  unsigned int eff_k_width = (kernel_size[1] - 1) * dilation[1] + 1;

  TensorDim out_dim;
  out_dim.batch(in_dim.batch());
  out_dim.channel(filter_size);
  out_dim.height((eff_in_height - eff_k_height) / stride[0] + 1);
  out_dim.width((eff_in_width - eff_k_width) / stride[1] + 1);

  out_dim.setTensorType(in_dim.getTensorType());

  context.setOutputDimensions({out_dim});

  NNTR_THROW_IF(eff_in_height < kernel_size[0] || eff_in_width < kernel_size[1],
                std::invalid_argument)
    << "Failed to initialize: in size + padding is smaller than effective "
       "kernel";

  unsigned int IM = std::numeric_limits<int>::max();

  NNTR_THROW_IF(eff_in_height - padding[0] - kernel_size[0] > IM ||
                  eff_in_width - padding[2] - kernel_size[1] > IM,
                std::invalid_argument)
    << "Failed to initialize: Calculated patch end is over int max";
}

void GroupConvLayer::forwarding(RunLayerContext &context, bool training) {
  int status = ML_ERROR_NONE;

  unsigned int filter_size = std::get<props::FilterSize>(conv_props);
  auto &stride = std::get<std::array<props::Stride, GROUPCONV_DIM>>(conv_props);
  auto &dilation =
    std::get<std::array<props::Dilation, GROUPCONV_DIM>>(conv_props);

  Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);
  Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);
  Tensor &filter_kernel = context.getWeight(wt_idx[ConvParams::weight]);
  unsigned int &group_n = std::get<props::SplitNumber>(conv_props);

  /** Calculate Group Convolution
   *
   * This is the 2D Matrix Shape [ height ] x [ width ]
   *   . Height : filter_size
   *   . Width  : Input Channel / group_n * Kernel_size[0] * Kernel_size[1]
   *
   *                              imKernel
   *                        +------|------|------+
   *                        |------|------|------|
   * [filter_size (height)] |------|------|------|
   *                        |------|------|------|
   *                        +------|------|------+
   *                [Input Channel / group_n * Kernel_size[0]
   *                       * Kernel_size[1] (width)]
   *
   *
   * After im2Col with channel_mode true (in : input)
   *
   * This is the 2D Matrix Shape [ height ] x [ width ]
   *   . Height : Input Channel / group_n * Kernel_size[0] * Kernel_size[1]
   *   . Width  : output_dim.height * output_dim.width
   *
   *                      +-|-|-|-|      |-|-|-|-+
   *   [Input Channel     | | | | |      | | | | |
   *   / group_n          |_|_|_|_|      |_|_|_|_|
   *   * Kernel_size[0]   | | | | | .... | | | | |
   *   * Kernel_size[1]   |_|_|_|_|      |_|_|_|_|
   *    (height)]         | | | | |      | | | | |
   *                      +_|_|_|_|      |_|_|_|_+
   *                     [ output_dim.height
   *                      * output_dim.width (width) ]
   *
   * Output Dimension
   *   -> [Channel / group_n ( = filter_size  / group_n = output_dim.channel /
   * group_n)] x [output_dim.height x output_dim.width]
   */
  const TensorDim &in_dim = input_.getDim();
  const TensorDim &out_dim = hidden_.getDim();
  const TensorDim &filter_dim = filter_kernel.getDim();

  // Assert the number of channels is divisible by the number of groups.
  NNTR_THROW_IF(in_dim.channel() % group_n != 0, std::invalid_argument)
    << "Failed to forwarding: Input channels must be divisible by number of "
       "groups.";
  NNTR_THROW_IF(out_dim.channel() % group_n != 0, std::invalid_argument)
    << "Failed to forwarding: Output channels must be divisible by number of "
       "groups.";

  // for Group convolution
  const size_t input_channels_per_group = in_dim.channel() / group_n;
  const size_t output_channels_per_group = out_dim.channel() / group_n;

  TensorDim filter_dim_squeezed{filter_kernel.batch(),
                                filter_kernel.getDim().getFeatureLen()};

  filter_dim_squeezed.setTensorType(filter_kernel.getTensorType());

  filter_kernel.reshape(filter_dim_squeezed);

  /**
   * Below sets the pad area values to zero
   * it is faster to do this way than seting selective area to zero
   */
  auto forwarding_job = [&](unsigned int s, unsigned int e, unsigned int pid,
                            void *user_data) {
    Tensor result = Tensor(calcCol2ImOutputDim(out_dim, filter_dim));
    result.setZero();
    for (unsigned int b = s; b < e; ++b) {
      Tensor out = hidden_.getBatchSlice(b, 1);
      out.reshape({filter_size, out_dim.width() * out_dim.height()});
      Tensor in_sub = input_.getBatchSlice(b, 1);

      size_t input_channel_stride = in_dim.height() * in_dim.width();
      size_t output_channel_stride = out_dim.height() * out_dim.width();
      size_t filter_stride =
        input_channels_per_group * filter_dim.height() * filter_dim.width();

      // im2col and dot production group by group.
      for (unsigned int g = 0; g < group_n; ++g) {
        size_t offset_in = input_channels_per_group * input_channel_stride * g;
        size_t offset_out =
          output_channels_per_group * output_channel_stride * g;
        size_t offset_filter = output_channels_per_group * filter_stride * g;

        Tensor in_sub_group = in_sub.getSharedDataTensor(
          {1, input_channels_per_group, in_dim.height(), in_dim.width()},
          offset_in);

        Tensor out_sub_group = out.getSharedDataTensor(
          {output_channels_per_group, out_dim.height() * out_dim.width()},
          offset_out);

        Tensor filter_kernel_sub_group = filter_kernel.getSharedDataTensor(
          {output_channels_per_group,
           input_channels_per_group * filter_dim.height() * filter_dim.width()},
          offset_filter);

        im2col(in_sub_group, filter_dim, padding, stride, dilation, result);
        filter_kernel_sub_group.dot(result, out_sub_group, false, true);
      }

      /**
       * Below is the original code from conv2d_layer.
       * im2col(in_sub, filter_dim, padding, stride, dilation, result);
       * // filter kernel is (K, CRS), result is (CRS, OH*OW)
       * filter_kernel.dot(result, out, false, true);
       */
    }
    result.deallocate();
  };

  auto workers = ParallelBatch(forwarding_job, in_dim.batch(), nullptr);

  if (workers.getNumWorkers() > 1) {
    workers.run();
  } else {
    forwarding_job(0, in_dim.batch(), 0, nullptr);
  }

  filter_kernel.reshape(filter_dim);
  if (auto &disable_bias = std::get<props::DisableBias>(*layer_impl_props);
      disable_bias.empty() || disable_bias.get() == false) {
    Tensor &bias_kernel = context.getWeight(wt_idx[ConvParams::bias]);
    status = hidden_.add_i(bias_kernel);
    if (status != ML_ERROR_NONE) {
      throw std::invalid_argument("[GroupConv] adding bias failed");
    }
  }
}

void GroupConvLayer::calcDerivative(RunLayerContext &context) {
  const unsigned int filter_size = std::get<props::FilterSize>(conv_props);
  const auto &stride =
    std::get<std::array<props::Stride, GROUPCONV_DIM>>(conv_props);
  const auto &dilation =
    std::get<std::array<props::Dilation, GROUPCONV_DIM>>(conv_props);
  unsigned int &group_n = std::get<props::SplitNumber>(conv_props);

  const Tensor &derivative = context.getIncomingDerivative(SINGLE_INOUT_IDX);
  Tensor &input_derivative = context.getOutgoingDerivative(SINGLE_INOUT_IDX);
  Tensor &filter_kernel = context.getWeight(wt_idx[ConvParams::weight]);

  const TensorDim &in_dim = input_derivative.getDim();
  const TensorDim &out_dim = derivative.getDim();
  const TensorDim &filter_dim = filter_kernel.getDim();

  // Assert the number of channels is divisible by the number of groups.
  NNTR_THROW_IF(in_dim.channel() % group_n != 0, std::invalid_argument)
    << "Failed to calcDerivative: Input channels must be divisible by number "
       "of groups.";
  NNTR_THROW_IF(out_dim.channel() % group_n != 0, std::invalid_argument)
    << "Failed to calcDerivative: Output channels must be divisible by number "
       "of groups.";

  const size_t input_channels_per_group = in_dim.channel() / group_n;
  const size_t output_channels_per_group = out_dim.channel() / group_n;

  TensorDim filter_dim_squeezed{filter_kernel.batch(),
                                filter_dim.getFeatureLen()};
  filter_kernel.reshape(filter_dim_squeezed);

  /// for each batch, each group
  /// filter_kernel^T X derivative -> column matrix
  /// col2im(column matrix) to reconstruct the original image

  auto compute_derivative = [&](unsigned int s, unsigned int e,
                                unsigned int pid, void *user_data) {
    Tensor result = Tensor(calcCol2ImOutputDim(out_dim, filter_dim));

    for (unsigned int b = s; b < e; ++b) {
      Tensor deriv_batch = derivative.getBatchSlice(b, 1);
      Tensor in_deriv_batch = input_derivative.getBatchSlice(b, 1);

      for (unsigned int g = 0; g < group_n; ++g) {
        Tensor deriv_sub = deriv_batch.getSharedDataTensor(
          {1, output_channels_per_group, out_dim.height(), out_dim.width()},
          output_channels_per_group * out_dim.height() * out_dim.width() * g);

        deriv_sub.reshape(
          {output_channels_per_group, out_dim.width() * out_dim.height()});

        Tensor in_deriv_sub = in_deriv_batch.getSharedDataTensor(
          {1, input_channels_per_group, in_dim.height(), in_dim.width()},
          input_channels_per_group * in_dim.height() * in_dim.width() * g);

        Tensor filter_kernel_sub = filter_kernel.getSharedDataTensor(
          {output_channels_per_group,
           input_channels_per_group * filter_dim.height() * filter_dim.width()},
          output_channels_per_group * g);

        filter_kernel_sub.dot(deriv_sub, result, true, false);
        col2im(result, filter_dim, padding, stride, dilation, in_deriv_sub);
      }
    }
    result.deallocate();
  };

  auto workers = ParallelBatch(compute_derivative, derivative.batch(), nullptr);

  if (workers.getNumWorkers() > 1) {
    workers.run();
  } else {
    compute_derivative(0, derivative.batch(), 0, nullptr);
  }

  filter_kernel.reshape(filter_dim);
}

void GroupConvLayer::calcGradient(RunLayerContext &context) {
  unsigned int filter_size = std::get<props::FilterSize>(conv_props);
  auto &stride = std::get<std::array<props::Stride, GROUPCONV_DIM>>(conv_props);
  auto &dilation =
    std::get<std::array<props::Dilation, GROUPCONV_DIM>>(conv_props);
  unsigned int &group_n = std::get<props::SplitNumber>(conv_props);

  const Tensor &derivative = context.getIncomingDerivative(SINGLE_INOUT_IDX);
  Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);

  Tensor &delK = context.getWeightGrad(wt_idx[ConvParams::weight]);
  delK.setZero();

  TensorDim filter_dim = delK.getDim();
  TensorDim filter_dim_squeezed{filter_dim.batch(), filter_dim.getFeatureLen()};

  delK.reshape(filter_dim_squeezed);

  size_t input_c_per_group = input_.channel() / group_n;
  size_t output_c_per_group = derivative.channel() / group_n;
  size_t filter_offset_per_group =
    output_c_per_group * filter_dim.getFeatureLen();

  /**
   * no need to set zero for im2col_result, as its lifespan is ITERATION,
   * so its zero padded values will still be zero
   */

  TensorDim out_dim_squeezed{output_c_per_group,
                             derivative.width() * derivative.height()};
  auto workers = ParallelBatch(input_.batch());

  if (workers.getNumWorkers() > 1) {
    TensorDim delK_ext = filter_dim_squeezed;
    delK_ext.batch(input_.batch());

    Tensor delK_par = Tensor(delK_ext);
    delK_par.setZero();

    auto calc_grad_job = [&](unsigned int s, unsigned int e, unsigned int pid,
                             void *user_data) {
      Tensor result =
        Tensor(calcCol2ImOutputDim(derivative.getDim(), filter_dim));
      result.setZero();
      for (unsigned int b = s; b < e; ++b) {
        for (unsigned int g = 0; g < group_n; ++g) {
          Tensor deriv_sub = derivative.getBatchSlice(b, 1).getSharedDataTensor(
            {output_c_per_group, derivative.height(), derivative.width()},
            output_c_per_group * derivative.height() * derivative.width() * g);
          // same as g * output_c_per_group *
          // derivative.getDim().getFeatureLen()

          Tensor delK_sub = delK_par.getBatchSlice(b, 1).getSharedDataTensor(
            {output_c_per_group, filter_dim.getFeatureLen()},
            filter_offset_per_group * g);

          deriv_sub.reshape(out_dim_squeezed);

          Tensor in_sub = input_.getBatchSlice(b, 1).getSharedDataTensor(
            {input_c_per_group, input_.height(), input_.width()},
            input_c_per_group * input_.height() * input_.width() * g);
          // same as g * output_c_per_group * input_.getDim().getFeatureLen()

          im2col(in_sub, filter_dim, padding, stride, dilation, result);
          deriv_sub.dot(result, delK_sub, false, false);
        }
      }
      result.deallocate();
    };

    workers.setCallback(calc_grad_job, nullptr);

    workers.run();

    for (unsigned int b = 0; b < input_.batch(); ++b) {
      Tensor delK_sub = delK_par.getBatchSlice(b, 1);
      delK.add_i(delK_sub);
    }

  } else {
    Tensor result =
      Tensor(calcCol2ImOutputDim(derivative.getDim(), filter_dim));
    result.setZero();

    for (unsigned int b = 0; b < input_.batch(); ++b) {
      for (unsigned int g = 0; g < group_n; ++g) {
        Tensor deriv_sub = derivative.getBatchSlice(b, 1).getSharedDataTensor(
          {output_c_per_group, derivative.height(), derivative.width()},
          g * output_c_per_group * derivative.height() * derivative.width());
        // same as g * output_c_per_group * derivative.getDim().getFeatureLen()

        deriv_sub.reshape(out_dim_squeezed);

        Tensor in_sub = input_.getBatchSlice(b, 1).getSharedDataTensor(
          {input_c_per_group, input_.height(), input_.width()},
          g * input_c_per_group * input_.height() * input_.width());
        // same as g * output_c_per_group * input_.getDim().getFeatureLen()

        im2col(in_sub, filter_dim, padding, stride, dilation, result);

        Tensor delK_sub = delK.getSharedDataTensor(
          {output_c_per_group, filter_dim.getFeatureLen()},
          g * filter_offset_per_group);

        deriv_sub.dot(result, delK_sub, false, false, b == 0 ? 0 : 1);
      }
    }
    result.deallocate();
  }
  delK.reshape(filter_dim);
  if (auto &disable_bias = std::get<props::DisableBias>(*layer_impl_props);
      disable_bias.empty() || disable_bias.get() == false) {
    Tensor &delBias = context.getWeightGrad(wt_idx[ConvParams::bias]);
    delBias.setZero();
    derivative.sum({0, 2, 3}, delBias);
  }
}

void GroupConvLayer::exportTo(Exporter &exporter,
                              const ml::train::ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
  exporter.saveResult(conv_props, method, this);
}

void GroupConvLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, conv_props);
  LayerImpl::setProperty(remain_props);
}

} /* namespace nntrainer */
