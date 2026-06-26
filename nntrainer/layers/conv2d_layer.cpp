// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   conv2d_layer.h
 * @date   02 June 2020
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is Convolution Layer Class for Neural Network
 *
 */
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#include <conv2d_layer.h>
#include <cpu_backend.h>
#include <layer_context.h>
#include <lazy_tensor.h>
#include <nntr_threads.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <profiler.h>
#include <tensor_dim.h>
#include <thread>
#include <thread_manager.h>
#include <util_func.h>

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

namespace {

/**
 * @brief In-place SiLU / swish (x * sigmoid(x)) over a contiguous buffer.
 * @details Fuses what would otherwise be a separate Activation layer (a full
 * extra read+write pass over the conv output) into the conv epilogue. The
 * sigmoid is evaluated in fp32 and cast back to T so an FP16 activation graph
 * does not lose precision (or overflow exp) in the half domain. Called after
 * all conv compute completes, so it is never nested inside another
 * parallel_for.
 *
 * @note ThreadManager::parallel_for invokes its callback through a type-erased
 * std::function PER INDEX, so iterating it at element granularity would pay a
 * non-inlinable call per element (measured ~3x slower than a serial inlined
 * loop). Instead we parallelize over a handful of contiguous chunks (one per
 * compute thread) and run a tight, fully-inlined inner loop inside each — the
 * std::function is then hit only ~nthreads times while the exp stays inlined.
 */
template <typename T>
static inline void convApplySwishInplace(T *data, size_t n) {
  auto &tm = ThreadManager::Global();
  const size_t nthreads = std::max<size_t>(1, tm.getComputeThreadCount());
  const size_t chunk = (n + nthreads - 1) / nthreads;
  tm.parallel_for(0, nthreads, [&](size_t t) {
    const size_t start = t * chunk;
    if (start >= n)
      return;
    const size_t end = std::min(start + chunk, n);
    for (size_t i = start; i < end; ++i) {
      const float x = static_cast<float>(data[i]);
      data[i] = static_cast<T>(x / (1.0f + std::exp(-x)));
    }
  });
}

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
                   const std::array<unsigned, 4> &padding,
                   const std::array<props::Stride, CONV2D_DIM> &mstride,
                   const std::array<props::Dilation, CONV2D_DIM> &dilation,
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
   * dependecy of data type.
   */
  auto apply_data = [&](auto *val) {
    using T = std::decay_t<decltype(*val)>;
    unsigned col_w = 0;
    for (int hs = -(int)pt; hs <= h_stride_end; hs += hstride) {
      for (int ws = -(int)pl; ws <= w_stride_end; ws += wstride) {
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
                   const std::array<unsigned int, 4> &padding,
                   const std::array<props::Stride, CONV2D_DIM> &mstride,
                   const std::array<props::Dilation, CONV2D_DIM> &dilation,
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
    const int hstride = mstride[0];

    /// Raw contiguous-NCHW input base + inner strides. `in` is a (batch-sliced)
    /// contiguous NCHW tensor, so element (0,c,h,w) lives at
    /// in_base + c*inHW + h*inW + w. Hoisting these out of the inner loop turns
    /// the old per-element in.getValue() -- which re-fetched the data pointer
    /// (getData<T>()) and recomputed a 4-D linear offset (getIndex(): a format
    /// branch + 4 muls + 3 adds) plus a per-element padding branch -- into one
    /// contiguous run copy. im2col is pure data movement; the previous form ran
    /// far below memory bandwidth because that overhead dominated the 4-byte
    /// move. Padding columns are left untouched (the caller zeroes the buffer
    /// once before im2col), so the fast path only writes the valid span.
    const T *in_base = in.getData<T>();
    const size_t inW = (size_t)in_width;
    const size_t inHW = (size_t)in_height * (size_t)in_width;
    const bool unit_dil =
      ((unsigned int)dilation[0] == 1 && (unsigned int)dilation[1] == 1);

    /// Each output row (oh) writes a disjoint band of `out_width` columns
    /// (rows [oh*out_width, (oh+1)*out_width) of the [OH*OW, CRS] matrix), so
    /// the per-row work is independent and bit-identical when parallelized.
    /// hs and base_im_w are derived from oh directly (no sequential carry).
    auto fill_row = [&](size_t oh) {
      int hs = -(int)pt + (int)oh * hstride;
      unsigned int base_im_w = (unsigned int)oh * out_width;
      unsigned int base_im_h = 0;
      int patch_height_end = eff_k_height + hs;
      /// map the patch to a single line looping through channel
      for (unsigned int c = 0; c < channel; ++c) {
        for (int h = hs; h < patch_height_end; h += dilation[0]) {
          if (h < 0 || in_height <= h) {
            base_im_h += k_width;
            continue;
          }

          if (unit_dil) {
            /// Fast path (dilation == 1): for each output column position the
            /// kernel-width window maps a contiguous source run
            /// in_row[w_lo,w_hi) to a contiguous dest run; copy it in one
            /// memcpy.
            const T *in_row = in_base + (size_t)c * inHW + (size_t)h * inW;
            unsigned int im_w = base_im_w;
            for (int ws = -(int)pl; ws <= w_stride_end; ws += mstride[1]) {
              int w_lo = ws < 0 ? 0 : ws;
              int w_hi = ws + (int)k_width;
              if (w_hi > in_width)
                w_hi = in_width;
              if (w_hi > w_lo) {
                T *dst =
                  out_data + (size_t)im_w * owidth + base_im_h + (w_lo - ws);
                std::memcpy(dst, in_row + w_lo,
                            (size_t)(w_hi - w_lo) * sizeof(T));
              }
              im_w++;
            }
          } else {
            /// General (dilated) path: original scalar gather, but via the
            /// hoisted base pointer (no per-element getData()/getIndex()).
            unsigned int im_w = base_im_w;
            for (int ws = -(int)pl; ws <= w_stride_end; ws += mstride[1]) {
              unsigned int im_h = base_im_h;
              int patch_width_end = eff_k_width + ws;

              for (int w = ws; w < patch_width_end; w += dilation[1]) {
                if (w < 0 || in_width <= w) {
                  im_h++;
                  continue;
                }
                out_data[(size_t)im_w * owidth + im_h] =
                  in_base[(size_t)c * inHW + (size_t)h * inW + w];
                im_h++;
              }
              im_w++;
            }
          }
          base_im_h += k_width;
        }
      }
    };

    ThreadManager::Global().parallel_for(0, out_height, fill_row);
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
} // namespace

enum ConvParams { weight, bias, im2col_scratch, qgemm_scratch };

Conv2DLayer::Conv2DLayer(
  const std::array<unsigned int, CONV2D_DIM * 2> &padding_) :
  LayerImpl(),
  padding(padding_),
  conv_props(props::FilterSize(), std::array<props::KernelSize, CONV2D_DIM>(),
             std::array<props::Stride, CONV2D_DIM>(), props::Padding2D(),
             std::array<props::Dilation, CONV2D_DIM>(), props::ConvGroups(),
             props::FusedActivation()) {
  wt_idx.fill(std::numeric_limits<unsigned>::max());
}

void Conv2DLayer::finalize(InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "Convolution layer takes only one input";

  const TensorDim &in_dim = context.getInputDimensions()[0];

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
    std::get<std::array<props::KernelSize, CONV2D_DIM>>(conv_props);
  auto &stride = std::get<std::array<props::Stride, CONV2D_DIM>>(conv_props);
  auto &dilation =
    std::get<std::array<props::Dilation, CONV2D_DIM>>(conv_props);

  auto &groups_prop = std::get<props::ConvGroups>(conv_props);
  unsigned int groups = groups_prop.empty() ? 1 : groups_prop.get();
  NNTR_THROW_IF(in_dim.channel() % groups != 0 || filter_size % groups != 0,
                std::invalid_argument)
    << "[Conv2D] input channels (" << in_dim.channel() << ") and filters ("
    << filter_size << ") must both be divisible by groups (" << groups << ")";

  auto in_t_type = in_dim.getTensorType();
  in_t_type.data_type = context.getWeightDataType();

  // A quantized (Q4_0/QINT4) 1x1 conv is computed as a matmul, so its filter is
  // stored as a [in_ch, out_ch] (K, N) weight that the quantized GEMM consumes
  // directly (no im2col-style [out_ch, CRS] squeeze). Non-quantized or larger
  // kernels keep the standard [out_ch, in_ch/groups, kh, kw] layout.
  const bool quant_matmul_filter =
    (in_t_type.data_type == nntrainer::Tdatatype::Q4_0 ||
     in_t_type.data_type == nntrainer::Tdatatype::QINT4) &&
    groups == 1;

  // Real conv kernel geometry — used for padding/output-size computation even
  // when the quantized weight is stored flattened as [CRS, out_ch].
  TensorDim real_kernel_dim(filter_size, in_dim.channel() / groups,
                            kernel_size[0], kernel_size[1], in_t_type);
  // A quantized (groups==1) conv stores its filter as a [CRS, out_ch] matmul
  // weight (CRS = in_ch*kh*kw), consumed by the quantized GEMM after im2col.
  TensorDim kernel_dim = quant_matmul_filter
                           ? TensorDim(1, 1,
                                       in_dim.channel() * kernel_size[0].get() *
                                         kernel_size[1].get(),
                                       filter_size, in_t_type)
                           : real_kernel_dim;

  // Bias is never quantized (no dequantizer for add); follow activation dtype
  // like other compute layers so a Q4_0/QINT4 weight does not force a Q4_0
  // bias.
  auto bias_t_type = in_dim.getTensorType();
  bias_t_type.data_type = context.getActivationDataType();
  TensorDim bias_dim = TensorDim(1, filter_size, 1, 1, bias_t_type);

  padding = std::get<props::Padding2D>(conv_props)
              .compute(in_dim, real_kernel_dim, {stride[0], stride[1]},
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

  // Forward scratch (groups==1 path only): the im2col column buffer and the
  // quantized-GEMM output are otherwise heap-allocated on every forwarding()
  // call. Request them once here (planned into the shared activation arena,
  // FORWARD_FUNC_LIFESPAN) and reuse — no per-forward malloc/free churn. The
  // grouped path keeps its local buffer. NOTE: the im2col buffer must still be
  // re-zeroed each forward (im2col skips padding positions and the arena is
  // reused across layers), so this saves the allocation, not the zeroing.
  wt_idx[ConvParams::im2col_scratch] = std::numeric_limits<unsigned int>::max();
  wt_idx[ConvParams::qgemm_scratch] = std::numeric_limits<unsigned int>::max();
  if (groups == 1) {
    auto scratch_type = in_dim.getTensorType();
    const unsigned int owoh = out_dim.width() * out_dim.height();
    const bool is_1x1_s1 = kernel_size[0].get() == 1 &&
                           kernel_size[1].get() == 1 && stride[0].get() == 1 &&
                           stride[1].get() == 1;
    // im2col column buffer [batch, 1, CRS, OH*OW]. Unused by the quant paths
    // that never materialize a col buffer: the 1x1 path (im2col is an identity
    // handled by an input transpose) and, where the fused backend op exists,
    // the non-1x1 path (gather is fused into the q8_0 activation packing).
    if (!(quant_matmul_filter && (is_1x1_s1 || NNTR_HAS_Q4_0_INDIRECT_CONV))) {
      TensorDim col_dim(in_dim.batch(), 1, real_kernel_dim.getFeatureLen(),
                        owoh, scratch_type);
      wt_idx[ConvParams::im2col_scratch] =
        context.requestTensor(col_dim, "im2col", Initializer::NONE, false,
                              TensorLifespan::FORWARD_FUNC_LIFESPAN);
    }
    // quantized GEMM output [batch, 1, OH*OW, out_ch] (quant path only).
    if (quant_matmul_filter) {
      TensorDim tmp_dim(in_dim.batch(), 1, owoh, filter_size, scratch_type);
      wt_idx[ConvParams::qgemm_scratch] =
        context.requestTensor(tmp_dim, "qgemm_out", Initializer::NONE, false,
                              TensorLifespan::FORWARD_FUNC_LIFESPAN);
    }
  }
}

void Conv2DLayer::forwarding(RunLayerContext &context, bool training) {
  int status = ML_ERROR_NONE;

  unsigned int filter_size = std::get<props::FilterSize>(conv_props);
  auto &stride = std::get<std::array<props::Stride, CONV2D_DIM>>(conv_props);
  auto &dilation =
    std::get<std::array<props::Dilation, CONV2D_DIM>>(conv_props);
  auto &kernel_size =
    std::get<std::array<props::KernelSize, CONV2D_DIM>>(conv_props);

  Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);
  Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);

  Tensor &filter_kernel = context.getWeight(wt_idx[ConvParams::weight]);

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
  auto &groups_prop = std::get<props::ConvGroups>(conv_props);
  unsigned int groups = groups_prop.empty() ? 1 : groups_prop.get();

  if (groups == 1) {
    // A quantized 1x1 conv stores its filter as a [in_ch, out_ch] matmul weight
    // (K, N). The quantized GEMM (dotQnK for Q4_0) takes the weight as the dot
    // *input* (activation is the receiver), so we keep that layout as-is and do
    // NOT squeeze it to [out_ch, CRS] like the FP32 path.
    const auto weight_dtype = filter_kernel.getDataType();
    const bool weight_is_quant = (weight_dtype == nntrainer::Tdatatype::Q4_0 ||
                                  weight_dtype == nntrainer::Tdatatype::QINT4);
    const unsigned int owoh = out_dim.width() * out_dim.height();

    TensorDim filter_dim_squeezed{filter_kernel.batch(),
                                  filter_kernel.getDim().getFeatureLen()};
    filter_dim_squeezed.setTensorType(filter_kernel.getTensorType());
    if (!weight_is_quant) {
      filter_kernel.reshape(filter_dim_squeezed);
    }

    /**
     * Below sets the pad area values to zero
     * it is faster to do this way than seting selective area to zero
     */
    const bool is_1x1_s1 = kernel_size[0].get() == 1 &&
                           kernel_size[1].get() == 1 && stride[0].get() == 1 &&
                           stride[1].get() == 1;
    // Pre-allocated forward scratch (requested once in finalize). The im2col
    // column buffer is used by the FP32 path and the quant non-1x1 *fallback*;
    // the quant 1x1 path (identity input transpose) and the quant non-1x1 fused
    // path (gather folded into the GEMM) need no col buffer — finalize skips
    // allocating it for those, so the condition here must match exactly.
    // Zero the column buffer once up front (im2col skips padding; the GEMM
    // output is fully overwritten so it needs no zeroing). Each batch element
    // writes a disjoint batch-slice, so this stays correct under ParallelBatch.
    const bool use_im2col_scratch =
      !(weight_is_quant && (is_1x1_s1 || NNTR_HAS_Q4_0_INDIRECT_CONV));
    Tensor *col_scratch =
      use_im2col_scratch
        ? &context.getTensor(wt_idx[ConvParams::im2col_scratch])
        : nullptr;
    Tensor *qgemm_scratch =
      weight_is_quant ? &context.getTensor(wt_idx[ConvParams::qgemm_scratch])
                      : nullptr;
    if (col_scratch != nullptr) {
      col_scratch->setZero();
    }

    auto forwarding_job = [&](unsigned int s, unsigned int e, unsigned int pid,
                              void *user_data) {
      for (unsigned int b = s; b < e; ++b) {
        Tensor out = hidden_.getBatchSlice(b, 1);
        Tensor in_sub = input_.getBatchSlice(b, 1);

        if (weight_is_quant) {
          // Quantized conv as matmul: act [OH*OW, CRS] . weight [CRS, out_ch]
          // -> [OH*OW, out_ch] -> out [out_ch, OH*OW]. CRS = in_ch*kh*kw.
          // NOTE: col must outlive `act` (act aliases col's storage); here col
          // is a view into the context-owned scratch, so its storage outlives
          // the loop iteration regardless.
          Tensor tmp = qgemm_scratch->getBatchSlice(b, 1);
          tmp.reshape(
            TensorDim(1, 1, owoh, filter_size, in_sub.getTensorType()));
          if (is_1x1_s1) {
            // 1x1 stride-1: im2col is an identity. The raw input is laid out as
            // [in_ch, OH*OW] (NCHW), so transpose to the act layout [OH*OW,
            // CRS] (CRS == in_ch here).
            in_sub.reshape({in_dim.channel(), owoh});
            Tensor act = in_sub.transpose("0:2:1");
            act.dot(filter_kernel, tmp, false, false);
          } else if (NNTR_HAS_Q4_0_INDIRECT_CONV) {
            // Fused im2col gather: the FP32 col buffer is never materialized.
            // The gather runs inside the GEMM's q8_0 activation packing,
            // directly from the NCHW input, producing rows byte-identical to
            // im2col -> bit-identical output vs the materialized path.
            ConvGatherParams geom;
            geom.in_ch = in_dim.channel();
            geom.in_h = in_dim.height();
            geom.in_w = in_dim.width();
            geom.k_h = kernel_size[0].get();
            geom.k_w = kernel_size[1].get();
            geom.pad_t = padding[0];
            geom.pad_l = padding[2];
            geom.stride_h = stride[0].get();
            geom.stride_w = stride[1].get();
            geom.dil_h = dilation[0].get();
            geom.dil_w = dilation[1].get();
            geom.out_w = out_dim.width();
            in_sub.convQ4_0Indirect(filter_kernel, tmp, geom);
          } else {
            // Fallback (no fused backend op): materialize im2col into the col
            // scratch, then the standard quant GEMM.
            // build the real kernel geometry (filter is stored as [CRS,out_ch])
            TensorDim kdim(filter_size, in_dim.channel(), kernel_size[0].get(),
                           kernel_size[1].get(), in_sub.getTensorType());
            Tensor col = col_scratch->getBatchSlice(b, 1);
            // im2col reshapes col in place to [OH*OW, CRS] (spatial-major),
            // which is ALREADY the act layout — no transpose (unlike the
            // raw-input 1x1 branch above). Transposing here gives [CRS, OH*OW]
            // and makes the GEMM emit CRS rows into the owoh-row `tmp`,
            // overflowing it whenever CRS > owoh (deep convs) -> heap
            // corruption.
            im2col(in_sub, kdim, padding, stride, dilation, col);
            col.dot(filter_kernel, tmp, false, false);
          }
          // [OH*OW, out_ch] -> [out_ch, OH*OW] written straight into the
          // (memory-planned) output. `tmp` is a separate scratch buffer and
          // `out` is a separate output view, so there is no aliasing.
          out.reshape({filter_size, owoh});
          tmp.transpose("0:2:1", out);
        } else {
          Tensor result = col_scratch->getBatchSlice(b, 1);
          out.reshape({filter_size, owoh});
          im2col(in_sub, filter_dim, padding, stride, dilation, result);
          // filter kernel is (K, CRS), result is (CRS, OH*OW)
          filter_kernel.dot(result, out, false, true);
        }
      }
    };

    auto workers = ParallelBatch(forwarding_job, in_dim.batch(), nullptr);

    if (workers.getNumWorkers() > 1) {
      workers.run();
    } else {
      forwarding_job(0, in_dim.batch(), 0, nullptr);
    }

    if (!weight_is_quant) {
      filter_kernel.reshape(filter_dim);
    }
  } else {
    // Grouped convolution: split channels into `groups` independent groups.
    const unsigned int ocg = filter_size / groups;      // out ch per group
    const unsigned int icg = in_dim.channel() / groups; // in ch per group
    const unsigned int fh = filter_dim.height(), fw = filter_dim.width();
    const unsigned int owoh = out_dim.width() * out_dim.height();
    const unsigned int ihw = in_dim.height() * in_dim.width();
    TensorDim fdim_g(ocg, icg, fh, fw, filter_dim.getTensorType());

    const bool is_true_depthwise = ocg == 1 && icg == 1;

    if (is_true_depthwise &&
        in_dim.getDataType() == nntrainer::Tdatatype::FP32) {
      // True depthwise (groups == channels): delegate to the CPU backend op so
      // the optimised kernel lives in the backend, not in the layer.
      nntrainer::getComputeOps()->depthwise_conv2d_fp32(
        input_.getData<float>(), filter_kernel.getData<float>(),
        hidden_.getData<float>(), in_dim.batch(), filter_size, in_dim.height(),
        in_dim.width(), out_dim.height(), out_dim.width(), fh, fw,
        stride[0].get(), stride[1].get(), padding[0], padding[2],
        dilation[0].get(), dilation[1].get());
#ifdef ENABLE_FP16
    } else if (is_true_depthwise &&
               in_dim.getDataType() == nntrainer::Tdatatype::FP16 &&
               hidden_.getDataType() == nntrainer::Tdatatype::FP16 &&
               filter_kernel.getDataType() == nntrainer::Tdatatype::FP32) {
      // FP16-activation depthwise: weights are never Q4_0 for groups>1 and stay
      // FP32 (BN-folded), so this is FP16 input/output x FP32 kernel. Keep it
      // on the tight channel-parallel direct-loop kernel instead of falling
      // into the generic grouped else-branch (per-channel im2col + FP16 GEMV),
      // which was ~2x slower for YOLOv11's detect-head depthwise convs.
      nntrainer::getComputeOps()->depthwise_conv2d_fp16(
        input_.getData<_FP16>(), filter_kernel.getData<float>(),
        hidden_.getData<_FP16>(), in_dim.batch(), filter_size, in_dim.height(),
        in_dim.width(), out_dim.height(), out_dim.width(), fh, fw,
        stride[0].get(), stride[1].get(), padding[0], padding[2],
        dilation[0].get(), dilation[1].get());
#endif
    } else {
      // getSharedDataTensor()/reshape() adopt the *passed* TensorDim's dtype
      // (TensorBase::getSharedDataTensor: ret->dim = dim_). A bare {..} dim
      // literal defaults to FP32, so on an FP16 activation graph every sub-view
      // below would silently relabel FP16-backed storage as FP32 -- im2col
      // would gather at the wrong precision and the dot would write 4-byte
      // floats into 2-byte half storage (overflow / garbage). Carry each
      // parent's real dtype onto its view. (For an all-FP32 graph these are
      // no-ops, so the historical path is unchanged.)
      const auto in_dt = input_.getDataType();
      const auto filt_dt = filter_kernel.getDataType();
      const auto out_dt = hidden_.getDataType();
      for (unsigned int b = 0; b < in_dim.batch(); ++b) {
        Tensor out = hidden_.getBatchSlice(b, 1);
        TensorDim out_rdim({filter_size, owoh});
        out_rdim.setDataType(out_dt);
        out.reshape(out_rdim);
        Tensor in_sub = input_.getBatchSlice(b, 1);
        TensorDim col_dim = calcCol2ImOutputDim(out_dim, fdim_g);
        col_dim.setDataType(in_dt);
        Tensor result = Tensor(col_dim);
        for (unsigned int g = 0; g < groups; ++g) {
          TensorDim ing_dim({1, icg, in_dim.height(), in_dim.width()});
          ing_dim.setDataType(in_dt);
          Tensor in_g =
            in_sub.getSharedDataTensor(ing_dim, (size_t)g * icg * ihw);
          TensorDim filtg_dim({ocg, icg * fh * fw});
          filtg_dim.setDataType(filt_dt);
          Tensor filt_g = filter_kernel.getSharedDataTensor(
            filtg_dim, (size_t)g * ocg * icg * fh * fw);
          TensorDim outg_dim({ocg, owoh});
          outg_dim.setDataType(out_dt);
          Tensor out_g =
            out.getSharedDataTensor(outg_dim, (size_t)g * ocg * owoh);
          result.setZero();
          im2col(in_g, fdim_g, padding, stride, dilation, result);
          filt_g.dot(result, out_g, false, true);
        }
        result.deallocate();
      }
    }
  }

  if (auto &disable_bias = std::get<props::DisableBias>(*layer_impl_props);
      disable_bias.empty() || disable_bias.get() == false) {
    Tensor &bias_kernel = context.getWeight(wt_idx[ConvParams::bias]);
    status = hidden_.add_i(bias_kernel);
    if (status != ML_ERROR_NONE) {
      throw std::invalid_argument("[Conv2D] adding bias failed");
    }
  }

  // Fused activation epilogue. When the graph sets activation=swish on the
  // conv, apply SiLU in-place on the freshly written output instead of
  // materializing a separate Activation layer (which would read the conv
  // output back from memory and write a second full tensor). Only SiLU is
  // fused here (YOLOv11's conv activation); any other activation type is left
  // to a dedicated Activation layer in the graph.
  if (auto &act = std::get<props::FusedActivation>(conv_props);
      !act.empty() && act.get() == ActivationType::ACT_SWISH) {
    const size_t n = hidden_.size();
#ifdef ENABLE_FP16
    if (hidden_.getDataType() == nntrainer::Tdatatype::FP16) {
      convApplySwishInplace(hidden_.getData<_FP16>(), n);
    } else
#endif
    {
      convApplySwishInplace(hidden_.getData<float>(), n);
    }
  }
}

void Conv2DLayer::calcDerivative(RunLayerContext &context) {
  NNTR_THROW_IF(!std::get<props::ConvGroups>(conv_props).empty() &&
                  std::get<props::ConvGroups>(conv_props).get() != 1,
                std::invalid_argument)
    << "[Conv2D] backward for grouped convolution (groups>1) is not yet "
       "implemented; only forward/inference is supported.";
  unsigned int filter_size = std::get<props::FilterSize>(conv_props);
  auto &stride = std::get<std::array<props::Stride, CONV2D_DIM>>(conv_props);
  auto &dilation =
    std::get<std::array<props::Dilation, CONV2D_DIM>>(conv_props);

  const Tensor &derivative = context.getIncomingDerivative(SINGLE_INOUT_IDX);
  Tensor &input_derivative = context.getOutgoingDerivative(SINGLE_INOUT_IDX);
  Tensor &filter_kernel = context.getWeight(wt_idx[ConvParams::weight]);

  TensorDim filter_dim = filter_kernel.getDim();
  TensorDim filter_dim_squeezed{filter_kernel.batch(),
                                filter_kernel.getDim().getFeatureLen()};

  filter_kernel.reshape(filter_dim_squeezed);

  /// for each batch
  /// filter_kernel^T X derivaitive  -> column matrix
  /// col2im(column matrix) to reconstruct the original image

  auto compute_derivative = [&](unsigned int s, unsigned int e,
                                unsigned int pid, void *user_data) {
    Tensor result =
      Tensor(calcCol2ImOutputDim(derivative.getDim(), filter_dim));

    for (unsigned int b = s; b < e; ++b) {
      Tensor deriv_sub = derivative.getBatchSlice(b, 1);
      Tensor in_deriv_sub = input_derivative.getBatchSlice(b, 1);
      deriv_sub.reshape(
        {filter_size, derivative.width() * derivative.height()});
      // filter_kernel is (K, CRS), deriv_sub is (K, OH*OW), result is (CRS,
      // OH*OW)
      filter_kernel.dot(deriv_sub, result, true, false);
      col2im(result, filter_dim, padding, stride, dilation, in_deriv_sub);
      // in_derv_sub is (C,H,W)
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

void Conv2DLayer::calcGradient(RunLayerContext &context) {
  NNTR_THROW_IF(!std::get<props::ConvGroups>(conv_props).empty() &&
                  std::get<props::ConvGroups>(conv_props).get() != 1,
                std::invalid_argument)
    << "[Conv2D] backward for grouped convolution (groups>1) is not yet "
       "implemented; only forward/inference is supported.";
  unsigned int filter_size = std::get<props::FilterSize>(conv_props);
  auto &stride = std::get<std::array<props::Stride, CONV2D_DIM>>(conv_props);
  auto &dilation =
    std::get<std::array<props::Dilation, CONV2D_DIM>>(conv_props);

  const Tensor &derivative = context.getIncomingDerivative(SINGLE_INOUT_IDX);
  Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);

  Tensor &delK = context.getWeightGrad(wt_idx[ConvParams::weight]);
  delK.setZero();

  TensorDim filter_dim = delK.getDim();
  TensorDim filter_dim_squeezed{filter_dim.batch(), filter_dim.getFeatureLen()};

  delK.reshape(filter_dim_squeezed);

  /**
   * no need to set zero for im2col_result, as its lifespan is ITERATION,
   * so its zero padded values will still be zero
   */

  TensorDim out_dim_squeezed{filter_size,
                             derivative.width() * derivative.height()};
  auto workers = ParallelBatch(input_.batch());
  /// input -(im2col)-> column_matrix -> filter x (column_matrix) = output
  /// so delK = dy x column_matrix ^ T;
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
        Tensor deriv_sub = derivative.getBatchSlice(b, 1);
        Tensor delK_sub = delK_par.getBatchSlice(b, 1);
        deriv_sub.reshape(out_dim_squeezed);

        Tensor in_sub = input_.getBatchSlice(b, 1);

        /**
         * @todo this result can be cached from the forward iteration at the
         * expense of memory. In this case, memory of im2col_result must be
         * saved for the whole batch. try this while benchmarking.
         */
        // deriv_sub is (K, OH*OW) and result is (CRS, OH*OW)
        im2col(in_sub, filter_dim, padding, stride, dilation, result);
        deriv_sub.dot(result, delK_sub, false, false);
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
      Tensor deriv_sub = derivative.getBatchSlice(b, 1);
      deriv_sub.reshape(out_dim_squeezed);

      Tensor in_sub = input_.getBatchSlice(b, 1);

      /**
       * @todo this result can be cached from the forward iteration at the
       * expense of memory. In this case, memory of im2col_result must be saved
       * for the whole batch. try this while benchmarking.
       */
      im2col(in_sub, filter_dim, padding, stride, dilation, result);
      deriv_sub.dot(result, delK, false, false, b == 0 ? 0.0f : 1.0f);
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

void Conv2DLayer::setBatch(RunLayerContext &context, unsigned int batch) {
  // The forward scratch buffers (im2col column buffer and quantized-GEMM
  // output) are requested in finalize() sized to the batch present at init.
  // When the runtime batch changes the framework resizes inputs/outputs but
  // not these layer-private scratch tensors, so rebatch them here — otherwise
  // forwarding()'s getBatchSlice(b, 1) for b >= the init batch reads past the
  // planned storage and aborts ("shared tensor bigger than tensor memory").
  if (wt_idx[ConvParams::im2col_scratch] !=
      std::numeric_limits<unsigned int>::max())
    context.updateTensor(wt_idx[ConvParams::im2col_scratch], batch);
  if (wt_idx[ConvParams::qgemm_scratch] !=
      std::numeric_limits<unsigned int>::max())
    context.updateTensor(wt_idx[ConvParams::qgemm_scratch], batch);
}

void Conv2DLayer::exportTo(Exporter &exporter,
                           const ml::train::ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
  exporter.saveResult(conv_props, method, this);
}

void Conv2DLayer::save(std::ofstream &file, RunLayerContext &run_context,
                       bool opt_var, ml::train::ExecutionMode mode,
                       bool trainable, ml::train::TensorDim::DataType dtype,
                       ml::train::ISA target_isa) const {
  // Optimizer-variable save (training only) has no conv-specific layout, so
  // defer to the base implementation.
  if (opt_var) {
    Layer::save(file, run_context, opt_var, mode, trainable, dtype, target_isa);
    return;
  }

  for (unsigned int i = 0; i < run_context.getNumWeights(); ++i) {
    if (!run_context.isGradientFirstAccess(i))
      continue;

    auto &weight = run_context.getWeight(i);

    // No conversion requested, or already the target dtype: save as-is.
    if (dtype == TensorDim::DataType::NONE || weight.getDataType() == dtype) {
      weight.save(file);
      continue;
    }

    NNTR_THROW_IF(dtype != TensorDim::DataType::Q4_0, std::runtime_error)
      << "[Conv2D] save: unsupported quantization dtype";
    NNTR_THROW_IF(weight.getDataType() != TensorDim::DataType::FP32,
                  std::runtime_error)
      << "[Conv2D] Q4_0 save only supports FP32 source weight.";

    // A conv FP32 filter is [out_ch, in_ch, kh, kw] in NCHW, i.e. already
    // row-major [out_ch, CRS] (CRS = in_ch*kh*kw) = [N rows, K cols]. This is
    // exactly the layout quantize_q4_0 consumes (N rows of K), so no transpose
    // is needed (the FC path in the base class transposes because its weight is
    // stored [K, N]). The bias and any non-matmul weight (CRS == 1 or
    // out_ch == 1) are kept FP32.
    const TensorDim dim = weight.getDim();
    const unsigned int out_ch = dim.batch();
    const unsigned int CRS = dim.channel() * dim.height() * dim.width();

    if (out_ch <= 1 || CRS <= 1 || out_ch % 32 != 0 || CRS % 32 != 0) {
      // Not Q4_0-eligible (bias, or block-misaligned): keep FP32 so the saved
      // tensor still matches what the runtime layer allocates for it.
      weight.save(file);
      continue;
    }

    // [1, 1, K=CRS, N=out_ch] is the matmul-weight shape the quantized conv
    // consumes at load (Conv2DLayer::finalize builds the same shape).
    Tensor quant_weight(1, 1, CRS, out_ch, {Tformat::NCHW, dtype});
    std::vector<char> tmp(quant_weight.size());

    quantize_q4_0(weight.getData<float>(), tmp.data(), out_ch, CRS, nullptr);
    repack_q4_0(quant_weight.getData<uint8_t>(), tmp.data(),
                quant_weight.size(), out_ch, CRS, target_isa);
    quant_weight.save(file);
  }
}

void Conv2DLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, conv_props);
  LayerImpl::setProperty(remain_props);
}

} /* namespace nntrainer */
