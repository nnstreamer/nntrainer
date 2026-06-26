// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   conv_indirect.h
 * @date   25 June 2026
 * @brief  Indirect (im2col-fused) convolution activation gather.
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NaN or Inf
 *
 * @details This header provides a single pure-C++ gather primitive used by the
 * indirect quantized convolution path: instead of materializing the full FP32
 * im2col column buffer and feeding it to the GEMM, the GEMM's activation packer
 * gathers each row tile directly from the NCHW input on demand. The gather is
 * factored here (platform-agnostic, no SIMD/backend dependency) so it can be
 * verified bit-identically against a naive im2col reference on every platform,
 * independent of the ARM-only quantize/GEMM micro-kernels that consume it.
 */

#ifndef __NNTR_CONV_INDIRECT_H__
#define __NNTR_CONV_INDIRECT_H__

#include <cstddef>
#include <cstring>

namespace nntrainer {

/**
 * @brief Geometry describing one conv layer's im2col gather.
 *
 * Mirrors the parameters Conv2DLayer feeds to im2col(). out_w is carried so a
 * flat output-spatial row index m can be split into (oh, ow) = (m / out_w,
 * m % out_w). K (the gathered row length) is in_ch * k_h * k_w.
 */
struct ConvGatherParams {
  int in_ch;    /**< input channels */
  int in_h;     /**< input height */
  int in_w;     /**< input width */
  int k_h;      /**< kernel height */
  int k_w;      /**< kernel width */
  int pad_t;    /**< top padding */
  int pad_l;    /**< left padding */
  int stride_h; /**< vertical stride */
  int stride_w; /**< horizontal stride */
  int dil_h;    /**< vertical dilation */
  int dil_w;    /**< horizontal dilation */
  int out_w;    /**< output width (maps row index -> output column) */
};

/**
 * @brief Gather nrows im2col activation rows (FP32) into a contiguous buffer.
 *
 * Reproduces im2col's [OH*OW, CRS] layout exactly for the row range
 * [m0, m0 + nrows): output-spatial row m maps to (oh, ow) = (m / out_w,
 * m % out_w); within a row the K = in_ch*k_h*k_w values are laid out as
 * [channel][k_h][k_w]; positions falling outside the input (padding) are
 * written as 0. The destination is fully written, so no pre-zeroing is needed.
 *
 * @param[out] dst   buffer of nrows*K floats, fully overwritten
 * @param[in]  in    batch-sliced contiguous NCHW input base pointer
 * @param[in]  p     gather geometry
 * @param[in]  m0    first output-spatial row to gather
 * @param[in]  nrows number of rows to gather
 */
inline void gather_conv_act_rows_fp32(float *dst, const float *in,
                                      const ConvGatherParams &p, int m0,
                                      int nrows) {
  const int K = p.in_ch * p.k_h * p.k_w;
  const long inHW = (long)p.in_h * (long)p.in_w;
  const bool unit_dil = (p.dil_h == 1 && p.dil_w == 1);
  const int khkw = p.k_h * p.k_w;

  for (int r = 0; r < nrows; ++r) {
    const int m = m0 + r;
    const int oh = m / p.out_w;
    const int ow = m % p.out_w;
    float *row = dst + (long)r * K;
    /// every K element is written below for unit dilation only along valid
    /// runs, so zero the row up front: padding positions stay 0 (matches
    /// im2col).
    std::memset(row, 0, (size_t)K * sizeof(float));

    const int h0 = oh * p.stride_h - p.pad_t;
    const int w0 = ow * p.stride_w - p.pad_l;

    for (int c = 0; c < p.in_ch; ++c) {
      const float *in_c = in + (long)c * inHW;
      float *row_c = row + (long)c * khkw;
      for (int kh = 0; kh < p.k_h; ++kh) {
        const int h = h0 + kh * p.dil_h;
        if (h < 0 || h >= p.in_h)
          continue; /// whole kernel row is outside the input -> left as 0
        const float *in_row = in_c + (long)h * p.in_w;
        float *dst_run = row_c + (long)kh * p.k_w;
        if (unit_dil) {
          /// the kernel-width window maps a contiguous input run to a
          /// contiguous dest run: copy the in-bounds span in one memcpy.
          int wlo = w0 < 0 ? 0 : w0;
          int whi = w0 + p.k_w;
          if (whi > p.in_w)
            whi = p.in_w;
          if (whi > wlo)
            std::memcpy(dst_run + (wlo - w0), in_row + wlo,
                        (size_t)(whi - wlo) * sizeof(float));
        } else {
          for (int kw = 0; kw < p.k_w; ++kw) {
            const int w = w0 + kw * p.dil_w;
            if (w >= 0 && w < p.in_w)
              dst_run[kw] = in_row[w];
          }
        }
      }
    }
  }
}

} // namespace nntrainer

#endif /* __NNTR_CONV_INDIRECT_H__ */
