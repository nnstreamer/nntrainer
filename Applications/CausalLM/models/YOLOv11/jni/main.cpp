// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   main.cpp
 * @date   18 June 2026
 * @brief  YOLOv11m (832x832, nc=1) detection inference example on nntrainer.
 *         Builds the full model (backbone + FPN head + 3-scale Detect head),
 *         loads converted weights, runs one forward pass, and post-processes
 *         (DFL decode + NMS) into final detection boxes.
 *
 *         Usage: yolov11_infer [RES_DIR] [INPUT_BIN]
 *           RES_DIR   dir with weights/ and input bins
 *                     (default: Applications/CausalLM/models/YOLOv11/res)
 *           INPUT_BIN [1,3,832,832] float32 NCHW (default:
 * RES_DIR/input_832.bin) Set env YOLO_VERIFY=1 to also compare raw logits /
 * decoded output to PyTorch references (ref_p3/p4/p5.bin, ref_decoded.bin) when
 * present.
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <utility>
#include <vector>

#include "c2psa_layer.h"
#include <app_context.h>
#include <engine.h>
#include <layer.h>
#include <model.h>
#include <tensor.h>
#include <tensor_api.h>

// Optional direct image input. Enabled only when stb_image.h is present (the
// build defines YOLO_WITH_STB_IMAGE via meson fs.exists). stb_image.h is NOT
// committed; download it to enable .jpg/.png input:
//   curl -fsSL
//   https://raw.githubusercontent.com/nothings/stb/master/stb_image.h \
//     -o Applications/CausalLM/models/YOLOv11/jni/stb_image.h
// Without it the example still builds and runs on .bin input.
#ifdef YOLO_WITH_STB_IMAGE
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#pragma GCC diagnostic pop
#endif

using ml::train::createLayer;
using ml::train::LayerHandle;
using ml::train::Tensor;
using ModelHandle = std::unique_ptr<ml::train::Model>;

namespace yolov11 {

// ===== graph block builders (Conv/BN/SiLU, C3k2, SPPF) =====
/**
 * @brief Build a Conv2d + BN + SiLU sub-graph block.
 *
 * @param name     Layer-name prefix (conv="{name}/conv", bn="{name}/bn")
 * @param in_ch    Input channels
 * @param out_ch   Output channels
 * @param k        Kernel size (square)
 * @param stride   Stride
 * @param padding  Padding
 * @param input    Input symbolic tensor
 * @return Output symbolic tensor after BN+SiLU
 */
inline Tensor convBnSilu(const std::string &name, int in_ch, int out_ch, int k,
                         int stride, int padding, Tensor input) {
  (void)in_ch; // shape is inferred from the graph

  LayerHandle conv(
    createLayer("conv2d", {nntrainer::withKey("name", name + "/conv"),
                           nntrainer::withKey("kernel_size", {k, k}),
                           nntrainer::withKey("filters", out_ch),
                           nntrainer::withKey("stride", {stride, stride}),
                           nntrainer::withKey("padding", padding),
                           nntrainer::withKey("disable_bias", "true")}));
  auto h = conv(input);

  LayerHandle bn(createLayer("batch_normalization",
                             {nntrainer::withKey("name", name + "/bn"),
                              nntrainer::withKey("momentum", "0.9"),
                              nntrainer::withKey("activation", "swish")}));
  return bn(h);
}

/**
 * @brief Build a Bottleneck sub-graph (cv1 3x3 + cv2 3x3 + residual add).
 *
 * PyTorch: return x + cv2(cv1(x))   [shortcut=True]
 *
 * @param name    Layer-name prefix (e.g. "m2/m0/inner0")
 * @param ch      Both input and output channel count
 * @param input   Input symbolic tensor
 * @return Output symbolic tensor
 */
inline Tensor bottleneck(const std::string &name, int ch, Tensor input) {
  auto h = convBnSilu(name + "/cv1", ch, ch, 3, 1, 1, input);
  h = convBnSilu(name + "/cv2", ch, ch, 3, 1, 1, h);
  // residual add
  LayerHandle add(
    createLayer("Addition", {nntrainer::withKey("name", name + "/add")}));
  return add({h, input});
}

/**
 * @brief Build a C3k sub-graph.
 *
 * Forward:
 *   inner_path = inner1(inner0(cv1(x)))   [chain of Bottlenecks]
 *   skip       = cv2(x)
 *   out        = cv3(concat([inner_path, skip], dim=1))
 *
 * @param name       Layer-name prefix (e.g. "m2/m0")
 * @param in_ch      Input channels  (64)
 * @param inner_ch   Inner (half) channels fed to Bottleneck chain (32)
 * @param out_ch     Output channels (64)
 * @param input      Input symbolic tensor
 * @return Output symbolic tensor
 */
inline Tensor c3kBlock(const std::string &name, int in_ch, int inner_ch,
                       int out_ch, Tensor input) {
  // cv1: 1x1, in_ch → inner_ch
  auto inner = convBnSilu(name + "/cv1", in_ch, inner_ch, 1, 1, 0, input);

  // Two Bottleneck blocks
  inner = bottleneck(name + "/inner0", inner_ch, inner);
  inner = bottleneck(name + "/inner1", inner_ch, inner);

  // cv2: 1x1, in_ch → inner_ch  (skip branch)
  auto skip = convBnSilu(name + "/cv2", in_ch, inner_ch, 1, 1, 0, input);

  // concat along channel dim (axis=1)
  LayerHandle cat(
    createLayer("concat", {nntrainer::withKey("name", name + "/cat"),
                           nntrainer::withKey("axis", 1)}));
  auto concat_out = cat({inner, skip});

  // cv3: 1x1, 2*inner_ch → out_ch
  return convBnSilu(name + "/cv3", 2 * inner_ch, out_ch, 1, 1, 0, concat_out);
}

/**
 * @brief Build a C3k2 sub-graph.
 *
 * Forward:
 *   y      = cv1(x)                  // 128→128 (1x1)
 *   y_a    = y[:, :c, :, :]          // first  64 channels
 *   y_b    = y[:, c:, :, :]          // second 64 channels
 *   y_c    = m0(y_b)                 // C3k block, 64→64
 *   out    = cv2(concat([y_a, y_b, y_c], dim=1))  // 192→256
 *
 * @param name     Layer-name prefix (e.g. "m2")
 * @param in_ch    Input channels  (128)
 * @param out_ch   Output channels (256)
 * @param c        Hidden channel count (64 = floor(out_ch * e))
 * @param input    Input symbolic tensor
 * @return Output symbolic tensor
 */
inline Tensor c3k2Block(const std::string &name, int in_ch, int out_ch, int c,
                        Tensor input) {
  // cv1: 1x1, in_ch → 2*c
  auto y = convBnSilu(name + "/cv1", in_ch, 2 * c, 1, 1, 0, input);

  // Split y along channel dim into y_a (first c) and y_b (second c)
  // Using two slice layers: axis=1, 1-indexed [start_index, end_index)
  LayerHandle sliceA(
    createLayer("slice", {nntrainer::withKey("name", name + "/slice_a"),
                          nntrainer::withKey("axis", 1),
                          nntrainer::withKey("start_index", 1),
                          nntrainer::withKey("end_index", c + 1)}));
  auto y_a = sliceA(y);

  LayerHandle sliceB(
    createLayer("slice", {nntrainer::withKey("name", name + "/slice_b"),
                          nntrainer::withKey("axis", 1),
                          nntrainer::withKey("start_index", c + 1),
                          nntrainer::withKey("end_index", 2 * c + 1)}));
  auto y_b = sliceB(y);

  // m0: C3k block, c → c  (inner_ch = c/2 = 32)
  auto y_c = c3kBlock(name + "/m0", c, c / 2, c, y_b);

  // cv2: 1x1, 3*c → out_ch  (concat of y_a, y_b, y_c)
  LayerHandle cat(
    createLayer("concat", {nntrainer::withKey("name", name + "/cat"),
                           nntrainer::withKey("axis", 1)}));
  auto concat_out = cat({y_a, y_b, y_c});

  return convBnSilu(name + "/cv2", 3 * c, out_ch, 1, 1, 0, concat_out);
}

/**
 * @brief Build a Conv2d + BN sub-graph block (NO activation).
 *
 * Used by SPPF cv1 which has act=Identity in PyTorch.
 *
 * @param name     Layer-name prefix (conv="{name}/conv", bn="{name}/bn")
 * @param in_ch    Input channels (unused, inferred from graph)
 * @param out_ch   Output channels
 * @param k        Kernel size
 * @param stride   Stride
 * @param padding  Padding
 * @param input    Input symbolic tensor
 * @return Output symbolic tensor after BN only (no activation)
 */
inline Tensor convBnOnly(const std::string &name, int in_ch, int out_ch, int k,
                         int stride, int padding, Tensor input) {
  (void)in_ch;

  LayerHandle conv(
    createLayer("conv2d", {nntrainer::withKey("name", name + "/conv"),
                           nntrainer::withKey("kernel_size", {k, k}),
                           nntrainer::withKey("filters", out_ch),
                           nntrainer::withKey("stride", {stride, stride}),
                           nntrainer::withKey("padding", padding),
                           nntrainer::withKey("disable_bias", "true")}));
  auto h = conv(input);

  LayerHandle bn(createLayer("batch_normalization",
                             {nntrainer::withKey("name", name + "/bn"),
                              nntrainer::withKey("momentum", "0.9")}));
  return bn(h);
}

/**
 * @brief Build a MaxPool2d layer with explicit padding.
 *
 * @param name     Layer name
 * @param k        Kernel size (square)
 * @param input    Input symbolic tensor
 * @return Output symbolic tensor
 */
inline Tensor maxPool(const std::string &name, int k, Tensor input) {
  int p = k / 2;
  // Padding2D format: "pt,pb,pl,pr"
  std::string pad_str = std::to_string(p) + "," + std::to_string(p) + "," +
                        std::to_string(p) + "," + std::to_string(p);
  LayerHandle pool(
    createLayer("pooling2d", {nntrainer::withKey("name", name),
                              nntrainer::withKey("pooling", "max"),
                              nntrainer::withKey("pool_size", {k, k}),
                              nntrainer::withKey("stride", {1, 1}),
                              nntrainer::withKey("padding", pad_str)}));
  return pool(input);
}

/**
 * @brief Build an SPPF sub-graph.
 *
 * PyTorch SPPF forward (verified by model inspection):
 *   y   = cv1(x)            # Conv+BN, act=Identity (NO SiLU)
 *   y   = [y, m(y), m(m(y)), m(m(m(y)))]   # 3× sequential MaxPool
 *   out = cv2(concat(y, 1)) # Conv+BN+SiLU
 *
 * @param name    Layer-name prefix (e.g. "m9")
 * @param in_ch   Input channels  (512)
 * @param input   Input symbolic tensor
 * @return Output symbolic tensor
 */
inline Tensor sppfBlock(const std::string &name, int in_ch, Tensor input) {
  int half = in_ch / 2;

  // cv1: 1x1, in_ch → in_ch/2, NO activation (act=Identity in PyTorch)
  auto x = convBnOnly(name + "/cv1", in_ch, half, 1, 1, 0, input);

  // Three sequential MaxPool(k=5,s=1,p=2)
  auto p1 = maxPool(name + "/pool1", 5, x);
  auto p2 = maxPool(name + "/pool2", 5, p1);
  auto p3 = maxPool(name + "/pool3", 5, p2);

  // concat([x, p1, p2, p3], dim=1) → half*4 channels
  LayerHandle cat(
    createLayer("concat", {nntrainer::withKey("name", name + "/cat"),
                           nntrainer::withKey("axis", 1)}));
  auto concat_out = cat({x, p1, p2, p3});

  // cv2: 1x1, (in_ch/2)*4 → in_ch, SiLU activation
  return convBnSilu(name + "/cv2", half * 4, in_ch, 1, 1, 0, concat_out);
}

// ===== Detect head graph builders =====
/** @brief 1x1 Conv2d with bias, no BN, no activation (detect output conv). */
inline Tensor convBias1x1(const std::string &name, int out_ch, Tensor input) {
  LayerHandle conv(
    createLayer("conv2d", {nntrainer::withKey("name", name + "/conv"),
                           nntrainer::withKey("kernel_size", {1, 1}),
                           nntrainer::withKey("filters", out_ch),
                           nntrainer::withKey("stride", {1, 1}),
                           nntrainer::withKey("padding", 0)}));
  return conv(input);
}

/** @brief Depthwise 3x3 (pad 1) + BN + SiLU. */
inline Tensor dwConvBnSilu(const std::string &name, int ch, Tensor input) {
  // depthwise = grouped conv2d with groups == channels
  LayerHandle dw(createLayer(
    "conv2d",
    {nntrainer::withKey("name", name + "/dw"),
     nntrainer::withKey("kernel_size", {3, 3}),
     nntrainer::withKey("filters", ch), nntrainer::withKey("groups", ch),
     nntrainer::withKey("stride", {1, 1}), nntrainer::withKey("padding", 1),
     nntrainer::withKey("disable_bias", "true")}));
  auto h = dw(input);
  LayerHandle bn(createLayer("batch_normalization",
                             {nntrainer::withKey("name", name + "/bn"),
                              nntrainer::withKey("momentum", "0.9"),
                              nntrainer::withKey("activation", "swish")}));
  return bn(h);
}

/** @brief Depthwise 3x3 (pad 1) + BN, NO activation (e.g. C2PSA position enc).
 */
inline Tensor dwConvBnOnly(const std::string &name, int ch, Tensor input) {
  // depthwise = grouped conv2d with groups == channels
  LayerHandle dw(createLayer(
    "conv2d",
    {nntrainer::withKey("name", name + "/dw"),
     nntrainer::withKey("kernel_size", {3, 3}),
     nntrainer::withKey("filters", ch), nntrainer::withKey("groups", ch),
     nntrainer::withKey("stride", {1, 1}), nntrainer::withKey("padding", 1),
     nntrainer::withKey("disable_bias", "true")}));
  auto h = dw(input);
  LayerHandle bn(createLayer("batch_normalization",
                             {nntrainer::withKey("name", name + "/bn"),
                              nntrainer::withKey("momentum", "0.9")}));
  return bn(h);
}

/**
 * @brief Build one Detect scale -> raw logits [1, 64+nc, H, W].
 * @param s     name prefix (e.g. "det0")
 * @param pi_ch input feature channels (256 for P3, 512 for P4/P5)
 * @param in    input feature map
 */
inline Tensor detectScale(const std::string &s, int pi_ch, Tensor in) {
  // box branch (cv2)
  auto x = convBnSilu(s + "/cv2_0", pi_ch, 64, 3, 1, 1, in);
  x = convBnSilu(s + "/cv2_1", 64, 64, 3, 1, 1, x);
  auto box = convBias1x1(s + "/cv2_2", 64, x);

  // cls branch (cv3) — depthwise-separable x2 then 1x1
  auto c = dwConvBnSilu(s + "/cv3_0_dw", pi_ch, in);
  c = convBnSilu(s + "/cv3_0_pw", pi_ch, 256, 1, 1, 0, c);
  c = dwConvBnSilu(s + "/cv3_1_dw", 256, c);
  c = convBnSilu(s + "/cv3_1_pw", 256, 256, 1, 1, 0, c);
  auto cls = convBias1x1(s + "/cv3_2", 1, c);

  LayerHandle cat(createLayer("concat", {nntrainer::withKey("name", s + "/out"),
                                         nntrainer::withKey("axis", 1)}));
  return cat({box, cls});
}

// ===== post-processing (DFL decode + dist2bbox + NMS) =====
// ---------------------------------------------------------------------------
// Anchor generation (ultralytics tal.py: make_anchors)
//
// For each scale with grid (H, W) and stride s:
//   sx = [0.5, 1.5, ..., W-0.5]
//   sy = [0.5, 1.5, ..., H-0.5]
//   row-major meshgrid: y outer, x inner → [H*W, 2] (x,y) in grid units
// Concat P3 → P4 → P5.
// ---------------------------------------------------------------------------
struct ScaleInfo {
  int H;
  int W;
  float stride;
};

/**
 * @brief Generate anchor points and stride tensor.
 * @param scales  Vector of (H, W, stride) for each detection scale.
 * @param[out] anchors     [num_anchors, 2] (x, y) in grid units.
 * @param[out] strides_out [num_anchors] stride per anchor.
 */
inline void makeAnchors(const std::vector<ScaleInfo> &scales,
                        std::vector<float> &anchors,
                        std::vector<float> &strides_out) {
  size_t total = 0;
  for (const auto &s : scales) {
    total += static_cast<size_t>(s.H) * s.W;
  }
  anchors.resize(total * 2);
  strides_out.resize(total);

  size_t off = 0;
  for (const auto &s : scales) {
    for (int iy = 0; iy < s.H; ++iy) {
      for (int ix = 0; ix < s.W; ++ix) {
        // anchor (x, y) in grid units with 0.5 offset
        anchors[off * 2 + 0] = static_cast<float>(ix) + 0.5f;
        anchors[off * 2 + 1] = static_cast<float>(iy) + 0.5f;
        strides_out[off] = s.stride;
        ++off;
      }
    }
  }
}

// ---------------------------------------------------------------------------
// DFL (Distribution Focal Loss) decode
//
// Input: raw_box [64, N] (64 = 4 * reg_max, reg_max=16)
//   Interpreted as [4, reg_max, N]: coord-major (4 outer, 16 inner).
//   i.e. bins [0..15] for coord 0, then [0..15] for coord 1, etc.
// Step:
//   1. For each coord c in {0,1,2,3} and anchor a:
//      take raw_box[c*reg_max .. c*reg_max+15][a] → 16-bin logits
//   2. Softmax over 16 bins
//   3. Weighted sum with weights [0,1,...,15]
// Output: dist [4, N] in grid units (ltrb).
// ---------------------------------------------------------------------------
inline void dfl(const float *raw_box, int reg_max, int N,
                std::vector<float> &dist) {
  // raw_box layout: [64, N] = [4*reg_max, N], C-order
  // raw_box[c*reg_max + k][a] = raw_box[(c*reg_max + k)*N + a]
  dist.resize(4 * N);
  for (int c = 0; c < 4; ++c) {
    for (int a = 0; a < N; ++a) {
      // Extract 16 logits for this coord and anchor
      float logits[16];
      for (int k = 0; k < reg_max; ++k) {
        logits[k] = raw_box[(c * reg_max + k) * N + a];
      }
      // Softmax
      float max_logit = *std::max_element(logits, logits + reg_max);
      float sum = 0.0f;
      float exp_v[16];
      for (int k = 0; k < reg_max; ++k) {
        exp_v[k] = std::exp(logits[k] - max_logit);
        sum += exp_v[k];
      }
      // Weighted sum with bin indices [0, 1, ..., 15]
      float val = 0.0f;
      for (int k = 0; k < reg_max; ++k) {
        val += (exp_v[k] / sum) * static_cast<float>(k);
      }
      dist[c * N + a] = val;
    }
  }
}

// ---------------------------------------------------------------------------
// dist2bbox + stride multiply → XYWH pixels at 832-scale
//
// dist [4, N] = (lt_x, lt_y, rb_x, rb_y) in grid units
// anchors [N, 2] = (ax, ay) in grid units
// strides [N]
//
// x1y1 = anchor - lt   (grid)
// x2y2 = anchor + rb   (grid)
// cx = (x1+x2)/2 * stride
// cy = (y1+y2)/2 * stride
// w  = (x2-x1) * stride
// h  = (y2-y1) * stride
//
// Output decoded_box [4, N] = (cx, cy, w, h) pixels
// ---------------------------------------------------------------------------
inline void dist2bbox(const std::vector<float> &dist, int N,
                      const std::vector<float> &anchors,
                      const std::vector<float> &strides,
                      std::vector<float> &decoded_box) {
  // dist layout: [4, N] = lt_x[N], lt_y[N], rb_x[N], rb_y[N]
  const float *lt_x = dist.data() + 0 * N;
  const float *lt_y = dist.data() + 1 * N;
  const float *rb_x = dist.data() + 2 * N;
  const float *rb_y = dist.data() + 3 * N;

  decoded_box.resize(4 * N);
  float *cx = decoded_box.data() + 0 * N;
  float *cy = decoded_box.data() + 1 * N;
  float *w = decoded_box.data() + 2 * N;
  float *h = decoded_box.data() + 3 * N;

  for (int a = 0; a < N; ++a) {
    float ax = anchors[a * 2 + 0];
    float ay = anchors[a * 2 + 1];
    float s = strides[a];

    float x1 = ax - lt_x[a];
    float y1 = ay - lt_y[a];
    float x2 = ax + rb_x[a];
    float y2 = ay + rb_y[a];

    cx[a] = (x1 + x2) * 0.5f * s;
    cy[a] = (y1 + y2) * 0.5f * s;
    w[a] = (x2 - x1) * s;
    h[a] = (y2 - y1) * s;
  }
}

// ---------------------------------------------------------------------------
// Full post-processing pipeline for one scale:
//   raw [65, H, W] → box logits [64, N] + cls logits [1, N]
//   → DFL → dist2bbox → sigmoid(cls)
//   → fills decoded [5, N] starting at offset anchor_off in output
// ---------------------------------------------------------------------------
inline void decodeOneScale(const float *raw, // [65, H, W] channel-major
                           int H, int W, float stride,
                           const std::vector<float> &anchors,
                           const std::vector<float> &strides_vec,
                           int anchor_off, // offset into global anchor array
                           int N_total,    // total anchors (14196)
                           std::vector<float> &decoded // [5, N_total]
) {
  const int N = H * W;
  const int reg_max = 16;

  // raw layout: [65, H, W] = [65, N] in row-major (channel outer, spatial
  // inner) box channels: 0..63, cls channel: 64 Pointer to box part:
  // raw[0..63][a] = raw[c*N + a]
  const float *raw_box = raw;          // [64, N]
  const float *raw_cls = raw + 64 * N; // [1, N]

  // DFL decode: dist [4, N]
  std::vector<float> dist;
  dfl(raw_box, reg_max, N, dist);

  // Anchors for this scale (subset of global anchors)
  // anchors[anchor_off..anchor_off+N-1] × 2
  std::vector<float> scale_anchors(N * 2);
  std::vector<float> scale_strides(N);
  for (int a = 0; a < N; ++a) {
    scale_anchors[a * 2 + 0] = anchors[(anchor_off + a) * 2 + 0];
    scale_anchors[a * 2 + 1] = anchors[(anchor_off + a) * 2 + 1];
    scale_strides[a] = strides_vec[anchor_off + a];
  }

  // dist2bbox → decoded_box [4, N]
  std::vector<float> decoded_box;
  dist2bbox(dist, N, scale_anchors, scale_strides, decoded_box);

  // Fill decoded [5, N_total] at this scale's anchor offset
  // decoded layout: [5, N_total] = (cx[N_total], cy[N_total], w[N_total],
  //                                  h[N_total], cls[N_total])
  for (int c = 0; c < 4; ++c) {
    for (int a = 0; a < N; ++a) {
      decoded[c * N_total + anchor_off + a] = decoded_box[c * N + a];
    }
  }
  // cls: sigmoid
  for (int a = 0; a < N; ++a) {
    decoded[4 * N_total + anchor_off + a] =
      1.0f / (1.0f + std::exp(-raw_cls[a]));
  }
}

// ---------------------------------------------------------------------------
// NMS (non_max_suppression matching ultralytics behavior)
//
// Input: decoded [5, N_total] = (cx, cy, w, h, score)
// Steps:
//   1. xywh → xyxy
//   2. filter score > conf_thres
//   3. per-class offset: box_nms = xyxy + cls_idx * max_wh (agnostic=False)
//      (nc=1, so cls_idx=0 always → no actual offset)
//   4. NMS with iou_thres
//   5. keep up to max_det
// Output: vector of [x1,y1,x2,y2,conf,cls] rows
// ---------------------------------------------------------------------------
struct Detection {
  float x1, y1, x2, y2, conf;
  int cls;
};

inline float iou(const Detection &a, const Detection &b) {
  float ix1 = std::max(a.x1, b.x1);
  float iy1 = std::max(a.y1, b.y1);
  float ix2 = std::min(a.x2, b.x2);
  float iy2 = std::min(a.y2, b.y2);
  float inter_w = std::max(0.0f, ix2 - ix1);
  float inter_h = std::max(0.0f, iy2 - iy1);
  float inter = inter_w * inter_h;
  if (inter == 0.0f)
    return 0.0f;
  float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
  float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
  return inter / (area_a + area_b - inter);
}

/**
 * @brief Run NMS on decoded predictions.
 * @param decoded    [5, N_total] (cx, cy, w, h, score)
 * @param N_total    total anchors
 * @param conf_thres confidence threshold
 * @param iou_thres  IoU threshold
 * @param max_det    max detections to keep
 * @return sorted (descending conf) list of Detection
 */
inline std::vector<Detection> nms(const std::vector<float> &decoded,
                                  int N_total, float conf_thres,
                                  float iou_thres, int max_det) {
  const float *cx_ptr = decoded.data() + 0 * N_total;
  const float *cy_ptr = decoded.data() + 1 * N_total;
  const float *w_ptr = decoded.data() + 2 * N_total;
  const float *h_ptr = decoded.data() + 3 * N_total;
  const float *sc_ptr = decoded.data() + 4 * N_total;

  // nc=1: single class, cls_idx always 0
  // per-class offset = cls_idx * 7680 = 0 for nc=1
  // Build candidate list: xywh → xyxy, filter by score
  std::vector<Detection> candidates;
  candidates.reserve(512);
  for (int a = 0; a < N_total; ++a) {
    float score = sc_ptr[a];
    if (score <= conf_thres)
      continue;
    float cx = cx_ptr[a];
    float cy = cy_ptr[a];
    float bw = w_ptr[a];
    float bh = h_ptr[a];
    float x1 = cx - bw * 0.5f;
    float y1 = cy - bh * 0.5f;
    float x2 = cx + bw * 0.5f;
    float y2 = cy + bh * 0.5f;
    candidates.push_back({x1, y1, x2, y2, score, 0});
  }

  // Sort descending by conf
  std::sort(
    candidates.begin(), candidates.end(),
    [](const Detection &a, const Detection &b) { return a.conf > b.conf; });

  // Greedy NMS
  // For agnostic=False with nc=1, per-class offset = cls_idx*7680 = 0
  // So effectively agnostic NMS here (same as agnostic since nc=1)
  std::vector<bool> suppressed(candidates.size(), false);
  std::vector<Detection> result;
  result.reserve(max_det);

  for (size_t i = 0; i < candidates.size(); ++i) {
    if (suppressed[i])
      continue;
    result.push_back(candidates[i]);
    if (static_cast<int>(result.size()) >= max_det)
      break;
    for (size_t j = i + 1; j < candidates.size(); ++j) {
      if (suppressed[j])
        continue;
      // Both class 0, so NMS boxes are the same as original boxes
      if (iou(candidates[i], candidates[j]) > iou_thres) {
        suppressed[j] = true;
      }
    }
  }

  return result;
}

} // namespace yolov11

namespace {

// Resource directory (yolov11m.safetensors + input/reference bins). Overridable
// via argv[1]; default assumes the binary runs from the nntrainer project root.
std::string RES_DIR = "Applications/CausalLM/models/YOLOv11/res";

/**
 * @brief Load binary file as float vector
 */
std::vector<float> loadBin(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    throw std::runtime_error("Cannot open: " + path);
  }
  f.seekg(0, std::ios::end);
  size_t n = f.tellg() / sizeof(float);
  f.seekg(0);
  std::vector<float> v(n);
  f.read(reinterpret_cast<char *>(v.data()), n * sizeof(float));
  return v;
}

#ifdef YOLO_WITH_STB_IMAGE
/** @brief True if the path looks like an image we can decode with stb_image. */
bool isImagePath(const std::string &p) {
  auto lower = p;
  std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
  for (const char *e : {".jpg", ".jpeg", ".png", ".bmp"})
    if (lower.size() >= std::strlen(e) &&
        lower.compare(lower.size() - std::strlen(e), std::strlen(e), e) == 0)
      return true;
  return false;
}

/**
 * @brief Decode an image and letterbox it to [1,3,size,size] float32 CHW RGB.
 *
 * Mirrors ultralytics letterbox: resize (bilinear, half-pixel centers) keeping
 * aspect ratio, pad the rest with 114, /255. Note the bilinear resampler is not
 * bit-identical to OpenCV's, so detections on a real image are very close to
 * but not exactly equal to the PyTorch/cv2 path; for an exact match feed the
 * .bin written by PyTorch/run_pytorch.py instead.
 */
std::vector<float> loadImageLetterbox(const std::string &path, int size = 832,
                                      float pad = 114.0f) {
  int w = 0, h = 0, c = 0;
  unsigned char *img = stbi_load(path.c_str(), &w, &h, &c, 3); // force RGB
  if (!img)
    throw std::runtime_error("Cannot decode image: " + path);

  const float r = std::min((float)size / h, (float)size / w);
  const int nh = (int)std::round(h * r), nw = (int)std::round(w * r);
  const int top = (size - nh) / 2, left = (size - nw) / 2;

  std::vector<float> out(3UL * size * size, pad / 255.0f);
  for (int oy = 0; oy < nh; ++oy) {
    float sy = (oy + 0.5f) * h / nh - 0.5f; // cv2 INTER_LINEAR mapping
    int y0 = (int)std::floor(sy);
    float fy = sy - y0;
    int y0c = std::min(std::max(y0, 0), h - 1), y1c = std::min(y0 + 1, h - 1);
    for (int ox = 0; ox < nw; ++ox) {
      float sx = (ox + 0.5f) * w / nw - 0.5f;
      int x0 = (int)std::floor(sx);
      float fx = sx - x0;
      int x0c = std::min(std::max(x0, 0), w - 1), x1c = std::min(x0 + 1, w - 1);
      for (int ch = 0; ch < 3; ++ch) {
        auto px = [&](int yy, int xx) {
          return (float)img[(yy * w + xx) * 3 + ch];
        };
        float v = px(y0c, x0c) * (1 - fx) * (1 - fy) +
                  px(y0c, x1c) * fx * (1 - fy) + px(y1c, x0c) * (1 - fx) * fy +
                  px(y1c, x1c) * fx * fy;
        out[(size_t)ch * size * size + (size_t)(top + oy) * size +
            (left + ox)] = v / 255.0f;
      }
    }
  }
  stbi_image_free(img);
  std::cout << "image " << path << " (" << w << "x" << h << ") letterboxed to "
            << size << "x" << size << std::endl;
  return out;
}
#endif // YOLO_WITH_STB_IMAGE

/** @brief Channel-axis slice [start, end) — slice layer uses 1-indexed bounds.
 */
inline Tensor sliceCh(const std::string &name, int start0, int end0,
                      Tensor in) {
  LayerHandle s(createLayer(
    "slice", {nntrainer::withKey("name", name), nntrainer::withKey("axis", 1),
              nntrainer::withKey("start_index", start0 + 1),
              nntrainer::withKey("end_index", end0 + 1)}));
  return s(in);
}

/** @brief Elementwise addition of two tensors. */
inline Tensor addT(const std::string &name, Tensor a, Tensor b) {
  LayerHandle l(createLayer("Addition", {nntrainer::withKey("name", name)}));
  return l({a, b});
}

/**
 * @brief Build the C2PSA block (model.10) from standard layers + the
 *        psa_attention custom op. Input/output [B, 512, H, W].
 *
 *   cv1 = Conv1x1+BN+SiLU; split -> a[256], b[256]
 *   qkv = Conv1x1+BN(b);  V = qkv[256:512]
 *   attn = psa_attention(qkv);  pe = DWConv3x3+BN(V);  attn += pe
 *   b = b + proj(attn);  b = b + ffn1(ffn0(b))
 *   out = Conv1x1+BN+SiLU(concat([a, b]))
 */
inline Tensor buildC2PSA(const std::string &n, Tensor x) {
  auto cv1 = yolov11::convBnSilu(n + "/cv1", 512, 512, 1, 1, 0, x);
  auto a = sliceCh(n + "/slice_a", 0, 256, cv1);
  auto b = sliceCh(n + "/slice_b", 256, 512, cv1);

  auto qkv = yolov11::convBnOnly(n + "/qkv", 256, 512, 1, 1, 0, b);

  // qkv has the per-head interleaved layout [Q32,K32,V64] x 4 heads.
  // Gather the V parts (channels [h*128+64 : h*128+128]) head-major -> [256].
  std::vector<Tensor> v_parts;
  for (int h = 0; h < 4; ++h)
    v_parts.push_back(sliceCh(n + "/slice_v" + std::to_string(h), h * 128 + 64,
                              h * 128 + 128, qkv));
  LayerHandle vcat(
    createLayer("concat", {nntrainer::withKey("name", n + "/vcat"),
                           nntrainer::withKey("axis", 1)}));
  auto v = vcat(v_parts);
  auto pe = yolov11::dwConvBnOnly(n + "/pe", 256, v);

  LayerHandle att(
    createLayer("psa_attention", {nntrainer::withKey("name", n + "/attn")}));
  auto attn = att(qkv);
  auto attn_pe = addT(n + "/add_pe", attn, pe);
  auto proj = yolov11::convBnOnly(n + "/proj", 256, 256, 1, 1, 0, attn_pe);
  auto b1 = addT(n + "/res1", b, proj);

  auto ffn0 = yolov11::convBnSilu(n + "/ffn0", 256, 512, 1, 1, 0, b1);
  // NOTE: ffn1 has BN but NO SiLU (verified against PyTorch; SiLU here yields
  // a ~0.78 error at model.10 that amplifies downstream).
  auto ffn1 = yolov11::convBnOnly(n + "/ffn1", 512, 256, 1, 1, 0, ffn0);
  auto b2 = addT(n + "/res2", b1, ffn1);

  LayerHandle cat(createLayer("concat", {nntrainer::withKey("name", n + "/cat"),
                                         nntrainer::withKey("axis", 1)}));
  auto cc = cat({a, b2});
  return yolov11::convBnSilu(n + "/cv2", 512, 512, 1, 1, 0, cc);
}

inline Tensor buildBackbone(Tensor xIn, Tensor &m4_out, Tensor &m6_out) {
  auto h = yolov11::convBnSilu("conv0", 3, 64, 3, 2, 1, xIn);
  h = yolov11::convBnSilu("conv1", 64, 128, 3, 2, 1, h);
  h = yolov11::c3k2Block("m2", 128, 256, 64, h);
  h = yolov11::convBnSilu("conv3", 256, 256, 3, 2, 1, h);
  m4_out = yolov11::c3k2Block("m4", 256, 512, 128, h);
  h = yolov11::convBnSilu("conv5", 512, 512, 3, 2, 1, m4_out);
  m6_out = yolov11::c3k2Block("m6", 512, 512, 256, h);
  h = yolov11::convBnSilu("conv7", 512, 512, 3, 2, 1, m6_out);
  h = yolov11::c3k2Block("m8", 512, 512, 256, h);
  h = yolov11::sppfBlock("m9", 512, h);
  return buildC2PSA("m10", h); // model.10 (C2PSA)
}

} // namespace

namespace {

/** @brief Upsample(nearest, x2) layer. */
Tensor upsampleX2(const std::string &name, Tensor in) {
  LayerHandle l(
    createLayer("upsample2d", {nntrainer::withKey("name", name),
                               nntrainer::withKey("upsample", "nearest"),
                               nntrainer::withKey("kernel_size", "2,2")}));
  return l(in);
}

/** @brief Channel-axis concat layer. */
Tensor concatCh(const std::string &name, const std::vector<Tensor> &ins) {
  LayerHandle l(createLayer("concat", {nntrainer::withKey("name", name),
                                       nntrainer::withKey("axis", 1)}));
  return l(ins);
}

/**
 * @brief Build the FPN head (model.11~22) and 3-scale Detect head (model.23).
 * @return raw detection logits for P3, P4, P5 (each [1, 4*reg_max+nc, H, W]).
 */
std::vector<Tensor> buildHead(Tensor m4, Tensor m6, Tensor m10) {
  auto m11 = upsampleX2("m11", m10);
  auto m12 = concatCh("m12", {m11, m6});
  auto m13 = yolov11::c3k2Block("m13", 1024, 512, 256, m12);

  auto m14 = upsampleX2("m14", m13);
  auto m15 = concatCh("m15", {m14, m4});
  auto m16 = yolov11::c3k2Block("m16", 1024, 256, 128, m15); // P3 feature

  auto m17 = yolov11::convBnSilu("m17", 256, 256, 3, 2, 1, m16);
  auto m18 = concatCh("m18", {m17, m13});
  auto m19 = yolov11::c3k2Block("m19", 768, 512, 256, m18); // P4 feature

  auto m20 = yolov11::convBnSilu("m20", 512, 512, 3, 2, 1, m19);
  auto m21 = concatCh("m21", {m20, m10});
  auto m22 = yolov11::c3k2Block("m22", 1024, 512, 256, m21); // P5 feature

  // Detect head (model.23): built from standard layers (conv2d/depthwiseconv2d/
  // batch_normalization/concat) — see detect_block.h.
  return {yolov11::detectScale("det0", 256, m16),
          yolov11::detectScale("det1", 512, m19),
          yolov11::detectScale("det2", 512, m22)};
}

/**
 * @brief Load all weights from a single nntrainer safetensors file.
 *
 * The file (produced by PyTorch/convert_weights.py) holds one tensor per model
 * weight, named "{layer}:{weight}" (e.g. "conv0/conv:filter",
 * "conv0/bn:moving_mean"). Tensors of the same layer are contiguous and stored
 * in the layer's weight-registration order, so we group consecutive tensors by
 * layer name and push them to layer->setWeights() in file order.
 */
/** @brief Register the YOLOv11 custom layers with the global AppContext.
 *  Only C2PSA is custom; the Detect head uses standard nntrainer layers. */
void registerCustomLayers() {
  auto &app_ctx = nntrainer::AppContext::Global();
  app_ctx.registerFactory(nntrainer::createLayer<yolov11::PSAAttentionLayer>);
}

/** @brief Optionally compare a logit tensor to a PyTorch reference .bin. */
void verifyAgainst(const std::string &ref_name, const float *out, size_t n) {
  std::ifstream f(RES_DIR + "/" + ref_name, std::ios::binary);
  if (!f) {
    std::cout << "  [verify] " << ref_name << " not found, skipped"
              << std::endl;
    return;
  }
  auto ref = loadBin(RES_DIR + "/" + ref_name);
  float max_diff = 0.0f;
  for (size_t i = 0; i < n && i < ref.size(); ++i)
    max_diff = std::max(max_diff, std::abs(out[i] - ref[i]));
  std::cout << "  [verify] " << ref_name << ": max_abs_diff=" << max_diff
            << std::endl;
}

} // namespace

int main(int argc, char *argv[]) {
  try {
    if (argc > 1)
      RES_DIR = argv[1];
    const std::string input_path =
      (argc > 2) ? argv[2] : (RES_DIR + "/input_832.bin");
    const bool verify = std::getenv("YOLO_VERIFY") != nullptr;

    registerCustomLayers();

    // Build the full model: input -> backbone -> head -> 3 detect outputs.
    ModelHandle model =
      ml::train::createModel(ml::train::ModelType::NEURAL_NET);
    model->setProperty({nntrainer::withKey("batch_size", "1")});

    auto x = Tensor({1, 3, 832, 832}, "input0");
    Tensor m4, m6;
    auto m10 = buildBackbone(x, m4, m6);
    auto outputs = buildHead(m4, m6, m10); // {P3, P4, P5} raw logits

    if (int ret =
          model->compile(x, outputs, ml::train::ExecutionMode::INFERENCE))
      throw std::runtime_error("compile failed: " + std::to_string(ret));
    // Load every weight from the single nntrainer safetensors produced by
    // PyTorch/convert_weights.py (tensor names match the model weight names).
    model->load(RES_DIR + "/yolov11m.safetensors",
                ml::train::ModelFormat::MODEL_FORMAT_SAFETENSORS);
    std::cout << "Model built and weights loaded." << std::endl;

    // Run one forward pass on the input.
    // argv[2] may be a raw [1,3,832,832] float32 .bin (e.g. from
    // run_pytorch.py), or — when built with stb_image.h present — an image
    // (.jpg/.png/...) which is decoded + letterboxed here.
#ifdef YOLO_WITH_STB_IMAGE
    auto input = isImagePath(input_path) ? loadImageLetterbox(input_path)
                                         : loadBin(input_path);
#else
    auto input = loadBin(input_path);
#endif
    std::vector<float *> in_ptr = {input.data()};
    auto outs = model->inference(1, in_ptr, std::vector<float *>());
    std::cout << "Inference done (" << outs.size() << " scale outputs)."
              << std::endl;

    // Post-process: DFL decode + dist2bbox + sigmoid -> [5, N] then NMS.
    std::vector<yolov11::ScaleInfo> scales = {
      {104, 104, 8.0f}, {52, 52, 16.0f}, {26, 26, 32.0f}};
    const int N_total = 104 * 104 + 52 * 52 + 26 * 26; // 14196
    std::vector<float> anchors, strides;
    yolov11::makeAnchors(scales, anchors, strides);

    std::vector<float> decoded(5 * N_total, 0.0f);
    int off = 0;
    for (size_t i = 0; i < scales.size(); ++i) {
      yolov11::decodeOneScale(outs[i], scales[i].H, scales[i].W,
                              scales[i].stride, anchors, strides, off, N_total,
                              decoded);
      off += scales[i].H * scales[i].W;
    }

    const float conf_thres =
      std::getenv("YOLO_CONF") ? std::stof(std::getenv("YOLO_CONF")) : 0.25f;
    const float iou_thres =
      std::getenv("YOLO_IOU") ? std::stof(std::getenv("YOLO_IOU")) : 0.70f;
    auto dets = yolov11::nms(decoded, N_total, conf_thres, iou_thres, 300);

    // JSON output — same field names as the PyTorch reference JSON
    std::cout << "\n[";
    for (size_t i = 0; i < dets.size(); ++i) {
      const auto &d = dets[i];
      if (i)
        std::cout << ",";
      std::printf("\n  {\"x1\": %.6g, \"y1\": %.6g, \"x2\": %.6g,"
                  " \"y2\": %.6g, \"conf\": %.6g, \"cls\": %d}",
                  d.x1, d.y1, d.x2, d.y2, d.conf, d.cls);
    }
    std::cout << (dets.empty() ? "" : "\n") << "]" << std::endl;

    if (verify) {
      std::cout << "\nVerification vs PyTorch references:" << std::endl;
      const size_t ns[3] = {65UL * 104 * 104, 65UL * 52 * 52, 65UL * 26 * 26};
      const char *names[3] = {"ref_p3.bin", "ref_p4.bin", "ref_p5.bin"};
      for (int i = 0; i < 3; ++i)
        verifyAgainst(names[i], outs[i], ns[i]);
      verifyAgainst("ref_decoded.bin", decoded.data(), decoded.size());
    }

    return 0;
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
}
