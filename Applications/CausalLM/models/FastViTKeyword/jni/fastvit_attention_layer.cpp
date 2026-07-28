// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   fastvit_attention_layer.cpp
 * @date   7 July 2026
 * @brief  Multi-head attention custom layer for FastViT-S12 stage 3.
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 */

#include "fastvit_attention_layer.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

#include <nntrainer_error.h>
#include <tensor.h>
#include <tensor_dim.h>

namespace fastvit_keyword {

void FastViTAttentionLayer::setProperty(
  const std::vector<std::string> &values) {
  nntrainer::LayerImpl::setProperty(values);
}

void FastViTAttentionLayer::finalize(nntrainer::InitLayerContext &context) {
  const auto &in_dim = context.getInputDimensions()[0];
  dim_ = in_dim.channel();
  // dim_ = 3 * C (e.g. 1536 for C=512)
  // C = dim_ / 3
  unsigned int C = dim_ / 3;
  head_dim_ = C / num_heads_;

  nntrainer::TensorDim out_dim(in_dim.batch(), C, in_dim.height(),
                               in_dim.width());
  context.setOutputDimensions({out_dim});
}

void FastViTAttentionLayer::forwarding(nntrainer::RunLayerContext &context,
                                       bool training) {
  const nntrainer::Tensor &in = context.getInput(0);
  nntrainer::Tensor &out = context.getOutput(0);

  const auto &in_dim = in.getDim();
  int B = in_dim.batch();
  int C3 = in_dim.channel(); // 3 * C
  int H = in_dim.height();
  int W = in_dim.width();
  int C = C3 / 3; // C = 512
  int num_heads = num_heads_;
  int head_dim = C / num_heads; // 32
  float scale = 1.0f / std::sqrt((float)head_dim);

  // The input tensor may be FP16 when the model uses FP32-FP16 tensor type.
  // Convert to FP32 for the attention computation, then write back.
  nntrainer::Tensor in_fp32 = in;
  if (in.getDataType() != ml::train::TensorDim::DataType::FP32) {
    in_fp32 = in.clone(ml::train::TensorDim::DataType::FP32);
  }

  const float *in_data = in_fp32.getData();

  // Output may also be FP16 — compute in FP32 then cast back.
  nntrainer::TensorDim out_dim = out.getDim();
  nntrainer::Tensor out_fp32(out_dim.batch(), out_dim.channel(),
                             out_dim.height(), out_dim.width(),
                             out_dim.getFormat(),
                             ml::train::TensorDim::DataType::FP32);
  float *out_data = out_fp32.getData();

  multiHeadAttention(in_data, out_data, B, C, H, W, num_heads, head_dim, scale);

  // Cast FP32 result back to the output tensor's dtype
  if (out.getDataType() != ml::train::TensorDim::DataType::FP32) {
    out.copy(out_fp32);
  }

}


void FastViTAttentionLayer::multiHeadAttention(const float *qkv, float *out,
                                               int B, int C, int H, int W,
                                               int num_heads, int head_dim,
                                               float scale) {
  int N = H * W;
  int C3 = 3 * C;

  // qkv layout: [B, 3*C, H, W] in NCHW
  // Q = qkv[b, 0:C, :, :]       -> [B, C, H, W]
  // K = qkv[b, C:2C, :, :]      -> [B, C, H, W]
  // V = qkv[b, 2C:3C, :, :]     -> [B, C, H, W]

  // We need to compute attention per batch, per head:
  //   Q_h: [N, hd]  (from Q reshaped)
  //   K_h: [hd, N]  (from K reshaped, transposed)
  //   V_h: [N, hd]  (from V reshaped)
  //   attn_h = softmax((Q_h * scale) @ K_h^T, dim=-1)  -> [N, N]
  //   out_h = attn_h @ V_h  -> [N, hd]
  //   out reshaped back to [C, H, W]

  // For each batch element
  for (int b = 0; b < B; ++b) {
    const float *Q_base = qkv + b * C3 * H * W;
    const float *K_base = Q_base + C * H * W;
    const float *V_base = K_base + C * H * W;

    // For each head
    for (int h = 0; h < num_heads; ++h) {
      // Q for this head: channels [h*hd : (h+1)*hd], spatial [H, W]
      // In NCHW: Q[h*hd + d][y][x] = Q_base[(h*hd + d) * H * W + y * W + x]
      // We need Q_h as [N, hd] where N = H*W
      // Q_h[n * hd + d] = Q_base[(h*hd + d) * N + n]  (n = y*W + x)

      // K for this head: same layout as Q
      // V for this head: same layout as Q

      // Compute attention scores: attn[N, N] = (Q * scale) @ K^T
      // attn[i, j] = scale * sum_d Q[i, d] * K[j, d]
      // = scale * sum_d Q_base[(h*hd + d)*N + i] * K_base[(h*hd + d)*N + j]

      // We'll compute this in blocks to avoid large temporary storage
      // For N=100 (10x10), N*N = 10000 which is fine

      std::vector<float> attn_scores(N * N);

      // Compute Q * K^T * scale
      for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
          float dot = 0.0f;
          for (int d = 0; d < head_dim; ++d) {
            float q_val = Q_base[(h * head_dim + d) * N + i];
            float k_val = K_base[(h * head_dim + d) * N + j];
            dot += q_val * k_val;
          }
          attn_scores[i * N + j] = dot * scale;
        }
      }

      // Softmax over j (last dim)
      for (int i = 0; i < N; ++i) {
        float max_val = attn_scores[i * N];
        for (int j = 1; j < N; ++j) {
          max_val = std::max(max_val, attn_scores[i * N + j]);
        }
        float sum = 0.0f;
        for (int j = 0; j < N; ++j) {
          attn_scores[i * N + j] = std::exp(attn_scores[i * N + j] - max_val);
          sum += attn_scores[i * N + j];
        }
        float inv_sum = 1.0f / sum;
        for (int j = 0; j < N; ++j) {
          attn_scores[i * N + j] *= inv_sum;
        }
      }

      // Compute out = attn @ V
      // out[i, d] = sum_j attn[i, j] * V[j, d]
      // = sum_j attn_scores[i*N + j] * V_base[(h*hd + d)*N + j]
      // Then write to out[b, h*hd + d, y, x] where i = y*W + x
      float *out_base = out + b * C * H * W;
      for (int i = 0; i < N; ++i) {
        for (int d = 0; d < head_dim; ++d) {
          float val = 0.0f;
          for (int j = 0; j < N; ++j) {
            val += attn_scores[i * N + j] * V_base[(h * head_dim + d) * N + j];
          }
          // Write to output: [B, C, H, W] in NCHW
          // out_base[(h*hd + d) * N + i] = val
          out_base[(h * head_dim + d) * N + i] = val;
        }
      }
    }
  }
}

} // namespace fastvit_keyword
