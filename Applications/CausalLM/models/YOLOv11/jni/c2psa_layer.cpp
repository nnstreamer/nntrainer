// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   c2psa_layer.cpp
 * @date   18 June 2026
 * @brief  PSA spatial multi-head attention custom layer (weight-free).
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs
 */

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include "c2psa_layer.h"

#include <layer_context.h>
#include <nntrainer_error.h>

namespace yolov11 {

void PSAAttentionLayer::finalize(nntrainer::InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "PSAAttentionLayer expects exactly 1 input (qkv)";

  const nntrainer::TensorDim &in_dim = context.getInputDimensions()[0];
  dim_ = in_dim.channel();

  const unsigned int expected = NUM_HEADS * KD * 2 + NUM_HEADS * VD; // 512
  NNTR_THROW_IF(dim_ != expected, std::invalid_argument)
    << "PSAAttentionLayer expects " << expected << " input channels, got "
    << dim_;

  nntrainer::TensorDim out_dim = in_dim;
  out_dim.channel(NUM_HEADS * VD); // 256
  context.setOutputDimensions({out_dim});
}

void PSAAttentionLayer::multiHeadAttention(const float *Q, const float *K,
                                           const float *V, float *out, int nh,
                                           int kd, int vd, int N) {
  const float scale = 1.0f / std::sqrt(static_cast<float>(kd));
  std::vector<float> score(static_cast<size_t>(N) * N);

  for (int h = 0; h < nh; ++h) {
    const float *Qh = Q + static_cast<size_t>(h) * kd * N;
    const float *Kh = K + static_cast<size_t>(h) * kd * N;
    const float *Vh = V + static_cast<size_t>(h) * vd * N;
    float *outh = out + static_cast<size_t>(h) * vd * N;

    // score[i,j] = (Q[h,:,i] . K[h,:,j]) * scale
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < N; ++j) {
        float s = 0.0f;
        for (int d = 0; d < kd; ++d)
          s += Qh[d * N + i] * Kh[d * N + j];
        score[static_cast<size_t>(i) * N + j] = s * scale;
      }
    }
    // softmax over key axis j for each query i
    for (int i = 0; i < N; ++i) {
      float *row = score.data() + static_cast<size_t>(i) * N;
      float mx = *std::max_element(row, row + N);
      float sum = 0.0f;
      for (int j = 0; j < N; ++j) {
        row[j] = std::exp(row[j] - mx);
        sum += row[j];
      }
      const float inv = 1.0f / sum;
      for (int j = 0; j < N; ++j)
        row[j] *= inv;
    }
    // out[h,d,i] = sum_j score[i,j] * V[h,d,j]
    for (int d = 0; d < vd; ++d) {
      for (int i = 0; i < N; ++i) {
        float s = 0.0f;
        const float *row = score.data() + static_cast<size_t>(i) * N;
        for (int j = 0; j < N; ++j)
          s += row[j] * Vh[d * N + j];
        outh[d * N + i] = s;
      }
    }
  }
}

void PSAAttentionLayer::forwarding(nntrainer::RunLayerContext &context,
                                   bool training) {
  const nntrainer::Tensor &in_t = context.getInput(0);
  nntrainer::Tensor &out_t = context.getOutput(0);

  const nntrainer::Tdatatype dtype = in_t.getDataType();
#ifdef ENABLE_FP16
  const bool fp16 = (dtype == nntrainer::Tdatatype::FP16);
#else
  const bool fp16 = false;
#endif
  NNTR_THROW_IF(dtype != nntrainer::Tdatatype::FP32 && !fp16,
                std::invalid_argument)
    << "PSAAttentionLayer only implements the FP32 and FP16 activation paths";

  const unsigned int B = in_t.batch();
  const unsigned int H = in_t.height();
  const unsigned int W = in_t.width();
  const int N = static_cast<int>(H * W);
  const unsigned int q_ch = NUM_HEADS * KD; // 128
  const unsigned int v_ch = NUM_HEADS * VD; // 256

  // The attention math runs in FP32 for numerical stability. For an FP16
  // activation model the input/output tensors are FP16, so stage them through
  // FP32 buffers at the layer boundary (FP16->FP32 in, FP32->FP16 out).
  std::vector<float> in_f32, out_f32;
  const float *in;
  float *out;
#ifdef ENABLE_FP16
  if (fp16) {
    in_f32.resize(in_t.size());
    const _FP16 *src = in_t.getData<_FP16>();
    for (size_t i = 0; i < in_f32.size(); ++i)
      in_f32[i] = static_cast<float>(src[i]);
    out_f32.resize(out_t.size());
    in = in_f32.data();
    out = out_f32.data();
  } else
#endif
  {
    in = in_t.getData<float>();
    out = out_t.getData<float>();
  }

  // qkv uses the ultralytics per-head interleaved layout: for head h the
  // 128 channels [h*128 : h*128+128] hold [Q(kd=32), K(kd=32), V(vd=64)].
  // Gather them into head-major contiguous Q/K/V buffers [nh,kd|vd,N].
  const unsigned int head_stride = 2 * KD + VD; // 128
  std::vector<float> Qb(static_cast<size_t>(q_ch) * N);
  std::vector<float> Kb(static_cast<size_t>(q_ch) * N);
  std::vector<float> Vb(static_cast<size_t>(v_ch) * N);

  for (unsigned int b = 0; b < B; ++b) {
    const float *qkv = in + static_cast<size_t>(b) * dim_ * N;
    for (unsigned int h = 0; h < NUM_HEADS; ++h) {
      const float *qh = qkv + static_cast<size_t>(h) * head_stride * N;
      std::memcpy(Qb.data() + static_cast<size_t>(h) * KD * N, qh,
                  static_cast<size_t>(KD) * N * sizeof(float));
      std::memcpy(Kb.data() + static_cast<size_t>(h) * KD * N,
                  qh + static_cast<size_t>(KD) * N,
                  static_cast<size_t>(KD) * N * sizeof(float));
      std::memcpy(Vb.data() + static_cast<size_t>(h) * VD * N,
                  qh + static_cast<size_t>(2 * KD) * N,
                  static_cast<size_t>(VD) * N * sizeof(float));
    }
    float *out_b = out + static_cast<size_t>(b) * v_ch * N;
    multiHeadAttention(Qb.data(), Kb.data(), Vb.data(), out_b, NUM_HEADS, KD,
                       VD, N);
  }

#ifdef ENABLE_FP16
  if (fp16) {
    _FP16 *dst = out_t.getData<_FP16>();
    for (size_t i = 0; i < out_f32.size(); ++i)
      dst[i] = static_cast<_FP16>(out_f32[i]);
  }
#endif
}

} // namespace yolov11
