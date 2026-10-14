// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   gated_delta_core.cpp
 * @date   2026
 * @see    https://github.com/nntrainer/nntrainer
 * @brief  Gated delta rule recurrent core for Qwen3.5 linear attention
 * @bug    No known bugs except for NYI items
 *
 * Math reference: HF modeling_qwen3_5.py torch_recurrent_gated_delta_rule
 * (identical recurrence to the chunk kernel used at HF inference, fp32).
 */

#include "gated_delta_core.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <tensor.h>

namespace causallm {

namespace {

constexpr float L2_EPS = 1e-6f;
constexpr float RMS_EPS = 1e-6f;

inline float sigmoidf(float x) { return 1.0f / (1.0f + std::exp(-x)); }

inline float siluf(float x) { return x * sigmoidf(x); }

inline float silu_grad(float x) {
  float s = sigmoidf(x);
  return s * (1.0f + x * (1.0f - s));
}

/** Numerically stable softplus. */
inline float softplusf(float x) {
  return x > 20.0f ? x : std::log1p(std::exp(x));
}

/** l2-normalize a head vector of length D in place of y (eps 1e-6). */
inline void l2norm_head(const float *x, float *y, unsigned int D) {
  float ss = 0.0f;
  for (unsigned int d = 0; d < D; ++d)
    ss += x[d] * x[d];
  float inv = 1.0f / std::sqrt(ss + L2_EPS);
  for (unsigned int d = 0; d < D; ++d)
    y[d] = x[d] * inv;
}

/**
 * @brief Backward of l2norm_head: given grad g w.r.t. y = x/||x||, accumulate
 * grad w.r.t. x into dx (dx_j = g_j/r - x_j * (g·x) / r^3, r = ||x||_eps).
 */
inline void l2norm_head_backward(const float *x, const float *g, float *dx,
                                 unsigned int D) {
  float ss = 0.0f, gx = 0.0f;
  for (unsigned int d = 0; d < D; ++d)
    ss += x[d] * x[d];
  float r = std::sqrt(ss + L2_EPS);
  for (unsigned int d = 0; d < D; ++d)
    gx += g[d] * x[d];
  float r3 = r * r * r;
  for (unsigned int d = 0; d < D; ++d)
    dx[d] += g[d] / r - x[d] * gx / r3;
}

/**
 * @brief One recurrence step for all heads.
 *
 * S layout: [NH][HD][HD], S_h[c * HD + d]; c = key channel, d = value
 * channel. When o_row/y_row are non-null the (pre-gate) core output and the
 * gated-norm output rows are written. With both null only the state is
 * advanced (used to regenerate checkpoints during backward).
 */
void run_step(const float *q_row, const float *k_row, const float *v_row,
              const float *z_row, const float *b_row, const float *a_row,
              const float *A_log, const float *dt_bias, const float *norm_w,
              unsigned int NH, unsigned int HD, float *S, float *o_row,
              float *y_row) {
  const float q_scale = 1.0f / std::sqrt(static_cast<float>(HD));
  const unsigned int HD2 = HD * HD;

  std::vector<float> qh(HD), kh(HD);
  std::vector<float> kv(HD), delta(HD), o(HD);

  for (unsigned int h = 0; h < NH; ++h) {
    l2norm_head(q_row + h * HD, qh.data(), HD);
    for (unsigned int d = 0; d < HD; ++d)
      qh[d] *= q_scale;
    l2norm_head(k_row + h * HD, kh.data(), HD);

    const float beta = sigmoidf(b_row[h]);
    const float zsp = a_row[h] + dt_bias[h];
    const float g = -std::exp(A_log[h]) * softplusf(zsp);
    const float eg = std::exp(g);

    float *S_h = S + h * HD2;

    // S <- S * exp(g)
    for (unsigned int i = 0; i < HD2; ++i)
      S_h[i] *= eg;

    // kv[d] = sum_c S[c][d] * k[c]
    for (unsigned int d = 0; d < HD; ++d) {
      float s = 0.0f;
      for (unsigned int c = 0; c < HD; ++c)
        s += S_h[c * HD + d] * kh[c];
      kv[d] = s;
    }
    // delta = (v - kv) * beta
    const float *v_h = v_row + h * HD;
    for (unsigned int d = 0; d < HD; ++d)
      delta[d] = (v_h[d] - kv[d]) * beta;
    // S <- S + k (outer) delta
    for (unsigned int c = 0; c < HD; ++c) {
      const float kc = kh[c];
      float *S_row = S_h + c * HD;
      for (unsigned int d = 0; d < HD; ++d)
        S_row[d] += kc * delta[d];
    }
    // o[d] = sum_c S[c][d] * q[c]
    for (unsigned int d = 0; d < HD; ++d) {
      float s = 0.0f;
      for (unsigned int c = 0; c < HD; ++c)
        s += S_h[c * HD + d] * qh[c];
      o[d] = s;
    }

    if (o_row)
      std::memcpy(o_row + h * HD, o.data(), HD * sizeof(float));

    if (y_row) {
      // gated norm (raw weight): y = w * rmsnorm(o) * silu(z)
      float ss = 0.0f;
      for (unsigned int d = 0; d < HD; ++d)
        ss += o[d] * o[d];
      const float inv = 1.0f / std::sqrt(ss / static_cast<float>(HD) + RMS_EPS);
      const float *z_h = z_row + h * HD;
      float *y_h = y_row + h * HD;
      for (unsigned int d = 0; d < HD; ++d)
        y_h[d] = norm_w[d] * (o[d] * inv) * siluf(z_h[d]);
    }
  }
}

} // namespace

GatedDeltaCoreLayer::GatedDeltaCoreLayer() : Layer() {
  weight_idx.fill(std::numeric_limits<unsigned int>::max());
  tensor_idx.fill(std::numeric_limits<unsigned int>::max());
}

void GatedDeltaCoreLayer::finalize(nntrainer::InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() != 6, std::invalid_argument)
    << "[GatedDeltaCoreLayer] requires exactly 6 inputs (q,k,v,z,b,a), got "
    << context.getNumInputs();

  auto &num_heads_p = std::get<props::GDeltaNumHeads>(gdelta_props);
  auto &head_dim_p = std::get<props::GDeltaHeadDim>(gdelta_props);
  num_heads = num_heads_p.get();
  head_dim = head_dim_p.get();
  backward_cache = std::get<props::GDeltaBackwardCache>(gdelta_props).get();
  const unsigned int QKV_W = num_heads * head_dim;
  const unsigned int NH = num_heads;

  const auto &in_dims = context.getInputDimensions();
  for (unsigned int i = 0; i < 6; ++i) {
    NNTR_THROW_IF(in_dims[i].channel() != 1, std::invalid_argument)
      << "[GatedDeltaCoreLayer] input " << i << " must be B×1×T×W";
  }
  const unsigned int B = in_dims[0].batch();
  const unsigned int T = in_dims[0].height();
  for (unsigned int i = 0; i < 4; ++i) {
    NNTR_THROW_IF(in_dims[i].width() != QKV_W, std::invalid_argument)
      << "[GatedDeltaCoreLayer] input " << i << " width must be " << QKV_W
      << ", got " << in_dims[i].width();
    NNTR_THROW_IF(in_dims[i].height() != T, std::invalid_argument)
      << "[GatedDeltaCoreLayer] input heights must agree";
  }
  for (unsigned int i = 4; i < 6; ++i) {
    NNTR_THROW_IF(in_dims[i].width() != NH, std::invalid_argument)
      << "[GatedDeltaCoreLayer] input " << i << " width must be " << NH
      << ", got " << in_dims[i].width();
    NNTR_THROW_IF(in_dims[i].height() != T, std::invalid_argument)
      << "[GatedDeltaCoreLayer] input heights must agree";
  }

  const auto fmt = context.getFormat();

  nntrainer::TensorDim w_dim({1, 1, 1, NH},
                             {fmt, ml::train::TensorDim::DataType::FP32});
  weight_idx[w_A_log] =
    context.requestWeight(w_dim, nntrainer::Initializer::NONE,
                          nntrainer::WeightRegularizer::NONE, 0.0f, 0.0f,
                          "gdelta_A_log", false);
  weight_idx[w_dt_bias] =
    context.requestWeight(w_dim, nntrainer::Initializer::NONE,
                          nntrainer::WeightRegularizer::NONE, 0.0f, 0.0f,
                          "gdelta_dt_bias", false);
  nntrainer::TensorDim nw_dim({1, 1, 1, head_dim},
                              {fmt, ml::train::TensorDim::DataType::FP32});
  weight_idx[w_norm] =
    context.requestWeight(nw_dim, nntrainer::Initializer::NONE,
                          nntrainer::WeightRegularizer::NONE, 0.0f, 0.0f,
                          "gdelta_norm_w", false);

  tensor_idx[t_state] = context.requestTensor(
    {B, NH, head_dim, head_dim}, "gdelta_state", nntrainer::Initializer::ZEROS,
    false, nntrainer::TensorLifespan::MAX_LIFESPAN);

  // Backward caches are training-only: the checkpoints bound the reverse
  // scan's recompute, and the pre-gate output feeds the gated-norm
  // derivative. Inference graphs skip both (they would otherwise cost
  // ~4MB*nseg + 8MB per layer at the 0.8B shape).
  if (backward_cache) {
    const unsigned int nseg = (T + CHECKPOINT_SEG - 1) / CHECKPOINT_SEG;
    tensor_idx[t_ckpt] = context.requestTensor(
      {B * nseg, 1, NH, head_dim * head_dim}, "gdelta_ckpt",
      nntrainer::Initializer::ZEROS, false,
      nntrainer::TensorLifespan::MAX_LIFESPAN);

    tensor_idx[t_o_raw] = context.requestTensor(
      {B, 1, T, QKV_W}, "gdelta_o_raw", nntrainer::Initializer::ZEROS, false,
      nntrainer::TensorLifespan::MAX_LIFESPAN);
  }

  context.setOutputDimensions({in_dims[0]});
}

void GatedDeltaCoreLayer::forwarding(nntrainer::RunLayerContext &context,
                                     bool training) {
  nntrainer::Tensor &q = context.getInput(Q_IDX);
  nntrainer::Tensor &k = context.getInput(K_IDX);
  nntrainer::Tensor &v = context.getInput(V_IDX);
  nntrainer::Tensor &z = context.getInput(Z_IDX);
  nntrainer::Tensor &b = context.getInput(B_IDX);
  nntrainer::Tensor &a = context.getInput(A_IDX);
  nntrainer::Tensor &out = context.getOutput(OUT_IDX);

  const float *A_log = context.getWeight(weight_idx[w_A_log]).getData<float>();
  const float *dt_bias =
    context.getWeight(weight_idx[w_dt_bias]).getData<float>();
  const float *norm_w = context.getWeight(weight_idx[w_norm]).getData<float>();

  const unsigned int B = q.batch();
  const unsigned int T = q.height();
  const unsigned int QKV_W = q.width();
  const unsigned int NH = num_heads;
  const unsigned int HD = head_dim;
  const unsigned int HD2 = HD * HD;

  NNTR_THROW_IF(training && !backward_cache, std::invalid_argument)
    << "[GatedDeltaCoreLayer] training forward requires the "
       "gdelta_backward_cache property (no backward caches requested).";

  const unsigned int nseg = (T + CHECKPOINT_SEG - 1) / CHECKPOINT_SEG;
  float *state_data = context.getTensor(tensor_idx[t_state]).getData<float>();
  float *ckpt_data =
    backward_cache
      ? context.getTensor(tensor_idx[t_ckpt]).getData<float>()
      : nullptr;
  float *o_raw =
    backward_cache ? context.getTensor(tensor_idx[t_o_raw]).getData<float>()
                   : nullptr;

  for (unsigned int bi = 0; bi < B; ++bi) {
    // Every full-sequence forward starts a fresh sample.
    float *S = state_data + static_cast<size_t>(bi) * NH * HD2;
    std::memset(S, 0, NH * HD2 * sizeof(float));

    const float *q_b = q.getData<float>() + static_cast<size_t>(bi) * T * QKV_W;
    const float *k_b = k.getData<float>() + static_cast<size_t>(bi) * T * QKV_W;
    const float *v_b = v.getData<float>() + static_cast<size_t>(bi) * T * QKV_W;
    const float *z_b = z.getData<float>() + static_cast<size_t>(bi) * T * QKV_W;
    const float *b_b = b.getData<float>() + static_cast<size_t>(bi) * T * NH;
    const float *a_b = a.getData<float>() + static_cast<size_t>(bi) * T * NH;
    float *y_b = out.getData<float>() + static_cast<size_t>(bi) * T * QKV_W;
    float *o_b = o_raw + static_cast<size_t>(bi) * T * QKV_W;

    for (unsigned int t = 0; t < T; ++t) {
      if (training && t % CHECKPOINT_SEG == 0)
        std::memcpy(ckpt_data + static_cast<size_t>(bi * nseg + t / CHECKPOINT_SEG) * NH * HD2,
                    S, NH * HD2 * sizeof(float));
      float *o_t = o_b ? o_b + t * QKV_W : nullptr;
      run_step(q_b + t * QKV_W, k_b + t * QKV_W, v_b + t * QKV_W,
               z_b + t * QKV_W, b_b + t * NH, a_b + t * NH, A_log, dt_bias,
               norm_w, NH, HD, S, o_t, y_b + t * QKV_W);
    }
  }
}

void GatedDeltaCoreLayer::incremental_forwarding(
  nntrainer::RunLayerContext &context, unsigned int from, unsigned int to,
  bool training) {
  NNTR_THROW_IF(training, std::invalid_argument)
    << "[GatedDeltaCoreLayer] training must use forwarding().";
  NNTR_THROW_IF(to == 0 || to <= from, std::invalid_argument)
    << "[GatedDeltaCoreLayer] invalid range: from=" << from << ", to=" << to;

  nntrainer::Tensor &q = context.getInput(Q_IDX);
  nntrainer::Tensor &k = context.getInput(K_IDX);
  nntrainer::Tensor &v = context.getInput(V_IDX);
  nntrainer::Tensor &z = context.getInput(Z_IDX);
  nntrainer::Tensor &b = context.getInput(B_IDX);
  nntrainer::Tensor &a = context.getInput(A_IDX);
  nntrainer::Tensor &out = context.getOutput(OUT_IDX);

  const float *A_log = context.getWeight(weight_idx[w_A_log]).getData<float>();
  const float *dt_bias =
    context.getWeight(weight_idx[w_dt_bias]).getData<float>();
  const float *norm_w = context.getWeight(weight_idx[w_norm]).getData<float>();

  const unsigned int B = q.batch();
  const unsigned int H = q.height(); // full seq length for row addressing
  const unsigned int QKV_W = q.width();
  const unsigned int NH = num_heads;
  const unsigned int HD = head_dim;
  const unsigned int HD2 = HD * HD;

  float *state_data = context.getTensor(tensor_idx[t_state]).getData<float>();

  if (from == 0) {
    // New sample: reset the recurrent state.
    std::memset(state_data, 0,
                static_cast<size_t>(B) * NH * HD2 * sizeof(float));
  }

  const bool single = (to - from == 1);
  // NNTrainer places the current token at offset 0 within each batch slice
  // during decode (same convention as CausalConv1DLayer).
  const unsigned int row_base = single ? 0 : from;

  for (unsigned int bi = 0; bi < B; ++bi) {
    float *S = state_data + static_cast<size_t>(bi) * NH * HD2;
    const float *q_b = q.getData<float>() + static_cast<size_t>(bi) * H * QKV_W;
    const float *k_b = k.getData<float>() + static_cast<size_t>(bi) * H * QKV_W;
    const float *v_b = v.getData<float>() + static_cast<size_t>(bi) * H * QKV_W;
    const float *z_b = z.getData<float>() + static_cast<size_t>(bi) * H * QKV_W;
    const float *b_b = b.getData<float>() + static_cast<size_t>(bi) * H * NH;
    const float *a_b = a.getData<float>() + static_cast<size_t>(bi) * H * NH;
    float *y_b = out.getData<float>() + static_cast<size_t>(bi) * H * QKV_W;

    for (unsigned int t = from; t < to; ++t) {
      const unsigned int r = row_base + (t - from);
      run_step(q_b + r * QKV_W, k_b + r * QKV_W, v_b + r * QKV_W,
               z_b + r * QKV_W, b_b + r * NH, a_b + r * NH, A_log, dt_bias,
               norm_w, NH, HD, S, nullptr, y_b + r * QKV_W);
    }
  }
}

void GatedDeltaCoreLayer::calcDerivative(nntrainer::RunLayerContext &context) {
  NNTR_THROW_IF(!backward_cache, std::invalid_argument)
    << "[GatedDeltaCoreLayer] calcDerivative requires the "
       "gdelta_backward_cache property.";

  const unsigned int NH = num_heads;
  const unsigned int HD = head_dim;
  const unsigned int HD2 = HD * HD;
  const unsigned int QKV_W = NH * HD;

  const nntrainer::Tensor &q = context.getInput(Q_IDX);
  const nntrainer::Tensor &k = context.getInput(K_IDX);
  const nntrainer::Tensor &v = context.getInput(V_IDX);
  const nntrainer::Tensor &z = context.getInput(Z_IDX);
  const nntrainer::Tensor &b = context.getInput(B_IDX);
  const nntrainer::Tensor &a = context.getInput(A_IDX);
  const nntrainer::Tensor &dy = context.getIncomingDerivative(OUT_IDX);
  const float *o_raw = context.getTensor(tensor_idx[t_o_raw]).getData<float>();
  const float *ckpt = context.getTensor(tensor_idx[t_ckpt]).getData<float>();

  nntrainer::Tensor &dq = context.getOutgoingDerivative(Q_IDX);
  nntrainer::Tensor &dk = context.getOutgoingDerivative(K_IDX);
  nntrainer::Tensor &dv = context.getOutgoingDerivative(V_IDX);
  nntrainer::Tensor &dz = context.getOutgoingDerivative(Z_IDX);
  nntrainer::Tensor &db = context.getOutgoingDerivative(B_IDX);
  nntrainer::Tensor &da = context.getOutgoingDerivative(A_IDX);

  const float *A_log = context.getWeight(weight_idx[w_A_log]).getData<float>();
  const float *dt_bias =
    context.getWeight(weight_idx[w_dt_bias]).getData<float>();
  const float *norm_w = context.getWeight(weight_idx[w_norm]).getData<float>();

  const unsigned int B = q.batch();
  const unsigned int T = q.height();
  const unsigned int nseg = (T + CHECKPOINT_SEG - 1) / CHECKPOINT_SEG;
  const float q_scale = 1.0f / std::sqrt(static_cast<float>(HD));

  // Scratch: per-step S_prev within one segment (CHECKPOINT_SEG * NH * HD2
  // floats, ~32MB for the 0.8B shape) plus per-step temporaries.
  std::vector<float> seg_states(static_cast<size_t>(CHECKPOINT_SEG) * NH * HD2);
  std::vector<float> D(NH * HD2, 0.0f); // grad w.r.t. S (running)
  std::vector<float> qh(HD), kh(HD), kv(HD), delta(HD), o(HD);
  std::vector<float> do_(NH * HD), ddelta(HD), dkh(HD), dqh(HD);
  std::vector<float> dS_next(NH * HD2);

  for (unsigned int bi = 0; bi < B; ++bi) {
    const float *q_b = q.getData<float>() + static_cast<size_t>(bi) * T * QKV_W;
    const float *k_b = k.getData<float>() + static_cast<size_t>(bi) * T * QKV_W;
    const float *v_b = v.getData<float>() + static_cast<size_t>(bi) * T * QKV_W;
    const float *z_b = z.getData<float>() + static_cast<size_t>(bi) * T * QKV_W;
    const float *b_b = b.getData<float>() + static_cast<size_t>(bi) * T * NH;
    const float *a_b = a.getData<float>() + static_cast<size_t>(bi) * T * NH;
    const float *dy_b =
      dy.getData<float>() + static_cast<size_t>(bi) * T * QKV_W;
    const float *o_b = o_raw + static_cast<size_t>(bi) * T * QKV_W;
    float *dq_b = dq.getData<float>() + static_cast<size_t>(bi) * T * QKV_W;
    float *dk_b = dk.getData<float>() + static_cast<size_t>(bi) * T * QKV_W;
    float *dv_b = dv.getData<float>() + static_cast<size_t>(bi) * T * QKV_W;
    float *dz_b = dz.getData<float>() + static_cast<size_t>(bi) * T * QKV_W;
    float *db_b = db.getData<float>() + static_cast<size_t>(bi) * T * NH;
    float *da_b = da.getData<float>() + static_cast<size_t>(bi) * T * NH;

    std::fill(D.begin(), D.end(), 0.0f);

    for (unsigned int seg = nseg; seg-- > 0;) {
      const unsigned int seg_start = seg * CHECKPOINT_SEG;
      const unsigned int seg_end = std::min(T, seg_start + CHECKPOINT_SEG);

      // Regenerate S_prev for every step of this segment from the checkpoint
      // (state BEFORE step seg_start).
      float *S = seg_states.data();
      std::memcpy(S, ckpt + static_cast<size_t>(bi * nseg + seg) * NH * HD2,
                  NH * HD2 * sizeof(float));
      for (unsigned int t = seg_start; t < seg_end; ++t) {
        std::memcpy(seg_states.data() + static_cast<size_t>(t - seg_start) * NH * HD2,
                    S, NH * HD2 * sizeof(float));
        run_step(q_b + t * QKV_W, k_b + t * QKV_W, v_b + t * QKV_W,
                 z_b + t * QKV_W, b_b + t * NH, a_b + t * NH, A_log, dt_bias,
                 norm_w, NH, HD, S, nullptr, nullptr);
      }

      // Reverse scan within the segment.
      for (unsigned int t = seg_end; t-- > seg_start;) {
        const float *S_prev =
          seg_states.data() + static_cast<size_t>(t - seg_start) * NH * HD2;

        // ---- gated norm backward (y = w * rmsnorm(o) * silu(z)) ----
        const float *dy_t = dy_b + static_cast<size_t>(t) * QKV_W;
        const float *o_t = o_b + static_cast<size_t>(t) * QKV_W;
        const float *z_t = z_b + static_cast<size_t>(t) * QKV_W;
        float *dz_t = dz_b + static_cast<size_t>(t) * QKV_W;

        std::fill(dS_next.begin(), dS_next.end(), 0.0f);

        for (unsigned int h = 0; h < NH; ++h) {
          const float *o_h = o_t + h * HD;
          const float *z_h = z_t + h * HD;
          const float *dy_h = dy_t + h * HD;

          float ss = 0.0f;
          for (unsigned int d = 0; d < HD; ++d)
            ss += o_h[d] * o_h[d];
          const float inv =
            1.0f / std::sqrt(ss / static_cast<float>(HD) + RMS_EPS);

          // G[d] = dy[d] * w[d] * silu(z[d]); sum Gr = sum G[d]*r[d]
          float sum_gr = 0.0f;
          for (unsigned int d = 0; d < HD; ++d) {
            const float r = o_h[d] * inv;
            const float G = dy_h[d] * norm_w[d] * siluf(z_h[d]);
            sum_gr += G * r;
          }
          for (unsigned int d = 0; d < HD; ++d) {
            const float r = o_h[d] * inv;
            const float G = dy_h[d] * norm_w[d] * siluf(z_h[d]);
            do_[h * HD + d] = inv * (G - r * sum_gr / static_cast<float>(HD));
            dz_t[h * HD + d] = dy_h[d] * norm_w[d] * r * silu_grad(z_h[d]);
          }
        }

        // ---- recurrence backward per head ----
        const float *q_t = q_b + static_cast<size_t>(t) * QKV_W;
        const float *k_t = k_b + static_cast<size_t>(t) * QKV_W;
        const float *v_t = v_b + static_cast<size_t>(t) * QKV_W;
        const float *bb_t = b_b + static_cast<size_t>(t) * NH;
        const float *aa_t = a_b + static_cast<size_t>(t) * NH;
        float *dv_t = dv_b + static_cast<size_t>(t) * QKV_W;
        float *dq_t = dq_b + static_cast<size_t>(t) * QKV_W;
        float *dk_t = dk_b + static_cast<size_t>(t) * QKV_W;
        float *db_t = db_b + static_cast<size_t>(t) * NH;
        float *da_t = da_b + static_cast<size_t>(t) * NH;

        for (unsigned int h = 0; h < NH; ++h) {
          float *D_h = D.data() + h * HD2;
          const float *S_h = S_prev + h * HD2;
          float *dSn_h = dS_next.data() + h * HD2;

          l2norm_head(q_t + h * HD, qh.data(), HD);
          for (unsigned int d = 0; d < HD; ++d)
            qh[d] *= q_scale;
          l2norm_head(k_t + h * HD, kh.data(), HD);

          const float beta = sigmoidf(bb_t[h]);
          const float zsp = aa_t[h] + dt_bias[h];
          const float sig_a = sigmoidf(zsp);
          const float eg =
            std::exp(-std::exp(A_log[h]) * softplusf(zsp));

          // D_total = D + q (outer) do
          float *Dt_h = dS_next.data() + h * HD2; // reuse as D_total
          for (unsigned int c = 0; c < HD; ++c)
            for (unsigned int d = 0; d < HD; ++d)
              Dt_h[c * HD + d] = D_h[c * HD + d] + qh[c] * do_[h * HD + d];

          // A = S_prev * e^g; kv, delta
          for (unsigned int d = 0; d < HD; ++d) {
            float s = 0.0f;
            for (unsigned int c = 0; c < HD; ++c)
              s += S_h[c * HD + d] * eg * kh[c];
            kv[d] = s;
            delta[d] = (v_t[h * HD + d] - kv[d]) * beta;
          }

          // ddelta[d] = sum_c D_total[c][d] * k[c]
          for (unsigned int d = 0; d < HD; ++d) {
            float s = 0.0f;
            for (unsigned int c = 0; c < HD; ++c)
              s += Dt_h[c * HD + d] * kh[c];
            ddelta[d] = s;
          }
          // dv = beta * ddelta; dbeta = sum ddelta * (v - kv)
          float dbeta = 0.0f;
          for (unsigned int d = 0; d < HD; ++d) {
            dv_t[h * HD + d] += beta * ddelta[d];
            dbeta += ddelta[d] * (v_t[h * HD + d] - kv[d]);
          }
          db_t[h] += dbeta * beta * (1.0f - beta);

          // dk: outer path (sum_d D_total[c][d]*delta[d]) + kv path
          // (−beta * sum_d S1[c][d]*ddelta[d]); dq: sum_d do[d]*S2[c][d]
          std::fill(dkh.begin(), dkh.end(), 0.0f);
          std::fill(dqh.begin(), dqh.end(), 0.0f);
          for (unsigned int c = 0; c < HD; ++c) {
            float s_k = 0.0f, s_q = 0.0f;
            const float *S_row = S_h + c * HD;
            for (unsigned int d = 0; d < HD; ++d) {
              s_k += Dt_h[c * HD + d] * delta[d] - beta * S_row[d] * eg * ddelta[d];
              s_q += do_[h * HD + d] * (S_row[d] * eg + kh[c] * delta[d]);
            }
            dkh[c] = s_k;
            dqh[c] = s_q;
          }
          l2norm_head_backward(q_t + h * HD, dqh.data(), dq_t + h * HD, HD);
          for (unsigned int d = 0; d < HD; ++d)
            dq_t[h * HD + d] *= q_scale;
          l2norm_head_backward(k_t + h * HD, dkh.data(), dk_t + h * HD, HD);

          // dg = sum_{c,d} D1 * S1 with D1 = D_total − beta*k (outer) ddelta;
          // note sum_c kh[c]*S1[c][d] == kv[d], so the delta term collapses:
          //   dg = sum D_total*S1 − beta * sum_d ddelta[d]*kv[d]
          float dgg = 0.0f;
          for (unsigned int i = 0; i < HD2; ++i)
            dgg += Dt_h[i] * (S_h[i] * eg);
          float dgg_kv = 0.0f;
          for (unsigned int d = 0; d < HD; ++d)
            dgg_kv += ddelta[d] * kv[d];
          dgg -= beta * dgg_kv;
          da_t[h] += dgg * (-std::exp(A_log[h]) * sig_a);

          // dS_prev = e^g * (D_total - beta * k (outer) ddelta)
          for (unsigned int c = 0; c < HD; ++c)
            for (unsigned int d = 0; d < HD; ++d)
              D_h[c * HD + d] =
                eg * (Dt_h[c * HD + d] - beta * kh[c] * ddelta[d]);
        }
        // D now holds the gradient w.r.t. S_prev (state entering step t),
        // carried into the next (earlier) step / segment.
      }
    }
  }
}

void GatedDeltaCoreLayer::calcGradient(nntrainer::RunLayerContext &context) {
  // All weights are frozen (requested trainable=false) and the graph builder
  // marks this node trainable=false, so this is only reachable if the node's
  // trainable flag is flipped manually. Nothing to accumulate.
  (void)context;
}

void GatedDeltaCoreLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, gdelta_props);
  NNTR_THROW_IF(!remain_props.empty(), std::invalid_argument)
    << "[GatedDeltaCoreLayer] Unknown Layer Properties count "
    << std::to_string(values.size());
}

} // namespace causallm

#ifdef PLUGGABLE
extern "C" {

nntrainer::Layer *create_gated_delta_core_layer() {
  return new causallm::GatedDeltaCoreLayer();
}

void destroy_gated_delta_core_layer(nntrainer::Layer *layer) { delete layer; }

nntrainer::LayerPluggable ml_train_layer_pluggable{
  create_gated_delta_core_layer, destroy_gated_delta_core_layer,
  causallm::GatedDeltaCoreLayer::type};
}
#endif
