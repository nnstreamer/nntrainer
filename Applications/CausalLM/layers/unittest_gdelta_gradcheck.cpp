// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   unittest_gdelta_gradcheck.cpp
 * @date   2026
 * @brief  Finite-difference gradient check for GatedDeltaCoreLayer's
 *         calcDerivative — the reverse-scan input gradients (dQ/dK/dV/dZ/
 *         dB/dA) against central differences of L(x) := dot(forward(x), dy).
 * @bug    No known bugs except for NYI items
 */

#include <gated_delta_core.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <random>
#include <vector>

namespace {

std::vector<nntrainer::Weight *>
makeWeightView(std::vector<nntrainer::Weight> &weights) {
  std::vector<nntrainer::Weight *> view;
  view.reserve(weights.size());
  for (auto &weight : weights)
    view.push_back(&weight);
  return view;
}

std::vector<nntrainer::Var_Grad *>
makeVarGradView(std::vector<nntrainer::Var_Grad> &vars) {
  std::vector<nntrainer::Var_Grad *> view;
  view.reserve(vars.size());
  for (auto &var : vars)
    view.push_back(&var);
  return view;
}

nntrainer::RunLayerContext
makeRunContext(const std::string &name, std::vector<nntrainer::Weight> &weights,
              std::vector<nntrainer::Var_Grad> &inputs,
              std::vector<nntrainer::Var_Grad> &outputs,
              std::vector<nntrainer::Var_Grad> &tensors) {
  return nntrainer::RunLayerContext(
    name, true, 0.0f, false, 1.0f, nullptr, false, makeWeightView(weights),
    makeVarGradView(inputs), makeVarGradView(outputs),
    makeVarGradView(tensors));
}

/**
 * @brief Central-difference check, same contract as the LoRA gradcheck
 *        harness: perturb x[i] by +-eps, re-run forward, and compare the
 *        analytic gradient against (L(x+e) - L(x-e)) / 2e where
 *        L(x) := dot(forward(x), dy).
 */
void checkGradient(float *x, const float *analytic, const float *dy,
                   size_t n, size_t out_n,
                   const std::function<const float *()> &forward,
                   float eps = 1e-3f, float tol = 5e-2f) {
  auto loss = [&](const float *out) {
    double l = 0.0;
    for (size_t j = 0; j < out_n; ++j)
      l += (double)out[j] * dy[j];
    return l;
  };
  for (size_t i = 0; i < n; ++i) {
    const float orig = x[i];

    x[i] = orig + eps;
    const double l1 = loss(forward());

    x[i] = orig - eps;
    const double l2 = loss(forward());

    x[i] = orig;

    const double numeric = (l1 - l2) / (2.0 * (double)eps);
    const double a = analytic[i];
    const double scale = std::max(std::abs(numeric), std::abs(a));
    const double err = std::abs(numeric - a);
    // relative when the gradient is not vanishing, absolute otherwise
    EXPECT_TRUE(err <= tol * scale || err <= 1e-4)
      << "element " << i << ": numeric=" << numeric << " analytic=" << a;
  }
}

/**
 * @brief Independent reference of the layer's full-sequence forward for one
 *        batch row, mirroring run_step + the gated norm. Written from the
 *        HF recurrence directly (not by calling the layer) so a mismatch
 *        localizes the forward bug.
 */
static void referenceForward(const std::vector<std::vector<float>> &ins,
                             const float *A_log, const float *dt_bias,
                             const float *norm_w, unsigned int NH,
                             unsigned int HD, unsigned int T,
                             std::vector<float> &y) {
  const unsigned int QKV_W = NH * HD;
  const float q_scale = 1.0f / std::sqrt(static_cast<float>(HD));
  auto sig = [](float x) { return 1.0f / (1.0f + std::exp(-x)); };
  auto sp = [](float x) { return x > 20.0f ? x : std::log1p(std::exp(x)); };
  auto l2 = [&](const float *x, float *o) {
    float ss = 0.0f;
    for (unsigned int d = 0; d < HD; ++d)
      ss += x[d] * x[d];
    const float inv = 1.0f / std::sqrt(ss + 1e-6f);
    for (unsigned int d = 0; d < HD; ++d)
      o[d] = x[d] * inv;
  };

  std::vector<float> S(NH * HD * HD, 0.0f), qh(HD), kh(HD), kv(HD), delta(HD),
    o(HD);
  for (unsigned int t = 0; t < T; ++t) {
    const float *q = ins[0].data() + t * QKV_W;
    const float *k = ins[1].data() + t * QKV_W;
    const float *v = ins[2].data() + t * QKV_W;
    const float *z = ins[3].data() + t * QKV_W;
    const float *b = ins[4].data() + t * NH;
    const float *a = ins[5].data() + t * NH;
    for (unsigned int h = 0; h < NH; ++h) {
      l2(q + h * HD, qh.data());
      for (unsigned int d = 0; d < HD; ++d)
        qh[d] *= q_scale;
      l2(k + h * HD, kh.data());
      const float beta = sig(b[h]);
      const float eg =
        std::exp(-std::exp(A_log[h]) * sp(a[h] + dt_bias[h]));
      float *S_h = S.data() + h * HD * HD;
      for (unsigned int i = 0; i < HD * HD; ++i)
        S_h[i] *= eg;
      for (unsigned int d = 0; d < HD; ++d) {
        float s = 0.0f;
        for (unsigned int c = 0; c < HD; ++c)
          s += S_h[c * HD + d] * kh[c];
        kv[d] = s;
      }
      for (unsigned int d = 0; d < HD; ++d)
        delta[d] = (v[h * HD + d] - kv[d]) * beta;
      for (unsigned int c = 0; c < HD; ++c)
        for (unsigned int d = 0; d < HD; ++d)
          S_h[c * HD + d] += kh[c] * delta[d];
      for (unsigned int d = 0; d < HD; ++d) {
        float s = 0.0f;
        for (unsigned int c = 0; c < HD; ++c)
          s += S_h[c * HD + d] * qh[c];
        o[d] = s;
      }
      float ss = 0.0f;
      for (unsigned int d = 0; d < HD; ++d)
        ss += o[d] * o[d];
      const float inv = 1.0f / std::sqrt(ss / static_cast<float>(HD) + 1e-6f);
      for (unsigned int d = 0; d < HD; ++d)
        y[t * QKV_W + h * HD + d] =
          norm_w[d] * (o[d] * inv) * (z[h * HD + d] * sig(z[h * HD + d]));
    }
  }
}

} // namespace

TEST(GDeltaGradcheck, CalcDerivativeMatchesFiniteDifference) {
  constexpr unsigned int NH = 2, HD = 8, T = 8;
  constexpr unsigned int QKV_W = NH * HD; // 16
  constexpr unsigned int NSEG = (T + 31) / 32; // 1 (CHECKPOINT_SEG = 32)

  causallm::GatedDeltaCoreLayer layer;
  layer.setProperty({"gdelta_num_heads=2", "gdelta_head_dim=8",
                     "gdelta_backward_cache=true"});
  nntrainer::InitLayerContext init_context(
    {nntrainer::TensorDim({1, 1, T, QKV_W}), // q
     nntrainer::TensorDim({1, 1, T, QKV_W}), // k
     nntrainer::TensorDim({1, 1, T, QKV_W}), // v
     nntrainer::TensorDim({1, 1, T, QKV_W}), // z
     nntrainer::TensorDim({1, 1, T, NH}),    // b
     nntrainer::TensorDim({1, 1, T, NH})},   // a
    // one bool per OUTPUT (the SwiGLU gradcheck with two inputs passes
    // {true}); this layer has a single output.
    {true}, false, "gdelta_gradcheck", "", 0.0f, {"NCHW", "FP32", "FP32"});
  ASSERT_NO_THROW(layer.finalize(init_context));

  std::vector<nntrainer::Weight> weights;
  std::vector<nntrainer::Var_Grad> inputs, outputs, tensors;

  // Frozen weights in finalize()'s request order: A_log, dt_bias, norm_w.
  nntrainer::Tensor a_log_tensor(nntrainer::TensorDim({1, 1, 1, NH}), true);
  nntrainer::Tensor dt_bias_tensor(nntrainer::TensorDim({1, 1, 1, NH}), true);
  nntrainer::Tensor norm_w_tensor(nntrainer::TensorDim({1, 1, 1, HD}), true);
  const float a_log_vals[NH] = {0.4f, -0.2f};
  const float dt_bias_vals[NH] = {0.1f, -0.2f};
  const float norm_w_vals[HD] = {1.0f, 0.9f, 1.1f, 0.8f,
                                 1.2f, 0.7f, 1.0f, 0.9f};
  std::copy(a_log_vals, a_log_vals + NH, a_log_tensor.getData<float>());
  std::copy(dt_bias_vals, dt_bias_vals + NH, dt_bias_tensor.getData<float>());
  std::copy(norm_w_vals, norm_w_vals + HD, norm_w_tensor.getData<float>());
  weights.emplace_back(a_log_tensor, nntrainer::Tensor(), nntrainer::Tensor(),
                       "A_log");
  weights.emplace_back(dt_bias_tensor, nntrainer::Tensor(),
                       nntrainer::Tensor(), "dt_bias");
  weights.emplace_back(norm_w_tensor, nntrainer::Tensor(),
                       nntrainer::Tensor(), "norm_w");

  // Internal tensors in finalize()'s request order: state, ckpt, o_raw.
  tensors.emplace_back(nntrainer::TensorDim({1, NH, HD, HD}),
                       nntrainer::Initializer::NONE, true, true, "state");
  tensors.emplace_back(nntrainer::TensorDim({1 * NSEG, 1, NH, HD * HD}),
                       nntrainer::Initializer::NONE, true, true, "ckpt");
  tensors.emplace_back(nntrainer::TensorDim({1, 1, T, QKV_W}),
                       nntrainer::Initializer::NONE, true, true, "o_raw");

  for (unsigned int i = 0; i < 4; ++i)
    inputs.emplace_back(init_context.getInputDimensions()[i],
                        nntrainer::Initializer::NONE, true, true,
                        (std::string("in") + std::to_string(i)));
  inputs.emplace_back(init_context.getInputDimensions()[4],
                      nntrainer::Initializer::NONE, true, true, "b");
  inputs.emplace_back(init_context.getInputDimensions()[5],
                      nntrainer::Initializer::NONE, true, true, "a");
  outputs.emplace_back(init_context.getOutSpecs()[0].variable_spec.dim,
                       nntrainer::Initializer::NONE, true, true, "output");

  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
  for (auto &in : inputs) {
    float *p = in.getVariableRef().getData<float>();
    const size_t n = in.getVariableRef().size();
    for (size_t i = 0; i < n; ++i)
      p[i] = dist(rng);
  }

  auto run_context =
    makeRunContext("gdelta_gradcheck", weights, inputs, outputs, tensors);

  std::vector<float> dy(T * QKV_W);
  {
    std::mt19937 rng_dy(7);
    std::uniform_real_distribution<float> ddy(-0.3f, 0.3f);
    for (auto &v : dy)
      v = ddy(rng_dy);
  }
  std::copy(dy.begin(), dy.end(),
            run_context.getOutputGradUnsafe(0).getData<float>());

  layer.forwarding(run_context, true);
  layer.calcDerivative(run_context);

  std::vector<std::vector<float>> analytic(6);
  for (unsigned int idx = 0; idx < 6; ++idx) {
    analytic[idx].resize(T * (idx < 4 ? QKV_W : NH));
    const float *src = run_context.getOutgoingDerivative(idx).getData<float>();
    std::copy(src, src + analytic[idx].size(), analytic[idx].begin());
  }

  auto forward = [&]() -> const float * {
    layer.forwarding(run_context, true);
    return run_context.getOutput(0).getData<float>();
  };

  /// [DIAG] forward-sensitivity probe for the elements whose finite-
  /// difference gradient came out ~0 while the analytic derivative did not:
  /// if the loss is bit-identical under the perturbation, the composed
  /// forward never reads them (an indexing bug); if it moves, the harness
  /// is at fault.
  {
    auto lossOf = [&]() {
      double l = 0.0;
      const float *o = run_context.getOutput(0).getData<float>();
      for (size_t j = 0; j < dy.size(); ++j)
        l += (double)o[j] * dy[j];
      return l;
    };
    forward();
    std::cout << "[DIAG] input sizes:";
    for (auto &in : inputs)
      std::cout << " " << in.getVariableRef().size();
    std::cout << " | L_base=" << lossOf() << " | y0:";
    const float *y0 = run_context.getOutput(0).getData<float>();
    for (unsigned int w = 0; w < QKV_W; ++w)
      std::cout << " " << y0[w];
    std::cout << "\n";
    struct Probe {
      unsigned int idx;
      size_t elem;
      const char *tag;
    };
    const Probe probes[] = {{0, 1, "q[t0,h0,w1]"}, {0, 8, "q[t0,h1,w0]"},
                            {4, 0, "b[t0,h0]"}, {4, 2, "b[t1,h0]"}};
    for (const auto &p : probes) {
      float *x = inputs[p.idx].getVariableRef().getData<float>();
      const float orig = x[p.elem];
      x[p.elem] = orig + 1e-3f;
      forward();
      const double lp = lossOf();
      x[p.elem] = orig - 1e-3f;
      forward();
      const double lm = lossOf();
      x[p.elem] = orig;
      std::cout << "[DIAG] " << p.tag << ": L+ " << lp << "  L- " << lm
                << "  dL " << (lp - lm) << "\n";
    }
    forward();

    // pre-gate core output at t0 and an independent reference of the whole
    // forward: separates "the forward computes something else for head 1"
    // from "the backward reads something else".
    {
      const float *o_raw_p = tensors[2].getVariableRef().getData<float>();
      std::cout << "[DIAG] o_raw[t0]:";
      for (unsigned int w = 0; w < QKV_W; ++w)
        std::cout << " " << o_raw_p[w];
      std::cout << "\n";

      std::vector<std::vector<float>> ins(6);
      for (unsigned int i = 0; i < 6; ++i) {
        const float *p = inputs[i].getVariableRef().getData<float>();
        ins[i].assign(p, p + inputs[i].getVariableRef().size());
      }
      std::vector<float> y_ref(T * QKV_W, 0.0f);
      referenceForward(ins, a_log_vals, dt_bias_vals, norm_w_vals, NH, HD, T,
                       y_ref);
      const float *y_layer = run_context.getOutput(0).getData<float>();
      double diff[2] = {0, 0};
      for (unsigned int t = 0; t < T; ++t)
        for (unsigned int h = 0; h < 2; ++h)
          for (unsigned int d = 0; d < HD; ++d) {
            const unsigned int idx = t * QKV_W + h * HD + d;
            diff[h] = std::max(diff[h], (double)std::abs(y_layer[idx] -
                                                         y_ref[idx]));
          }
      std::cout << "[DIAG] forward-vs-reference maxdiff: head0 " << diff[0]
                << "  head1 " << diff[1] << "\n";
    }
  }

  // q, k, v, z share the [T, QKV_W] layout; b/a are [T, NH].
  static const char *in_names[6] = {"q", "k", "v", "z", "b", "a"};
  for (unsigned int idx = 0; idx < 6; ++idx) {
    float *x = inputs[idx].getVariableRef().getData<float>();
    std::cout << "[DIAG] checking input " << in_names[idx] << " ("
              << analytic[idx].size() << " elems)\n";
    checkGradient(x, analytic[idx].data(), dy.data(), analytic[idx].size(),
                  T * QKV_W, forward);
  }
}
