// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   unittest_layers_fully_connected_lora_qat.cpp
 * @date   31 July 2026
 * @brief  Correctness checks for FullyConnectedLayer's per-channel int4 QAT
 *         fake-quantization (EMA-calibrated) and its straight-through
 *         estimator (STE) backward.
 * @bug    No known bugs except for NYI items
 */

#include <layer_context.h>
#include <var_grad.h>
#include <weight.h>

#include <fc_layer.h>

#include <gtest/gtest.h>

#include <cmath>
#include <utility>
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
 * @brief Reference recomputation of FullyConnectedLayer::fakeQuantizeQInt4's
 *        forward math, given the per-channel EMA scale already calibrated by
 *        a real forward pass (retrieved via getRegisteredChannelScales()).
 *        This is NOT a reimplementation of the EMA calibration itself --
 *        only of the quantize-with-a-known-scale step -- so a test built on
 *        it still exercises the real calibration code.
 */
float fakeQuantWithScale(float v, float d) {
  if (d < 1e-10f)
    return v;
  float q = std::round(v / d);
  q = std::max(-8.0f, std::min(7.0f, q));
  return q * d;
}

/** Channels are the width axis: channel n is column n of the (K x N) weight
 *  (nntrainer/GGML convention: height=K, width=N), matching
 *  fakeQuantizeQInt4 exactly. */
nntrainer::Tensor
applyFakeQuantWithScales(const nntrainer::Tensor &x,
                         const std::vector<float> &channel_d) {
  const size_t K = x.getDim().height();
  const size_t N = x.getDim().width();
  nntrainer::Tensor out = x.clone();
  for (size_t k = 0; k < K; ++k) {
    for (size_t n = 0; n < N; ++n) {
      out.getValue<float>(k * N + n) =
        fakeQuantWithScale(x.getValue<float>(k * N + n), channel_d[n]);
    }
  }
  return out;
}

/**
 * @brief Common fixture: a FullyConnectedLayer with lora_qat enabled, a
 *        zeroed (and frozen) base weight and bias disabled -- so the
 *        layer's entire output is the LoRA path, letting these tests check
 *        the QAT/STE math in isolation from the base matmul.
 */
struct FCLoraQATFixture {
  static constexpr unsigned int height = 2, in_dim = 4, unit = 6,
                                lora_rank = 32;

  nntrainer::FullyConnectedLayer layer;
  nntrainer::InitLayerContext init_context;
  std::vector<nntrainer::Weight> weights;
  std::vector<nntrainer::Var_Grad> inputs, outputs, tensors;
  std::vector<float> loraA_init, loraB_init;
  // rc's initializer calls build(), which fills the members above -- keep
  // rc declared last so it's constructed last (member init order follows
  // declaration order, not the constructor's initializer-list order).
  nntrainer::RunLayerContext rc;

  FCLoraQATFixture(const std::string &name) :
    init_context({nntrainer::TensorDim({1, 1, height, in_dim})}, {true}, false,
                 name, "", 0.0f, {"NCHW", "FP32", "FP32"}),
    rc(build(name)) {}

  nntrainer::RunLayerContext build(const std::string &name) {
    layer.setProperty(
      {"unit=" + std::to_string(unit), "lora_rank=" + std::to_string(lora_rank),
       "lora_qat=true", "disable_bias=true", "weight_initializer=zeros"});
    layer.finalize(init_context);

    // weight (base, in_dim x unit): left at its ZEROS initializer value, so
    // the layer's output is exactly the LoRA contribution.
    nntrainer::Tensor weight_v(nntrainer::TensorDim({1, 1, in_dim, unit}),
                               true);
    weight_v.setZero();
    weights.emplace_back(weight_v, nntrainer::Tensor(), nntrainer::Tensor(),
                         "weight");

    // loraA (in_dim x lora_rank = 128 elements = 32 channels)
    nntrainer::Tensor loraA_v(nntrainer::TensorDim({1, 1, in_dim, lora_rank}),
                              true);
    loraA_init.resize(in_dim * lora_rank);
    for (unsigned int i = 0; i < loraA_init.size(); ++i)
      loraA_init[i] = std::sin(0.37f * i + 0.11f) * 0.6f;
    std::copy(loraA_init.begin(), loraA_init.end(), loraA_v.getData<float>());
    nntrainer::Tensor loraA_g(nntrainer::TensorDim({1, 1, in_dim, lora_rank}),
                              true);
    loraA_g.setZero();
    weights.emplace_back(loraA_v, loraA_g, nntrainer::Tensor(), "loraA");

    // loraB (lora_rank x unit = 192 elements = 6 channels)
    nntrainer::Tensor loraB_v(nntrainer::TensorDim({1, 1, lora_rank, unit}),
                              true);
    loraB_init.resize(lora_rank * unit);
    for (unsigned int i = 0; i < loraB_init.size(); ++i)
      loraB_init[i] = std::sin(0.53f * i + 0.29f) * 0.4f;
    std::copy(loraB_init.begin(), loraB_init.end(), loraB_v.getData<float>());
    nntrainer::Tensor loraB_g(nntrainer::TensorDim({1, 1, lora_rank, unit}),
                              true);
    loraB_g.setZero();
    weights.emplace_back(loraB_v, loraB_g, nntrainer::Tensor(), "loraB");

    // loraTmp/loraOut: scratch tensors finalize() also requests via
    // context.requestTensor(); forwarding()/calcGradient() index into the
    // RunLayerContext's tensors vector by the same relative position, so
    // these must be present even though the test never reads loraOut
    // directly.
    tensors.emplace_back(nntrainer::TensorDim({1, 1, height, lora_rank}),
                         nntrainer::Initializer::NONE, true, true,
                         "hidden_tmp_lora");
    tensors.emplace_back(nntrainer::TensorDim({1, 1, height, unit}),
                         nntrainer::Initializer::NONE, true, true,
                         "hidden_lora");

    inputs.emplace_back(init_context.getInputDimensions()[0],
                        nntrainer::Initializer::NONE, true, true, "input");
    for (unsigned int i = 0; i < height * in_dim; ++i)
      inputs[0].getVariableRef().getData<float>()[i] =
        std::sin(0.71f * i + 0.05f) * 0.5f;

    outputs.emplace_back(init_context.getOutSpecs()[0].variable_spec.dim,
                         nntrainer::Initializer::NONE, true, true, "output");

    return makeRunContext(name, weights, inputs, outputs, tensors);
  }
};

} // namespace

TEST(FullyConnectedLoraQAT, ForwardMatchesReferenceFakeQuantAndIsNotIdentity) {
  FCLoraQATFixture fx("fc_qat_forward");

  fx.layer.forwarding(fx.rc, true);

  auto [a_cd, b_cd] =
    nntrainer::FullyConnectedLayer::getRegisteredChannelScales(
      "fc_qat_forward");
  ASSERT_EQ(a_cd.size(), fx.lora_rank);
  ASSERT_EQ(b_cd.size(), fx.unit);

  nntrainer::Tensor loraA_ref(
    nntrainer::TensorDim({1, 1, fx.in_dim, fx.lora_rank}), true);
  std::copy(fx.loraA_init.begin(), fx.loraA_init.end(),
            loraA_ref.getData<float>());
  nntrainer::Tensor loraB_ref(
    nntrainer::TensorDim({1, 1, fx.lora_rank, fx.unit}), true);
  std::copy(fx.loraB_init.begin(), fx.loraB_init.end(),
            loraB_ref.getData<float>());

  nntrainer::Tensor a_fq_ref = applyFakeQuantWithScales(loraA_ref, a_cd);
  nntrainer::Tensor b_fq_ref = applyFakeQuantWithScales(loraB_ref, b_cd);

  // Sanity: quantization actually changed something. If this ever fails,
  // the test data no longer exercises rounding and the checks below would
  // pass vacuously even with a broken STE wiring.
  float max_abs_diff = 0.0f;
  for (unsigned int i = 0; i < fx.loraA_init.size(); ++i)
    max_abs_diff = std::max(
      max_abs_diff, std::abs(a_fq_ref.getData<float>()[i] - fx.loraA_init[i]));
  EXPECT_GT(max_abs_diff, 1e-6f)
    << "fake-quant left loraA unchanged; test data no longer discriminates";

  nntrainer::Tensor lora_tmp_ref(
    nntrainer::TensorDim({1, 1, fx.height, fx.lora_rank}), true);
  fx.inputs[0].getVariableRef().dot(a_fq_ref, lora_tmp_ref, false, false);
  nntrainer::Tensor expected_out(
    nntrainer::TensorDim({1, 1, fx.height, fx.unit}), true);
  lora_tmp_ref.dot(b_fq_ref, expected_out, false, false);

  const float *actual = fx.rc.getOutput(0).getData<float>();
  const float *expected = expected_out.getData<float>();
  for (unsigned int i = 0; i < fx.height * fx.unit; ++i)
    EXPECT_NEAR(actual[i], expected[i], 1e-5f)
      << "forward output mismatch at " << i;
}

TEST(FullyConnectedLoraQAT,
     CalcDerivativeAndCalcGradientUseFakeQuantizedWeightsSTE) {
  FCLoraQATFixture fx("fc_qat_backward");

  fx.layer.forwarding(fx.rc, true);

  auto [a_cd, b_cd] =
    nntrainer::FullyConnectedLayer::getRegisteredChannelScales(
      "fc_qat_backward");

  nntrainer::Tensor loraA_ref(
    nntrainer::TensorDim({1, 1, fx.in_dim, fx.lora_rank}), true);
  std::copy(fx.loraA_init.begin(), fx.loraA_init.end(),
            loraA_ref.getData<float>());
  nntrainer::Tensor loraB_ref(
    nntrainer::TensorDim({1, 1, fx.lora_rank, fx.unit}), true);
  std::copy(fx.loraB_init.begin(), fx.loraB_init.end(),
            loraB_ref.getData<float>());

  nntrainer::Tensor a_fq_ref = applyFakeQuantWithScales(loraA_ref, a_cd);
  nntrainer::Tensor b_fq_ref = applyFakeQuantWithScales(loraB_ref, b_cd);

  std::vector<float> dy(fx.height * fx.unit);
  for (unsigned int i = 0; i < dy.size(); ++i)
    dy[i] = std::cos(0.61f * i + 0.17f) * 0.3f;
  std::copy(dy.begin(), dy.end(),
            fx.rc.getOutputGradUnsafe(0).getData<float>());

  fx.layer.calcDerivative(fx.rc);
  fx.layer.calcGradient(fx.rc);

  // --- calcDerivative: dL/dx must use a_fq . b_fq (STE), not raw loraA/loraB.
  nntrainer::Tensor lora_contrib_ref(
    nntrainer::TensorDim({1, 1, fx.in_dim, fx.unit}), true);
  a_fq_ref.dot(b_fq_ref, lora_contrib_ref, false, false);
  // base weight is zero, so the effective weight the real calcDerivative
  // adds against is exactly lora_contrib_ref (lora_scaling defaults to 1
  // since lora_alpha was not set).
  nntrainer::Tensor expected_deriv(
    nntrainer::TensorDim({1, 1, fx.height, fx.in_dim}), true);
  nntrainer::Tensor dy_tensor(nntrainer::TensorDim({1, 1, fx.height, fx.unit}),
                              true);
  std::copy(dy.begin(), dy.end(), dy_tensor.getData<float>());
  expected_deriv.dot_deriv_wrt_1(lora_contrib_ref, dy_tensor, false, false);

  const float *actual_deriv = fx.rc.getOutgoingDerivative(0).getData<float>();
  const float *exp_deriv = expected_deriv.getData<float>();
  for (unsigned int i = 0; i < fx.height * fx.in_dim; ++i)
    EXPECT_NEAR(actual_deriv[i], exp_deriv[i], 1e-5f)
      << "calcDerivative mismatch at " << i
      << " -- outgoing derivative does not match the fake-quantized-weight "
         "(STE) formula";

  // --- calcGradient: dL/dloraA and dL/dloraB must use the STE chain
  // through b_fq/a_fq, matching exactly what forwarding() computed.
  nntrainer::Tensor lora_tmp_ref(
    nntrainer::TensorDim({1, 1, fx.height, fx.lora_rank}), true);
  fx.inputs[0].getVariableRef().dot(a_fq_ref, lora_tmp_ref, false, false);

  nntrainer::Tensor expected_djdtmp(
    nntrainer::TensorDim({1, 1, fx.height, fx.lora_rank}), true);
  expected_djdtmp.dot_deriv_wrt_1(b_fq_ref, dy_tensor, false, false);

  nntrainer::Tensor expected_djdla(
    nntrainer::TensorDim({1, 1, fx.in_dim, fx.lora_rank}), true);
  fx.inputs[0].getVariableRef().dot_deriv_wrt_2(expected_djdla, expected_djdtmp,
                                                false, false);

  nntrainer::Tensor expected_djdlb(
    nntrainer::TensorDim({1, 1, fx.lora_rank, fx.unit}), true);
  lora_tmp_ref.dot_deriv_wrt_2(expected_djdlb, dy_tensor, false, false);

  const float *actual_djdla = fx.rc.getWeightGrad(1).getData<float>();
  const float *exp_djdla = expected_djdla.getData<float>();
  for (unsigned int i = 0; i < fx.in_dim * fx.lora_rank; ++i)
    EXPECT_NEAR(actual_djdla[i], exp_djdla[i], 1e-5f)
      << "calcGradient dL/dloraA mismatch at " << i;

  const float *actual_djdlb = fx.rc.getWeightGrad(2).getData<float>();
  const float *exp_djdlb = expected_djdlb.getData<float>();
  for (unsigned int i = 0; i < fx.lora_rank * fx.unit; ++i)
    EXPECT_NEAR(actual_djdlb[i], exp_djdlb[i], 1e-5f)
      << "calcGradient dL/dloraB mismatch at " << i;
}

TEST(FullyConnectedLoraQAT, EMABootstrapsThenBlendsAndFreezesDuringEval) {
  FCLoraQATFixture fx("fc_qat_ema");

  // First training forward: EMA has no history yet, so it must bootstrap
  // directly to the fresh per-channel scale (not blend against zero).
  fx.layer.forwarding(fx.rc, true);
  auto [a_cd_1, b_cd_1] =
    nntrainer::FullyConnectedLayer::getRegisteredChannelScales("fc_qat_ema");
  ASSERT_FALSE(a_cd_1.empty());

  // Bootstrap value: each channel's scale is exactly amax/7 of the
  // channel's initial elements.
  const size_t K = fx.in_dim, N = fx.lora_rank;
  for (size_t n = 0; n < N; ++n) {
    float amax = 0.0f;
    for (size_t k = 0; k < K; ++k)
      amax = std::max(amax, std::abs(fx.loraA_init[k * N + n]));
    EXPECT_FLOAT_EQ(a_cd_1[n], amax / 7.0f) << "bootstrap channel " << n;
  }

  // A second training forward with different loraA values must blend the
  // EMA: d = (1 - momentum) * old + momentum * fresh, momentum = 0.1.
  for (unsigned int i = 0; i < fx.in_dim * fx.lora_rank; ++i)
    fx.weights[1].getVariableRef().getData<float>()[i] =
      std::sin(1.9f * i + 0.4f) * 1.2f;
  fx.layer.forwarding(fx.rc, true);
  auto [a_cd_2, b_cd_2] =
    nntrainer::FullyConnectedLayer::getRegisteredChannelScales("fc_qat_ema");

  // Recompute the expected fresh per-channel scale for the second call and
  // confirm the registry reflects an EMA blend, not a hard overwrite.
  const nntrainer::Tensor &loraA_now = fx.weights[1].getVariableRef();
  const float mom = 0.1f;
  for (size_t n = 0; n < a_cd_2.size(); ++n) {
    float amax = 0.0f;
    for (size_t k = 0; k < K; ++k)
      amax = std::max(amax, std::abs(loraA_now.getValue<float>(k * N + n)));
    const float fresh = amax / 7.0f;
    const float expected_blend = (1.0f - mom) * a_cd_1[n] + mom * fresh;
    EXPECT_FLOAT_EQ(a_cd_2[n], expected_blend)
      << "EMA channel " << n
      << " did not blend as (1-momentum)*old + momentum*fresh";
  }

  // A validation-mode (training=false) forward must NOT update the EMA:
  // the registry should be unchanged, and calling it twice in a row must
  // be bit-identical (no residual state mutation on repeated eval calls).
  fx.layer.forwarding(fx.rc, false);
  auto [a_cd_3, b_cd_3] =
    nntrainer::FullyConnectedLayer::getRegisteredChannelScales("fc_qat_ema");
  ASSERT_EQ(a_cd_3.size(), a_cd_2.size());
  for (size_t n = 0; n < a_cd_3.size(); ++n)
    EXPECT_FLOAT_EQ(a_cd_3[n], a_cd_2[n])
      << "validation-mode forward updated EMA channel " << n;

  fx.layer.forwarding(fx.rc, false);
  auto [a_cd_4, b_cd_4] =
    nntrainer::FullyConnectedLayer::getRegisteredChannelScales("fc_qat_ema");
  for (size_t n = 0; n < a_cd_4.size(); ++n)
    EXPECT_FLOAT_EQ(a_cd_4[n], a_cd_3[n])
      << "repeated validation-mode forward is not idempotent at channel " << n;
}

/**
 * @brief The scale is per output channel and plain FP32: d = amax/7 of the
 *        channel's elements, so the channel's largest magnitude lands
 *        exactly on code 7, and a negative extreme at 0.9x of it rounds to
 *        code -6 (the -8 code stays headroom). No FP16 rounding anywhere --
 *        the saved adapter stores these exact fp32 scales.
 */
TEST(FullyConnectedLoraQAT, ChannelScalesMapTheChannelMaximumToCode7) {
  FCLoraQATFixture fx("fc_qat_channel_scale");
  // a large positive maximum in every channel of loraA, plus a negative
  // extreme at 0.9x of it
  float *a = fx.weights[1].getVariableRef().getData<float>();
  const size_t K = fx.in_dim, N = fx.lora_rank;
  for (size_t i = 0; i < K * N; ++i)
    a[i] = 0.1f * std::sin(0.7f * i);
  for (size_t n = 0; n < N; ++n) {
    a[0 * N + n] = 3.0f + 0.25f * n;
    a[1 * N + n] = -0.9f * (3.0f + 0.25f * n);
  }
  fx.layer.forwarding(fx.rc, true);
  auto [a_cd, b_cd] =
    nntrainer::FullyConnectedLayer::getRegisteredChannelScales(
      "fc_qat_channel_scale");
  ASSERT_EQ(a_cd.size(), N);

  const nntrainer::Tensor a_fq =
    applyFakeQuantWithScales(fx.weights[1].getVariableRef(), a_cd);
  for (size_t n = 0; n < N; ++n) {
    const float d = a_cd[n];
    EXPECT_FLOAT_EQ(d, (3.0f + 0.25f * n) / 7.0f)
      << "channel " << n << " scale is not amax/7";
    const size_t at_pos = 0 * N + n, at_neg = 1 * N + n;
    EXPECT_FLOAT_EQ(a_fq.getValue<float>(at_pos), 7.0f * d)
      << "channel " << n << " maximum is not on code 7";
    EXPECT_NEAR(a_fq.getValue<float>(at_pos), a[at_pos], 2e-3f * a[at_pos])
      << "channel " << n << " maximum is not preserved";
    EXPECT_FLOAT_EQ(a_fq.getValue<float>(at_neg), -6.0f * d)
      << "channel " << n << " negative extreme is not on code -6";
  }
}
