// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Thummala Pallavi <t.pallavi@samsung.com>
 *
 * @file unittest_layers_rmsnorm_cl.cpp
 * @date 7 June 2024
 * @brief RMS Norm Layer Test
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Thummala Pallavi <t.pallavi@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <tuple>

#include <gtest/gtest.h>

#include <layers_common_tests.h>
#include <rmsnorm_layer_cl.h>

auto semantic_rms = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::RMSNormLayerCl>,
  nntrainer::RMSNormLayerCl::type, {"epsilon=0.001"},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 1);

GTEST_PARAMETER_TEST(RMSNormGPU, LayerSemanticsGpu,
                     ::testing::Values(semantic_rms));

auto rms_plain_skip_CG = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::RMSNormLayerCl>, {"epsilon=0.001"},
  "2:3:3:3", "rms_normtest.nnlayergolden",
  LayerGoldenTestParamOptions::SKIP_CALC_DERIV |
    LayerGoldenTestParamOptions::SKIP_CALC_GRAD |
    LayerGoldenTestParamOptions::USE_INC_FORWARD,
  "nchw", "fp32", "fp32");

GTEST_PARAMETER_TEST(RMSNormGPU, LayerGoldenTest,
                     ::testing::Values(rms_plain_skip_CG));

#ifdef ENABLE_FP16
auto rms_plain_skip_CG_fp16 = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::RMSNormLayerCl>, {"epsilon=0.001"},
  "2:3:3:3", "rms_normtest_fp16_new.nnlayergolden",
  LayerGoldenTestParamOptions::SKIP_CALC_DERIV |
    LayerGoldenTestParamOptions::SKIP_CALC_GRAD |
    LayerGoldenTestParamOptions::USE_INC_FORWARD,
  "nchw", "fp16", "fp16");

GTEST_PARAMETER_TEST(RMSNormGPU16, LayerGoldenTest,
                     ::testing::Values(rms_plain_skip_CG_fp16));

#endif
