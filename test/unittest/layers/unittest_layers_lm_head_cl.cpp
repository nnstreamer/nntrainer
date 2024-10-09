// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Yash Singh <yash.singh@samsung.com>
 *
 * @file unittest_layers_lm_head_cl.cpp
 * @date 1 Oct 2024
 * @brief LM Head Layer Test
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Yash Singh <yash.singh@samsung.com>
 * @bug No known bugs except for NYI items
 */

#include <gtest/gtest.h>
#include <layers_common_tests.h>
#include <lm_head_layer_cl.h>
#include <tuple>

auto semantic_lm_head_gpu = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::CustomLMHeadLayerCl>,
  nntrainer::CustomLMHeadLayerCl::type, {"unit=1"},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 1);

GTEST_PARAMETER_TEST(LM_HeadGPU, LayerSemanticsGpu,
                     ::testing::Values(semantic_lm_head_gpu));

auto lm_head_gpu = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::CustomLMHeadLayerCl>,
  {"unit=5", "use_vocab_selection=false"}, "3:10:1:1",
  "lm_head_GPU1.nnlayergolden",
  LayerGoldenTestParamOptions::SKIP_CALC_DERIV |
    LayerGoldenTestParamOptions::SKIP_CALC_GRAD |
    LayerGoldenTestParamOptions::USE_INC_FORWARD,
  "nhwc", "fp32", "fp32");

GTEST_PARAMETER_TEST(LM_HeadGPU, LayerGoldenTest,
                     ::testing::Values(lm_head_gpu));

#ifdef ENABLE_FP16
auto lm_head_gpu_w16a16 = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::CustomLMHeadLayerCl>,
  {"unit=5", "use_vocab_selection=false"}, "3:1:1:10",
  "lm_head_GPU1_w16a16.nnlayergolden",
  LayerGoldenTestParamOptions::SKIP_CALC_DERIV |
    LayerGoldenTestParamOptions::SKIP_CALC_GRAD |
    LayerGoldenTestParamOptions::USE_INC_FORWARD,
  "nchw", "fp16", "fp16");

GTEST_PARAMETER_TEST(LM_HeadGPU16, LayerGoldenTest,
                     ::testing::Values(lm_head_gpu_w16a16));
#endif
