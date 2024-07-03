// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Niket Agarwal <niket.a@samsung.com>
 *
 * @file unittest_layers_concat_cl.cpp
 * @date 2 July 2024
 * @brief Concat Layer Test
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Niket Agarwal <niket.a@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <tuple>

#include <gtest/gtest.h>

#include <concat_cl.h>
#include <layers_common_tests.h>

auto semantic_concat_gpu = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::ConcatLayerCl>,
  nntrainer::ConcatLayerCl::type, {},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 1);

GTEST_PARAMETER_TEST(ConcatGPU, LayerSemanticsGpu,
                     ::testing::Values(semantic_concat_gpu));

auto concat_dim3 = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::ConcatLayerCl>, {"axis=3"},
  "2:3:3:2,2:3:3:3", "concat_dim3.nnlayergolden",
  LayerGoldenTestParamOptions::SKIP_CALC_DERIV, "nchw", "fp32", "fp32");

auto concat_dim2 = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::ConcatLayerCl>, {"axis=2"},
  "2:3:2:3,2:3:3:3", "concat_dim2.nnlayergolden",
  LayerGoldenTestParamOptions::SKIP_CALC_DERIV, "nchw", "fp32", "fp32");

auto concat_dim1 = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::ConcatLayerCl>, {"axis=1"},
  "2:2:3:3,2:3:3:3", "concat_dim1.nnlayergolden",
  LayerGoldenTestParamOptions::SKIP_CALC_DERIV, "nchw", "fp32", "fp32");

GTEST_PARAMETER_TEST(ConcatGPU, LayerGoldenTest,
                     ::testing::Values(concat_dim3, concat_dim2, concat_dim1));

#ifdef ENABLE_FP16
auto concat_dim3_w16a16 = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::ConcatLayerCl>, {"axis=3"},
  "2:3:3:2,2:3:3:3", "concat_dim3_w16a16.nnlayergolden",
  LayerGoldenTestParamOptions::SKIP_CALC_DERIV, "nchw", "fp16", "fp16");

auto concat_dim2_w16a16 = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::ConcatLayerCl>, {"axis=2"},
  "2:3:2:3,2:3:3:3", "concat_dim2_w16a16.nnlayergolden",
  LayerGoldenTestParamOptions::SKIP_CALC_DERIV, "nchw", "fp16", "fp16");

auto concat_dim1_w16a16 = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::ConcatLayerCl>, {"axis=1"},
  "2:2:3:3,2:3:3:3", "concat_dim1_w16a16.nnlayergolden",
  LayerGoldenTestParamOptions::SKIP_CALC_DERIV, "nchw", "fp16", "fp16");

GTEST_PARAMETER_TEST(ConcatGPU16, LayerGoldenTest,
                     ::testing::Values(concat_dim3_w16a16, concat_dim2_w16a16,
                                       concat_dim1_w16a16));
#endif
