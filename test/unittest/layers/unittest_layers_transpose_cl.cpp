// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Niket Agarwal <niket.a@samsung.com>
 *
 * @file   unittest_layers_transpose_cl.cpp
 * @date   31 July 2024
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Niket Agarwal <niket.a@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Transpose Layer Test
 */
#include <tuple>

#include <gtest/gtest.h>

#include <layers_common_tests.h>
#include <transpose_cl.h>

auto semantic_transpose_gpu = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::TransposeLayerCl>,
  nntrainer::TransposeLayerCl::type, {},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 1);

GTEST_PARAMETER_TEST(TransposeGPU, LayerSemanticsGpu,
                     ::testing::Values(semantic_transpose_gpu));

auto transpose_basic_plain_axis0 =
  LayerGoldenTestParamType(nntrainer::createLayer<nntrainer::TransposeLayerCl>,
                           {}, "2:3:3:3", "transpose_axis0.nnlayergolden",
                           LayerGoldenTestParamOptions::SKIP_CALC_DERIV |
                             LayerGoldenTestParamOptions::SKIP_CALC_GRAD |
                             LayerGoldenTestParamOptions::USE_INC_FORWARD,
                           "nchw", "fp32", "fp32");

// auto transpose_basic_plain_axis1 =
//   LayerGoldenTestParamType(nntrainer::createLayer<nntrainer::TransposeLayerCl>,
//                            {}, "2:3:3:3", "transpose_axis1.nnlayergolden",
//                            LayerGoldenTestParamOptions::SKIP_CALC_DERIV |
//                              LayerGoldenTestParamOptions::SKIP_CALC_GRAD |
//                              LayerGoldenTestParamOptions::USE_INC_FORWARD,
//                            "nchw", "fp32", "fp32");

// auto transpose_basic_plain_axis2 =
//   LayerGoldenTestParamType(nntrainer::createLayer<nntrainer::TransposeLayerCl>,
//                            {}, "2:3:3:3", "transpose_axis2.nnlayergolden",
//                            LayerGoldenTestParamOptions::SKIP_CALC_DERIV |
//                              LayerGoldenTestParamOptions::SKIP_CALC_GRAD |
//                              LayerGoldenTestParamOptions::USE_INC_FORWARD,
//                            "nchw", "fp32", "fp32");

GTEST_PARAMETER_TEST(TransposeGPU, LayerGoldenTest,
                     ::testing::Values(transpose_basic_plain_axis0));

#ifdef ENABLE_FP16
auto transpose_basic_plain_w16a16_axis0 =
  LayerGoldenTestParamType(nntrainer::createLayer<nntrainer::TransposeLayerCl>,
                           {}, "2:3:3:3", "transpose_fp16_axis0.nnlayergolden",
                           LayerGoldenTestParamOptions::SKIP_CALC_DERIV |
                             LayerGoldenTestParamOptions::SKIP_CALC_GRAD |
                             LayerGoldenTestParamOptions::USE_INC_FORWARD,
                           "nchw", "fp16", "fp16");

// auto transpose_basic_plain_w16a16_axis1 =
//   LayerGoldenTestParamType(nntrainer::createLayer<nntrainer::TransposeLayerCl>,
//                            {}, "2:3:3:3",
//                            "transpose_fp16_axis1.nnlayergolden",
//                            LayerGoldenTestParamOptions::SKIP_CALC_DERIV |
//                              LayerGoldenTestParamOptions::SKIP_CALC_GRAD |
//                              LayerGoldenTestParamOptions::USE_INC_FORWARD,
//                            "nchw", "fp16", "fp16");

// auto transpose_basic_plain_w16a16_axis2 =
//   LayerGoldenTestParamType(nntrainer::createLayer<nntrainer::TransposeLayerCl>,
//                            {}, "2:3:3:3",
//                            "transpose_fp16_axis2.nnlayergolden",
//                            LayerGoldenTestParamOptions::SKIP_CALC_DERIV |
//                              LayerGoldenTestParamOptions::SKIP_CALC_GRAD |
//                              LayerGoldenTestParamOptions::USE_INC_FORWARD,
//                            "nchw", "fp16", "fp16");

GTEST_PARAMETER_TEST(TransposeGPU16, LayerGoldenTest,
                     ::testing::Values(transpose_basic_plain_w16a16_axis0));
#endif
