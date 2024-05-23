// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Yash Singh <yash.singh@samsung.com>
 *
 * @file unittest_layers_addition_cl.cpp
 * @date 17 May 2024
 * @brief Addition Layer Test
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Yash Singh <yash.singh@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <tuple>

#include <gtest/gtest.h>

#include <addition_layer_cl.h>
#include <layers_common_tests.h>

auto semantic_addition_gpu = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::AdditionLayerCL>,
  nntrainer::AdditionLayerCL::type, {},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 1);

auto semantic_addition_multi_gpu = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::AdditionLayerCL>,
  nntrainer::AdditionLayerCL::type, {},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 2);

GTEST_PARAMETER_TEST(AdditionGPU, LayerSemantics,
                     ::testing::Values(semantic_addition_gpu,
                                       semantic_addition_multi_gpu));

auto addition_w32a32 = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::AdditionLayerCL>, {}, "2:3:3:3,2:3:3:3",
  "added_w32a32.nnlayergolden", LayerGoldenTestParamOptions::DEFAULT, "nchw",
  "fp32", "fp32");

auto addition_w32a32_2 = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::AdditionLayerCL>, {}, "3:4:3:4,3:4:3:4",
  "added_w32a32_2.nnlayergolden", LayerGoldenTestParamOptions::DEFAULT, "nchw",
  "fp32", "fp32");

auto addition_w32a32_3 = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::AdditionLayerCL>, {},
  "20:55:50:55,20:55:50:55", "added_w32a32_3.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp32", "fp32");

GTEST_PARAMETER_TEST(AdditionGPU, LayerGoldenTest,
                     ::testing::Values(addition_w32a32, addition_w32a32_2,
                                       addition_w32a32_3));
