// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file unittest_layers_addition.cpp
 * @date 7 July 2021
 * @brief Addition Layer Test
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <tuple>

#include <gtest/gtest.h>

#include <addition_layer.h>
#include <layers_common_tests.h>

auto semantic_addition = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::AdditionLayer>,
  nntrainer::AdditionLayer::type, {},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 1);

auto semantic_addition_multi = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::AdditionLayer>,
  nntrainer::AdditionLayer::type, {},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 2);

GTEST_PARAMETER_TEST(Addition, LayerSemantics,
                     ::testing::Values(semantic_addition,
                                       semantic_addition_multi));

auto addition_1 = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::AdditionLayer>, {}, "2:5:3:2, 2:5:3:2",
  "addition.nnlayergolden", LayerGoldenTestParamOptions::DEFAULT, "nchw",
  "fp32", "fp32");

auto addition_1_nhwc = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::AdditionLayer>, {}, "2:2:5:3, 2:2:5:3",
  "addition.nnlayergolden", LayerGoldenTestParamOptions::DEFAULT, "nhwc",
  "fp32", "fp32");

GTEST_PARAMETER_TEST(Addition, LayerGoldenTest,
                     ::testing::Values(addition_1, addition_1_nhwc));
