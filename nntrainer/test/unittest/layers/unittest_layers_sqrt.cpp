// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 SeungBaek Hong <sb92.hong@samsung.com>
 *
 * @file unittest_layers_sqrt.cpp
 * @date 19 March 2025
 * @brief Sqrt Layer Test
 * @see	https://github.com/nnstreamer/nntrainer
 * @author SeungBaek Hong <sb92.hong@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <tuple>

#include <gtest/gtest.h>

#include <layers_common_tests.h>
#include <sqrt_layer.h>

auto semantic_sqrt = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::SQRTLayer>, nntrainer::SQRTLayer::type, {},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 1);

auto semantic_sqrt_multi = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::SQRTLayer>, nntrainer::SQRTLayer::type, {},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 2);

GTEST_PARAMETER_TEST(SQRT, LayerSemantics,
                     ::testing::Values(semantic_sqrt, semantic_sqrt_multi));
