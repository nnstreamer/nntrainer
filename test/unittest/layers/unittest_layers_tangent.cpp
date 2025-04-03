// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 SeungBaek Hong <sb92.hong@samsung.com>
 *
 * @file unittest_layers_tangent.cpp
 * @date 19 March 2025
 * @brief Tangent Layer Test
 * @see	https://github.com/nnstreamer/nntrainer
 * @author SeungBaek Hong <sb92.hong@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <tuple>

#include <gtest/gtest.h>

#include <layers_common_tests.h>
#include <tangent_layer.h>

auto semantic_tangent = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::TangentLayer>,
  nntrainer::TangentLayer::type, {},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 1);

auto semantic_tangent_multi = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::TangentLayer>,
  nntrainer::TangentLayer::type, {},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 2);

GTEST_PARAMETER_TEST(Tangent, LayerSemantics,
                     ::testing::Values(semantic_tangent,
                                       semantic_tangent_multi));
