// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 SeungBaek Hong <sb92.hong@samsung.com>
 *
 * @file unittest_layers_subtraction.cpp
 * @date 22 July 2024
 * @brief Subtraction Layer Test
 * @see	https://github.com/nnstreamer/nntrainer
 * @author SeungBaek Hong <sb92.hong@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <tuple>

#include <gtest/gtest.h>

#include <layers_common_tests.h>
#include <subtraction_layer.h>

auto semantic_subtraction = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::SubtractionLayer>,
  nntrainer::SubtractionLayer::type, {},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 1);

auto semantic_subtraction_multi = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::SubtractionLayer>,
  nntrainer::SubtractionLayer::type, {},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 2);

GTEST_PARAMETER_TEST(Subtraction, LayerSemantics,
                     ::testing::Values(semantic_subtraction,
                                       semantic_subtraction_multi));
