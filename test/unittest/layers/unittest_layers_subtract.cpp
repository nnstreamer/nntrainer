// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 SeungBaek Hong <sb92.hong@samsung.com>
 *
 * @file unittest_layers_subtract.cpp
 * @date 26 August 2024
 * @brief Subtract Layer Test
 * @see	https://github.com/nnstreamer/nntrainer
 * @author SeungBaek Hong <sb92.hong@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <tuple>

#include <gtest/gtest.h>

#include <layers_common_tests.h>
#include <subtract_layer.h>

auto semantic_subtract = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::SubtractLayer>,
  nntrainer::SubtractLayer::type, {},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 1);

auto semantic_subtract_multi = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::SubtractLayer>,
  nntrainer::SubtractLayer::type, {},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 2);

GTEST_PARAMETER_TEST(Subtract, LayerSemantics,
                     ::testing::Values(semantic_subtract,
                                       semantic_subtract_multi));
