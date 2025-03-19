// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 SeungBaek Hong <sb92.hong@samsung.com>
 *
 * @file unittest_layers_sine.cpp
 * @date 19 March 2025
 * @brief Sine Layer Test
 * @see	https://github.com/nnstreamer/nntrainer
 * @author SeungBaek Hong <sb92.hong@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <tuple>

#include <gtest/gtest.h>

#include <layers_common_tests.h>
#include <sine_layer.h>

auto semantic_sine = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::SineLayer>, nntrainer::SineLayer::type, {},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 1);

auto semantic_sine_multi = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::SineLayer>, nntrainer::SineLayer::type, {},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 2);

GTEST_PARAMETER_TEST(Sine, LayerSemantics,
                     ::testing::Values(semantic_sine, semantic_sine_multi));
