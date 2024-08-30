// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 SeungBaek Hong <sb92.hong@samsung.com>
 *
 * @file unittest_layers_div.cpp
 * @date 30 August 2024
 * @brief Div Layer Test
 * @see	https://github.com/nnstreamer/nntrainer
 * @author SeungBaek Hong <sb92.hong@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <tuple>

#include <gtest/gtest.h>

#include <div_layer.h>
#include <layers_common_tests.h>

auto semantic_div = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::DivLayer>, nntrainer::DivLayer::type, {},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 1);

auto semantic_div_multi = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::DivLayer>, nntrainer::DivLayer::type, {},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 2);

GTEST_PARAMETER_TEST(Div, LayerSemantics,
                     ::testing::Values(semantic_div, semantic_div_multi));
