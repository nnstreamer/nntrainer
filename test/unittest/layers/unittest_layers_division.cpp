// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 SeungBaek Hong <sb92.hong@samsung.com>
 *
 * @file unittest_layers_division.cpp
 * @date 22 July 2024
 * @brief Division Layer Test
 * @see	https://github.com/nnstreamer/nntrainer
 * @author SeungBaek Hong <sb92.hong@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <tuple>

#include <gtest/gtest.h>

#include <division_layer.h>
#include <layers_common_tests.h>

auto semantic_division = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::DivisionLayer>,
  nntrainer::DivisionLayer::type, {},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 1);

auto semantic_division_multi = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::DivisionLayer>,
  nntrainer::DivisionLayer::type, {},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 2);

GTEST_PARAMETER_TEST(Division, LayerSemantics,
                     ::testing::Values(semantic_division,
                                       semantic_division_multi));
