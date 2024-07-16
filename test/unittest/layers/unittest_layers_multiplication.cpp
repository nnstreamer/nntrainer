// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 SeungBaek Hong <sb92.hong@samsung.com>
 *
 * @file unittest_layers_multiplication.cpp
 * @date 22 July 2024
 * @brief Multiplication Layer Test
 * @see	https://github.com/nnstreamer/nntrainer
 * @author SeungBaek Hong <sb92.hong@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <tuple>

#include <gtest/gtest.h>

#include <layers_common_tests.h>
#include <multiplication_layer.h>

auto semantic_multiplication = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::MultiplicationLayer>,
  nntrainer::MultiplicationLayer::type, {},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 1);

auto semantic_multiplication_multi = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::MultiplicationLayer>,
  nntrainer::MultiplicationLayer::type, {},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 2);

GTEST_PARAMETER_TEST(Multiplication, LayerSemantics,
                     ::testing::Values(semantic_multiplication,
                                       semantic_multiplication_multi));
