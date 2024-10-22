// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 SeungBaek Hong <sb92.hong@samsung.com>
 *
 * @file unittest_layers_multiply.cpp
 * @date 30 August 2024
 * @brief Mul Layer Test
 * @see	https://github.com/nnstreamer/nntrainer
 * @author SeungBaek Hong <sb92.hong@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <tuple>

#include <gtest/gtest.h>

#include <layers_common_tests.h>
#include <multiply_layer.h>

auto semantic_multiply = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::MultiplyLayer>,
  nntrainer::MultiplyLayer::type, {},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 1);

auto semantic_multiply_multi = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::MultiplyLayer>,
  nntrainer::MultiplyLayer::type, {},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 2);

GTEST_PARAMETER_TEST(Multiply, LayerSemantics,
                     ::testing::Values(semantic_multiply,
                                       semantic_multiply_multi));
