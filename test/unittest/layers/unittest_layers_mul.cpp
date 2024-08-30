// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 SeungBaek Hong <sb92.hong@samsung.com>
 *
 * @file unittest_layers_mul.cpp
 * @date 30 August 2024
 * @brief Mul Layer Test
 * @see	https://github.com/nnstreamer/nntrainer
 * @author SeungBaek Hong <sb92.hong@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <tuple>

#include <gtest/gtest.h>

#include <layers_common_tests.h>
#include <mul_layer.h>

auto semantic_mul = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::MulLayer>, nntrainer::MulLayer::type, {},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 1);

auto semantic_mul_multi = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::MulLayer>, nntrainer::MulLayer::type, {},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 2);

GTEST_PARAMETER_TEST(Mul, LayerSemantics,
                     ::testing::Values(semantic_mul, semantic_mul_multi));
