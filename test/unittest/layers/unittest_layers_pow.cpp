// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 SeungBaek Hong <sb92.hong@samsung.com>
 *
 * @file unittest_layers_pow.cpp
 * @date 20 Nov 2024
 * @brief Pow Layer Test
 * @see	https://github.com/nnstreamer/nntrainer
 * @author SeungBaek Hong <sb92.hong@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <tuple>

#include <gtest/gtest.h>

#include <layers_common_tests.h>
#include <pow_layer.h>

auto semantic_pow = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::PowLayer>, nntrainer::PowLayer::type,
  {"exponent=3"}, LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT,
  false, 1);

auto semantic_pow_multi = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::PowLayer>, nntrainer::PowLayer::type,
  {"exponent=3"}, LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT,
  false, 2);

GTEST_PARAMETER_TEST(Pow, LayerSemantics,
                     ::testing::Values(semantic_pow, semantic_pow_multi));
