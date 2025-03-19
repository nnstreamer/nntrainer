// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 SeungBaek Hong <sb92.hong@samsung.com>
 *
 * @file unittest_layers_cosine.cpp
 * @date 19 March 2025
 * @brief Cosine Layer Test
 * @see	https://github.com/nnstreamer/nntrainer
 * @author SeungBaek Hong <sb92.hong@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <tuple>

#include <gtest/gtest.h>

#include <cosine_layer.h>
#include <layers_common_tests.h>

auto semantic_cosine = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::CosineLayer>, nntrainer::CosineLayer::type,
  {}, LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 1);

auto semantic_cosine_multi = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::CosineLayer>, nntrainer::CosineLayer::type,
  {}, LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 2);

GTEST_PARAMETER_TEST(Cosine, LayerSemantics,
                     ::testing::Values(semantic_cosine, semantic_cosine_multi));
