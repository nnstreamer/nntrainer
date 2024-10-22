// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 SeungBaek Hong <sb92.hong@samsung.com>
 *
 * @file unittest_layers_add.cpp
 * @date 5 August 2024
 * @brief Add Layer Test
 * @see	https://github.com/nnstreamer/nntrainer
 * @author SeungBaek Hong <sb92.hong@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <tuple>

#include <gtest/gtest.h>

#include <add_layer.h>
#include <layers_common_tests.h>

auto semantic_add = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::AddLayer>, nntrainer::AddLayer::type, {},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 1);

auto semantic_add_multi = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::AddLayer>, nntrainer::AddLayer::type, {},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 2);

GTEST_PARAMETER_TEST(Add, LayerSemantics,
                     ::testing::Values(semantic_add, semantic_add_multi));
