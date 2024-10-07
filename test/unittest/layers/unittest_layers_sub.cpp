// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 SeungBaek Hong <sb92.hong@samsung.com>
 *
 * @file unittest_layers_sub.cpp
 * @date 26 August 2024
 * @brief Sub Layer Test
 * @see	https://github.com/nnstreamer/nntrainer
 * @author SeungBaek Hong <sb92.hong@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <tuple>

#include <gtest/gtest.h>

#include <layers_common_tests.h>
#include <sub_layer.h>

auto semantic_sub = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::SubLayer>, nntrainer::SubLayer::type, {},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 1);

auto semantic_sub_multi = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::SubLayer>, nntrainer::SubLayer::type, {},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 2);

GTEST_PARAMETER_TEST(Sub, LayerSemantics,
                     ::testing::Values(semantic_sub, semantic_sub_multi));
