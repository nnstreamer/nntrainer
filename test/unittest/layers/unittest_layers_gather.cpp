// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 SeungBaek Hong <sb92.hong@samsung.com>
 *
 * @file unittest_layers_gather.cpp
 * @date 4 April 2025
 * @brief Gather Layer Test
 * @see	https://github.com/nnstreamer/nntrainer
 * @author SeungBaek Hong <sb92.hong@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <tuple>

#include <gtest/gtest.h>

#include <gather_layer.h>
#include <layers_common_tests.h>

auto semantic_gather = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::GatherLayer>, nntrainer::GatherLayer::type,
  {"axis=2"}, LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false,
  2);

GTEST_PARAMETER_TEST(Gather, LayerSemantics,
                     ::testing::Values(semantic_gather));
