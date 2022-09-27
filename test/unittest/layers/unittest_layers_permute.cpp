// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file unittest_layers_permute.cpp
 * @date 7 July 2021
 * @brief Permute Layer Test
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <tuple>

#include <gtest/gtest.h>

#include <layers_common_tests.h>
#include <permute_layer.h>

auto semantic_permute = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::PermuteLayer>,
  nntrainer::PermuteLayer::type, {"direction=3,2,1"}, 0, false, 1);

GTEST_PARAMETER_TEST(Permute, LayerSemantics,
                     ::testing::Values(semantic_permute));

auto permute_prop1 = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::PermuteLayer>,
  nntrainer::PermuteLayer::type, {"direction=random_string"},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 1);

auto permute_prop2 = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::PermuteLayer>,
  nntrainer::PermuteLayer::type, {"direction=1,2,4"},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 1);

auto permute_prop3 = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::PermuteLayer>,
  nntrainer::PermuteLayer::type, {"direction=1,2,3,0"},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 1);

auto permute_prop4 = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::PermuteLayer>,
  nntrainer::PermuteLayer::type, {"direction=3:1:2"},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 1);

auto permute_prop5 = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::PermuteLayer>,
  nntrainer::PermuteLayer::type, {"direction=3|1|2"},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 1);

GTEST_PARAMETER_TEST(Permute, LayerPropertySemantics,
                     ::testing::Values(permute_prop1, permute_prop2,
                                       permute_prop3, permute_prop4,
                                       permute_prop5));
