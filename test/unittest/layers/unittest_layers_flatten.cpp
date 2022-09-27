// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file unittest_layers_flatten.cpp
 * @date 6 July 2021
 * @brief Flatten Layer Test
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <tuple>

#include <gtest/gtest.h>

#include <flatten_layer.h>
#include <layers_common_tests.h>

auto semantic_flatten = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::FlattenLayer>,
  nntrainer::FlattenLayer::type, {},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 1);

GTEST_PARAMETER_TEST(Flatten, LayerSemantics,
                     ::testing::Values(semantic_flatten));
