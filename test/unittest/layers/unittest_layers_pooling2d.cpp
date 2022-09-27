// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file unittest_layers_pooling.cpp
 * @date 6 July 2021
 * @brief Pooling2d Layer Test
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <tuple>

#include <gtest/gtest.h>

#include <layers_common_tests.h>
#include <pooling2d_layer.h>

auto semantic_pooling2d_max = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::Pooling2DLayer>,
  nntrainer::Pooling2DLayer::type, {"pooling=max", "pool_size=1,1"},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 1);

auto semantic_pooling2d_avg = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::Pooling2DLayer>,
  nntrainer::Pooling2DLayer::type, {"pooling=average", "pool_size=1,1"},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 1);

auto semantic_pooling2d_global_avg = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::Pooling2DLayer>,
  nntrainer::Pooling2DLayer::type, {"pooling=global_average"},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 1);

auto semantic_pooling2d_global_max = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::Pooling2DLayer>,
  nntrainer::Pooling2DLayer::type, {"pooling=global_max"},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 1);

GTEST_PARAMETER_TEST(Pooling2DMax, LayerSemantics,
                     ::testing::Values(semantic_pooling2d_max,
                                       semantic_pooling2d_avg,
                                       semantic_pooling2d_global_max,
                                       semantic_pooling2d_global_avg));

auto pooling2d_prop = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::Pooling2DLayer>,
  nntrainer::Pooling2DLayer::type, {"pool_size="},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 1);

GTEST_PARAMETER_TEST(Pooling2DMax, LayerPropertySemantics,
                     ::testing::Values(pooling2d_prop));
