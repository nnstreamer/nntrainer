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
  nntrainer::Pooling2DLayer::type, {"pooling=max", "pool_size=1,1"}, 0, false);

auto semantic_pooling2d_avg =
  LayerSemanticsParamType(nntrainer::createLayer<nntrainer::Pooling2DLayer>,
                          nntrainer::Pooling2DLayer::type,
                          {"pooling=average", "pool_size=1,1"}, 0, false);

auto semantic_pooling2d_global_avg = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::Pooling2DLayer>,
  nntrainer::Pooling2DLayer::type, {"pooling=global_average"}, 0, false);

auto semantic_pooling2d_global_max = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::Pooling2DLayer>,
  nntrainer::Pooling2DLayer::type, {"pooling=global_max"}, 0, false);

INSTANTIATE_TEST_CASE_P(Pooling2DMax, LayerSemantics,
                        ::testing::Values(semantic_pooling2d_max,
                                          semantic_pooling2d_avg,
                                          semantic_pooling2d_global_max,
                                          semantic_pooling2d_global_avg));
