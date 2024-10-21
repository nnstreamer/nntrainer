// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file unittest_layers_reduce_mean.cpp
 * @date 25 November 2021
 * @brief Reduce Mean Layer Test
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <tuple>

#include <gtest/gtest.h>

#include <layers_common_tests.h>
#include <reduce_mean_layer.h>

auto semantic_reduce_mean_all = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::ReduceMeanLayer>,
  nntrainer::ReduceMeanLayer::type, {"axis=1"},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 1);

auto semantic_reduce_mean = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::ReduceMeanLayer>,
  nntrainer::ReduceMeanLayer::type, {"axis=1"},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 1);

GTEST_PARAMETER_TEST(ReduceMean, LayerSemantics,
                     ::testing::Values(semantic_reduce_mean,
                                       semantic_reduce_mean_all));
