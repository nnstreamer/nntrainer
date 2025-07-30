// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Sumon Nath <sumon.nath@samsung.com>
 *
 * @file unittest_layers_reduce_sum.cpp
 * @date 29 July 2025
 * @brief Reduce Sum Layer Test
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Sumon Nath <sumon.nath@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <tuple>

#include <gtest/gtest.h>

#include <layers_common_tests.h>
#include <reduce_sum_layer.h>

auto semantic_reduce_sum = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::ReduceSumLayer>,
  nntrainer::ReduceSumLayer::type, {"axis=1"},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 1);

GTEST_PARAMETER_TEST(ReduceSum, LayerSemantics,
                     ::testing::Values(semantic_reduce_sum));
