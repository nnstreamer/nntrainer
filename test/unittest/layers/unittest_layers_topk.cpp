// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Sachin Singh <sachin.3@samsung.com>
 *
 * @file unittest_layers_topk.cpp
 * @date 28 July 2025
 * @brief Topk Layer Test
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Sachin Singh <sachin.3@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <tuple>

#include <gtest/gtest.h>

#include <layers_common_tests.h>
#include <topk_layer.h>

auto semantic_topk_k1 = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::TopkLayer>, nntrainer::TopkLayer::type,
  {"k=1"}, LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 1);

GTEST_PARAMETER_TEST(Topk, LayerSemantics, ::testing::Values(semantic_topk_k1));
