// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Sachin Singh <sachin.3@samsung.com>
 *
 * @file unittest_layers_unsqueeze.cpp
 * @date 08 July 2025
 * @brief Unsqueeze Layer Test
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Sachin Singh <sachin.3@samsung.com>
 * @author Abhinav Dwivedi <abhinav.d@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <tuple>

#include <gtest/gtest.h>

#include <layers_common_tests.h>
#include <unsqueeze_layer.h>

auto semantic_unsqueeze_axis2 = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::UnsqueezeLayer>,
  nntrainer::UnsqueezeLayer::type, {"axis=2"},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 1);
auto semantic_unsqueeze_axis1 = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::UnsqueezeLayer>,
  nntrainer::UnsqueezeLayer::type, {"axis=1"},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 1);
auto semantic_unsqueeze_axis0 = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::UnsqueezeLayer>,
  nntrainer::UnsqueezeLayer::type, {"axis=0"},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 1);

GTEST_PARAMETER_TEST(Unsqueeze, LayerSemantics,
                     ::testing::Values(semantic_unsqueeze_axis0,
                                       semantic_unsqueeze_axis1,
                                       semantic_unsqueeze_axis2));
