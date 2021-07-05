// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file unittest_layers_fully_connected.cpp
 * @date 5 July 2021
 * @brief Conv2d Layer Test
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <tuple>

#include <gtest/gtest.h>

#include <conv2d_layer.h>
#include <layers_common_tests.h>

auto semantic_conv2d = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::Conv2DLayer>, nntrainer::Conv2DLayer::type,
  {"filters=1", "kernel_size=1,1", "padding=1,1"}, {}, 0, false);

INSTANTIATE_TEST_CASE_P(Convolution2D, LayerSemantics,
                        ::testing::Values(semantic_conv2d));
