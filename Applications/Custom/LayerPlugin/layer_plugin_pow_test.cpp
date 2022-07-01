// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   layer_plugin_pow_test.cpp
 * @date   26 January 2021
 * @brief  This file contains the execution part of pow layer in LayerPlugin
 * example
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */
#include <tuple>

#include <gtest/gtest.h>

#include <layer_plugin_common_test.h>
#include <layers_common_tests.h>
#include <pow.h>

GTEST_PARAMETER_TEST(PowLayer, LayerPluginCommonTest,
                     ::testing::Values(std::make_tuple("libpow_layer.so",
                                                       "pow")));

auto semantic_pow =
  LayerSemanticsParamType(nntrainer::createLayer<custom::PowLayer>,
                          custom::PowLayer::type, {}, 0, false, 1);

GTEST_PARAMETER_TEST(PowLayer, LayerSemantics, ::testing::Values(semantic_pow));
