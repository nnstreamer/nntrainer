// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   layer_plugin_rnnt_loss_test.cpp
 * @date   10 June 2021
 * @brief  This file contains the execution part of rnnt loss layer in
 * LayerPlugin example
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */
#include <tuple>

#include <gtest/gtest.h>

#include <layer_plugin_common_test.h>
#include <layers_common_tests.h>
#include <rnnt_loss.h>

INSTANTIATE_TEST_CASE_P(
  RNNTLossLayer, LayerPluginCommonTest,
  ::testing::Values(std::make_tuple("librnnt_loss_layer.so", "rnnt_loss")));

auto semantic_rnnt =
  LayerSemanticsParamType(nntrainer::createLayer<custom::RNNTLossLayer>,
                          custom::RNNTLossLayer::type, {}, 0, false);

INSTANTIATE_TEST_CASE_P(RNNTLossLayer, LayerSemantics,
                        ::testing::Values(semantic_rnnt));
